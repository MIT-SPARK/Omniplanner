"""Omniplanner node with runtime constraint plumbing and mission state.

Thin subclass of :class:`omniplanner_ros.omniplanner_node.OmniPlannerRos` that
adds:

* a persistent set of runtime constraints (e.g. ``forbidden-poi`` /
  ``forbidden-edge``) on ``~/constraints``. Plugins read
  ``node.active_constraints``; ``MultiRobotPddlConstrainedPlannerRos`` merges
  them with per-message constraints before grounding.
* mission state, so a replan only plans what is left. Executors report every
  finished action on ``/action_done``; a
  :class:`omniplanner.world_state.WorldStateTracker` turns those into visited
  places, inspected objects and -- when the map does not track it -- held
  objects. Plugins ask :meth:`world_state_for` for the state to plan from.
  ``~/reset_mission_state`` forgets visits and inspections for a new mission.
* putting down first: an object a robot holds that the new goal no longer
  mentions is put down where the robot stands before its new plan starts.
* plan validity under scene changes: changes applied to the map arrive on
  ``~/scene_changes`` (the scene change writer's applied topic). Those our own
  executors made are ignored; the rest are judged, after a short settle, by
  :func:`omniplanner.plan_validity.affecting_changes` against what the current
  plans still act on and what the goal asks for. A change that invalidates the
  plan -- or a failed pick, up to ``max_pick_retries`` -- is sent to the goal
  manager on ``~/replan_request`` once this node holds a map that contains it.

This module does **not** override ``register_plugin`` or duplicate the
plan-handling logic in ``OmniPlannerRos``; the plugin-side ``pddl_callback``
and the upstream ``plugin.on_plan_compiled`` hook handle the per-call work.
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from dsg_pddl.pddl_grounding import ConstraintFact
from heracles_ros_interfaces.msg import SceneChangeMsg
from omniplanner.plan_validity import affecting_changes, plan_dependencies
from omniplanner.world_state import (
    MAP_METADATA_KEY,
    WorldStateTracker,
    kept_objects,
    split_held,
)
from omniplanner_msgs.msg import ConstraintList
from rclpy.executors import MultiThreadedExecutor
from robot_executor_interface.action_descriptions import ActionSequence, Place
from robot_executor_msgs.msg import ActionDoneMsg
from std_msgs.msg import String
from std_srvs.srv import Trigger

from omniplanner_ros.omniplanner_node import OmniPlannerRos


class OmniPlannerRepairRos(OmniPlannerRos):
    """OmniPlannerRos plus persistent constraints and mission state."""

    def __init__(self):
        # Initialize state before super().__init__() so plugins registered
        # during the parent constructor can read it safely if they need to.
        self._constraints_lock = threading.Lock()
        self.active_constraints: list[ConstraintFact] = []

        self._world_lock = threading.Lock()
        self.world_tracker = WorldStateTracker()
        # {robot -> objects} the plan being built must put down first; set by
        # world_state_for, consumed by finalize_plans for the same request.
        self._pending_release: dict = {}

        # Plan validity: what the published plans still act on, the goal they
        # were made for, scene changes waiting to be judged, and pick failures.
        self._plan_lock = threading.Lock()
        self._plan_deps: dict = {}
        self._planning_goal = ""
        self._active_goal = ""
        self._pending_changes: list = []
        self._pending_version = 0
        self._pick_failures: dict = {}

        super().__init__()

        # A completed path counts as visiting what it passes this close to (and
        # any object whose nearest place it reaches; see symbols_near).
        self.declare_parameter("visited_radius_m", 1.0)
        self.world_tracker.visited_radius_m = float(
            self.get_parameter("visited_radius_m").value
        )

        self.create_subscription(
            ConstraintList,
            "~/constraints",
            self._constraints_callback,
            10,
        )
        self.create_subscription(
            ActionDoneMsg, "/action_done", self._action_done_callback, 50
        )
        self.create_service(
            Trigger, "~/reset_mission_state", self._reset_mission_state_callback
        )

        # Scene changes, as applied to the map. A plan they invalidate is
        # replanned through the goal manager (~/replan_request).
        self.declare_parameter("move_threshold_m", 0.5)
        self.declare_parameter("change_settle_s", 1.0)
        self.declare_parameter("max_pick_retries", 2)
        self._move_threshold_m = float(self.get_parameter("move_threshold_m").value)
        self._change_settle_s = float(self.get_parameter("change_settle_s").value)
        self._max_pick_retries = int(self.get_parameter("max_pick_retries").value)
        self._replan_pub = self.create_publisher(String, "~/replan_request", 10)
        self.create_subscription(
            SceneChangeMsg, "~/scene_changes", self._scene_changes_callback, 10
        )
        self._settle_timer = None
        self.get_logger().info(
            "OmniPlannerRepairRos listening for persistent constraints on "
            "~/constraints and mission progress on /action_done"
        )

    def _constraints_callback(self, msg: ConstraintList) -> None:
        new_constraints = [
            ConstraintFact(predicate=c.predicate, symbols=list(c.symbols))
            for c in msg.constraints
        ]
        with self._constraints_lock:
            self.active_constraints = new_constraints
        self.get_logger().info(
            f"Updated persistent constraints: {len(new_constraints)} fact(s)"
        )

    # ------------- mission state -------------

    def _current_dsg(self):
        with self.dsg_lock:
            return self.dsg_last

    def _action_done_callback(self, msg: ActionDoneMsg) -> None:
        points = [(p.x, p.y) for p in msg.points]
        with self._world_lock:
            new = self.world_tracker.record(
                msg.action_type,
                msg.success,
                msg.robot_name,
                object_id=msg.object_id,
                points=points,
                dsg=self._current_dsg(),
            )
        if new:
            self.get_logger().info(
                f"{msg.robot_name} {msg.action_type.lower()} done: {sorted(new)}"
            )
        obj = (msg.object_id or "").lower()
        if msg.action_type in ("PICK", "GAZE") and msg.success and obj:
            with self._plan_lock:
                self._plan_deps.pop(obj, None)  # done: a later move no longer matters
            self._pick_failures.pop(obj, None)
        elif msg.action_type == "PICK" and not msg.success and obj:
            tries = self._pick_failures.get(obj, 0) + 1
            self._pick_failures[obj] = tries
            if tries <= self._max_pick_retries:
                self._request_replan(
                    f"pick of {obj} by {msg.robot_name} failed "
                    f"(attempt {tries} of {self._max_pick_retries})"
                )
            else:
                self.get_logger().error(
                    f"pick of {obj} failed {tries} times; not replanning for it again"
                )

    # ------------- plan validity under scene changes -------------

    def _scene_changes_callback(self, msg: SceneChangeMsg) -> None:
        if msg.source.startswith("executor/"):
            return  # our own pick or place: the plan already accounts for it
        changes = [
            {
                "kind": c.kind,
                "symbol": c.symbol,
                "old": (c.old_position.x, c.old_position.y),
                "new": (c.new_position.x, c.new_position.y),
            }
            for c in msg.changes
        ]
        with self._plan_lock:
            self._pending_changes += changes
            self._pending_version = max(self._pending_version, msg.map_version)
            if self._settle_timer is None:
                # A detector reports a burst; judge it once, after it settles.
                self._settle_timer = self.create_timer(
                    self._change_settle_s, self._judge_changes
                )
        self.get_logger().info(
            f"{msg.source}: {len(changes)} scene change(s) at map_version "
            f"{msg.map_version}; judging after {self._change_settle_s:.1f} s"
        )

    def _judge_changes(self) -> None:
        with self._plan_lock:
            self._settle_timer.cancel()
            self.destroy_timer(self._settle_timer)
            self._settle_timer = None
            changes, self._pending_changes = self._pending_changes, []
            version, self._pending_version = self._pending_version, 0
            deps, goal = dict(self._plan_deps), self._active_goal
        with self._world_lock:
            state = self.world_tracker.snapshot(self._current_dsg())
        affected = affecting_changes(
            changes, deps, goal, state, move_threshold_m=self._move_threshold_m
        )
        if not affected:
            self.get_logger().info(
                f"scene change does not affect the plan: "
                f"{sorted({c['symbol'] for c in changes})}"
            )
            return
        reason = "scene changed: " + "; ".join(r for _, r in affected)
        self._wait_for_map_then_replan(version, reason)

    def _map_version(self) -> int:
        dsg = self._current_dsg()
        try:
            state = dsg.metadata.get().get(MAP_METADATA_KEY, {})
            return int(state.get("map_version") or 0)
        except Exception:
            return 0

    def _wait_for_map_then_replan(self, version, reason, waited=0.0):
        """Replan only once the planner holds a map with the change in it."""
        if self._map_version() >= version or waited >= 10.0:
            if waited >= 10.0:
                self.get_logger().warning(
                    f"map_version {version} not received after 10 s; replanning anyway"
                )
            self._request_replan(reason)
            return
        timer = None

        def again():
            timer.cancel()
            self.destroy_timer(timer)
            self._wait_for_map_then_replan(version, reason, waited + 0.2)

        timer = self.create_timer(0.2, again)

    def _request_replan(self, reason: str) -> None:
        self.get_logger().warning(f"plan invalidated: {reason}; requesting a replan")
        self._replan_pub.publish(String(data=reason))

    def _reset_mission_state_callback(self, request, response):
        with self._world_lock:
            self.world_tracker.reset()
        response.success = True
        response.message = "forgot visited places and inspected objects"
        self.get_logger().info(f"Mission state reset: {response.message}")
        return response

    def world_state_for(self, pddl_goal: str):
        """The state a plan for this goal should start from.

        Objects held but not wanted by the goal are left out -- they are put
        down before the plan runs (see finalize_plans) -- so the planner plans
        with those hands empty.
        """
        dsg = self._current_dsg()
        with self._world_lock:
            state = self.world_tracker.snapshot(dsg)
        planning, release = split_held(state, kept_objects(pddl_goal))
        self._pending_release = release
        self._planning_goal = pddl_goal
        self.get_logger().info(
            f"Planning from mission state: {len(planning.visited)} visited, "
            f"{len(planning.inspected)} inspected, holding={planning.holding}"
            + (f", putting down first: {release}" if release else "")
        )
        return planning

    def finalize_plans(self, plan_dict, robot_poses):
        plan_dict = self._put_down_first(plan_dict, robot_poses)
        # What these plans act on, for judging later scene changes.
        with self._plan_lock:
            self._plan_deps = plan_dependencies(plan_dict.values())
            self._active_goal = self._planning_goal
        return plan_dict

    def _put_down_first(self, plan_dict, robot_poses):
        release, self._pending_release = self._pending_release, {}
        if not release:
            return plan_dict
        plan_id = next((p.plan_id for p in plan_dict.values()), "")
        for robot, objects in release.items():
            name = next((n for n in self.robot_adaptors if n.lower() == robot), robot)
            pose = robot_poses.get(name)
            if pose is None:
                self.get_logger().error(
                    f"No pose for {name}: cannot put down {objects} first"
                )
                continue
            here = np.array([float(pose[0]), float(pose[1]), float(pose[2])])
            put_down = [
                Place(
                    frame=self.dsg_frame,
                    object_class="",
                    robot_point=here,
                    object_point=here,
                    object_id=obj,
                )
                for obj in objects
            ]
            plan = plan_dict.get(name)
            if plan is None:
                plan_dict[name] = ActionSequence(
                    plan_id=plan_id, robot_name=name, actions=put_down
                )
            else:
                plan.actions[:0] = put_down
            self.get_logger().info(f"{name} puts down {objects} before its new plan")
        return plan_dict


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OmniPlannerRepairRos()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
