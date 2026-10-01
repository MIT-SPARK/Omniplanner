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

This module does **not** override ``register_plugin`` or duplicate the
plan-handling logic in ``OmniPlannerRos``; the plugin-side ``pddl_callback``
and the upstream ``plugin.on_plan_compiled`` hook handle the per-call work.
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from dsg_pddl.pddl_grounding import ConstraintFact
from omniplanner.world_state import WorldStateTracker, kept_objects, split_held
from omniplanner_msgs.msg import ConstraintList
from rclpy.executors import MultiThreadedExecutor
from robot_executor_interface.action_descriptions import ActionSequence, Place
from robot_executor_msgs.msg import ActionDoneMsg
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
        self.get_logger().info(
            f"Planning from mission state: {len(planning.visited)} visited, "
            f"{len(planning.inspected)} inspected, holding={planning.holding}"
            + (f", putting down first: {release}" if release else "")
        )
        return planning

    def finalize_plans(self, plan_dict, robot_poses):
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
