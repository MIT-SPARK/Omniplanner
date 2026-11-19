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
  A plan that has finished, or whose robot was stopped (by the goal manager
  or the operator, on ``executor_stop_topic``; a Husky by an empty Path), is
  no longer judged: each plan keeps its own goal, and a change is only held
  against the plans still running.
  A scene change that invalidates the plan pauses every robot in it at once,
  on ``executor_pause_topic``, so none drives on while the map catches up.
  The replanned sequence releases each robot; if no plan is found they stay
  stopped and the goal manager tells the operator.
* held replans: with ``confirm_replans`` (the default) a replan caused by
  change detection -- a scene change whose source starts with ``awcd/`` --
  waits for the operator. Every other replan (a map edit from ``sget/`` or
  ``console/``, a map reload, a cut route, a failed pick) goes out at once.
  The robots are paused as above, and the robots paused are the question's
  robots. The question is latched on
  ``~/replan_pending``: "" when nothing is held, else JSON ``{id, robots,
  reasons, reason, source, stopped, t}`` (stopped: the held Huskies, which
  have no Path to resume). Nothing is held when no one subscribes to it. ``~/replan_decision`` answers it with JSON
  ``{id, decision}`` ("confirm" or "decline"); a decision for another id, or
  with none, is ignored. Confirm sends one ``~/replan_request`` naming every
  held change, and those still waiting for their map; decline sends
  nothing. A held change is never lost: any other replan carries it (in
  the same request when it is a scene change's, else right after), and so
  does a new plan that leaves no robot held for it, if its goal still names
  the object (one that does not answers the change). A robot given a new plan, or
  whose plan ends, leaves the question (same id); a question left with no
  robot is cleared, and a further change makes a new question (new id). A
  change whose pause did not reach every robot is not held: it replans at
  once. An empty Path for a held Husky is the hold's own stop: the Husky
  stays held, and its plan judged, until a decline ends it. A change asked
  after the operator answered pauses its robots again. A held robot is not
  flagged by the progress watchdog, and a held Spot is
  reported as progress on its path_status, so the goal manager does not take
  it for stalled.
* the map version this node plans on, latched on ``~/map_version``, so the
  goal manager replans a goal the running plan covers once the map changed.
* routes under map edits: each new map is checked against the part of every
  robot's route still ahead. A place on it that is gone, or two places it
  drives between that are no longer connected (e.g. an SGET edit), pauses the
  robots and replans, so they go around instead of following the old path.
  Every map change is logged with that verdict.
* map epochs: heracles writes ``map_epoch`` next to ``map_version``, and a
  reload of the map database changes the epoch and restarts the version. The
  version alone can go back or repeat, so this node counts (epoch, version)
  changes itself and publishes that count on ``~/map_version``: it only ever
  goes up. A new epoch is a whole new map: the route and the objects the plan
  acts on are judged against it afresh.
* platforms: a Spot is paused on ``executor_pause_topic``; a Husky (a Path
  plan) has no executor to pause and is stopped with an empty Path on its
  plan topic. Only the robots a message can reach are reported as stopped.
* progress: a robot given a plan that neither moves nor reports an action for
  ``progress_timeout_s`` is reported, in the log and on ``~/progress_alert``.
  A Husky reports nothing, so its progress is its pose reaching its Path's
  poses, which also records the places it visited. That progress goes out on
  ``/<robot>/omniplanner_node/path_status`` (JSON: plan_id, status progress,
  done or stalled, reached, steps), so the goal manager knows when a Husky has
  finished. A Spot moving goes out there too, as progress: one long follow
  reports no action until it ends.
* busy robots: a goal with a ``busy-robots`` constraint (see
  :data:`omniplanner.world_state.BUSY_ROBOTS`) adds work to the running
  mission. It is planned on free robots only; a busy robot, or one holding
  an object, is sent nothing -- not even an empty plan -- unless the goal
  names it. Without that constraint a goal is a repair of the mission, and
  any robot may be interrupted; one holding an object still places it or
  puts it down first.
* robot-scoped goals: a goal with a ``use-robots`` constraint is planned for
  those robots only, and every other robot is left alone the same way: no
  plan (not even an empty one), and what it holds is not put down.

This module does **not** override ``register_plugin`` or duplicate the
plan-handling logic in ``OmniPlannerRos``; the plugin-side ``pddl_callback``
and the upstream ``plugin.on_plan_compiled`` hook handle the per-call work.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from typing import Optional

import numpy as np
import rclpy
import spark_dsg
from dsg_pddl.dsg_pddl_grounding_multirobot import assigned_robots
from dsg_pddl.pddl_grounding import ConstraintFact
from heracles_ros_interfaces.msg import SceneChangeMsg
from nav_msgs.msg import Path
from omniplanner.plan_validity import (
    MOVED,
    REMOVED,
    affecting_changes,
    broken_route,
    plan_dependencies,
    route_places,
)
from omniplanner.world_state import (
    MAP_METADATA_KEY,
    WorldStateTracker,
    busy_robots,
    goal_tokens,
    kept_objects,
    new_mission,
    split_held,
    unavailable_robots,
    visits_to_forget,
)
from omniplanner_msgs.msg import ConstraintList
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from robot_executor_interface.action_descriptions import ActionSequence, Place
from robot_executor_msgs.msg import ActionDoneMsg
from std_msgs.msg import Bool, Int64, String
from std_srvs.srv import Trigger

from omniplanner_ros.omniplanner_node import OmniPlannerRos, plan_is_empty

# robot_type values whose plan is a nav_msgs/Path (a Husky running phoenix).
PATH_ROBOT_TYPES = {"husky", "phoenix"}


def plan_ids(plan_dict):
    """{robot: plan_id} of compiled plans. A phoenix (husky) plan is a
    nav_msgs/Path, which has no plan_id: it maps to ""."""
    return {r: getattr(p, "plan_id", "") or "" for r, p in plan_dict.items()}


def joined_goals(goals) -> str:
    """Several plans' goals as one goal ("" for none), each goal once."""
    goals = list(dict.fromkeys(g for g in goals if g))
    return goals[0] if len(goals) == 1 else f"(and {' '.join(goals)})" if goals else ""


# ActionDoneMsg.header.frame_id of a failed pick the operator chose not to try
# again (spot_tools: the answer "give up" to the after-pick question).
OPERATOR_GAVE_UP = "operator_gave_up"

# A scene change from change detection (AWCD) is the one replan the operator
# confirms (see confirm_replans).
HELD_SOURCE_PREFIX = "awcd/"
# How often a held Spot is reported as progress on its path_status.
HELD_PROGRESS_EVERY_S = 10.0
# The lead of a scene change's replan reason.
SCENE_LEAD = "scene changed: "


def scene_reason(reasons) -> str:
    """Scene change reasons as one, with one lead: the goal manager's
    removed_symbols reads every part after it."""
    return SCENE_LEAD + "; ".join(r.removeprefix(SCENE_LEAD) for r in reasons)


def _note(msg) -> str:
    """How an ActionDoneMsg says its action ended ("" for most)."""
    return getattr(getattr(msg, "header", None), "frame_id", "") or ""


def last_waypoint_reached(near) -> Optional[int]:
    """How far along the rest of its path a robot has come: the index of the
    last waypoint of the first stretch it is near (near[i]: waypoint i ahead
    is within reach of the robot), or None when it is near none.

    Only the first stretch counts. A path that comes back over its own ground
    (out to a place and home again, or twice past the same corner) has later
    waypoints at the spot the robot is standing on: being near those says
    nothing about the waypoints in between, which it has yet to drive.
    Waypoints before the stretch were cut short, and count as passed.
    """
    idx = np.flatnonzero(np.asarray(near, dtype=bool))
    if len(idx) == 0:
        return None
    gaps = np.flatnonzero(np.diff(idx) > 1)
    return int(idx[gaps[0]] if len(gaps) else idx[-1])


# How far from the robot's track since the last waypoint it reached a waypoint it
# skipped may lie, in visit radii: a robot rounds corners inside them.
TRACK_TOLERANCE_RADII = 1.5
# The track is kept at this spacing (metres), and at most this many points.
TRACK_STEP_M = 0.2
TRACK_MAX_POINTS = 4000


def distance_to_track(points, track) -> np.ndarray:
    """Each point's distance (metres, in the plane) to the polyline ``track``,
    less the length of the nearest segment: between two pose updates the robot
    may have driven any route about that long, not the chord (pose updates can
    be sparse, or stop for a while). A robot standing still has short chords.
    The first segment joins the waypoint the track starts at to the robot, not
    two poses: it gets no such allowance."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    tr = np.asarray(track, dtype=float).reshape(-1, 2)
    if len(tr) == 1:
        return np.hypot(*(pts - tr[0]).T)
    a, b = tr[:-1], tr[1:]
    ab = b - a
    ll = np.maximum((ab**2).sum(axis=1), 1e-12)
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(axis=2) / ll, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d = np.hypot(*(pts[:, None, :] - proj).transpose(2, 0, 1))
    slack = np.sqrt(ll)
    slack[0] = 0.0
    return np.maximum(d - slack[None], 0.0).min(axis=1)


def waypoints_passed_on_the_way(ahead_points, track, radius) -> np.ndarray:
    """Which of the waypoints ahead the robot can be counted at: waypoint j
    only if every waypoint before it (after the last one reached) lies along
    the robot's track since then.

    A path that comes back over its own ground has a later waypoint at the
    spot the robot stands on. Standing there is not having driven the path to
    it: the waypoints between (the far end of an out-and-back) are nowhere
    near where the robot has been, so that waypoint is out of reach, however
    near it is. A waypoint skipped by cutting a corner lies along the track.
    """
    n = len(ahead_points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    on_track = distance_to_track(ahead_points, track) <= TRACK_TOLERANCE_RADII * radius
    before_ok = np.concatenate([[True], np.cumprod(on_track[:-1]).astype(bool)])
    return before_ok


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
        # Robots the plan being built must leave alone (an additional task),
        # set by world_state_for like _pending_release; those it did leave
        # alone keep their running plan, which this node goes on watching.
        self._pending_busy: set = set()
        self._left_alone: set = set()
        self._running_plans: dict = {}

        # Plan validity: what the published plans still act on, the goal they
        # were made for, scene changes waiting to be judged, and pick failures.
        self._plan_lock = threading.Lock()
        self._plan_deps: dict = {}
        self._plan_robots: list = []
        self._pause_pubs: dict = {}
        self._planning_goal = ""
        self._active_goal = ""
        # {robot -> the goal its running plan was made for}; _active_goal is
        # these, joined. A plan that finished or was stopped drops out.
        self._plan_goals: dict = {}
        self._pending_changes: list = []
        self._pending_version = 0
        self._pick_failures: dict = {}
        # Routes still ahead: the places each Follow passes, the plan each
        # robot runs, how many of its actions are done, and the map version
        # the route was last checked on.
        self._route: dict = {}
        self._plan_ids: dict = {}
        self._actions_done: dict = {}
        self._route_version = None
        self._published_version = None
        self._version_pub = None
        # The (map_epoch, map_version) of the map held now, how many times it
        # has changed (what ~/map_version publishes), the epoch a burst of
        # scene changes arrived in, and the versions scene changes announced.
        self._map_key = None
        self._map_generation = 0
        self._pending_epoch = ""
        self._announced: set = set()
        # {robot -> what was dispatched to it and when it last made progress},
        # for the progress watchdog and a Husky's visits.
        self._dispatched: dict = {}
        # Held replans (see confirm_replans): the question asked, the changes
        # still waiting for the map before they are asked, and when this node
        # stopped each Husky itself. See _hold_state.
        self._hold_state()

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
        self.declare_parameter(
            "executor_pause_topic", "/{robot}/spot_executor_node/pause"
        )
        self._move_threshold_m = float(self.get_parameter("move_threshold_m").value)
        self._change_settle_s = float(self.get_parameter("change_settle_s").value)
        self._max_pick_retries = int(self.get_parameter("max_pick_retries").value)
        self._executor_pause_topic = str(
            self.get_parameter("executor_pause_topic").value
        )
        # A robot that is stopped (by the goal manager when no plan can be
        # found, or by the operator) is no longer running its plan.
        self.declare_parameter(
            "executor_stop_topic", "/{robot}/spot_executor_node/stop"
        )
        stop_topic = str(self.get_parameter("executor_stop_topic").value)
        for robot, adaptor in self.robot_adaptors.items():
            if self._is_path_robot(robot):
                self.create_subscription(
                    Path,
                    adaptor.plan_topic,
                    lambda msg, r=robot: msg.poses or self._stop_seen(r),
                    1,
                )
            else:
                self.create_subscription(
                    Bool,
                    stop_topic.format(robot=robot),
                    lambda msg, r=robot: msg.data and self._stop_seen(r),
                    10,
                )
        self._replan_pub = self.create_publisher(String, "~/replan_request", 10)
        # confirm_replans: a replan caused by change detection (a scene change
        # from an awcd/ source) waits for the operator's decision on
        # ~/replan_decision; the question is on ~/replan_pending. Map edits,
        # reloads, cut routes and failed picks replan at once either way, so
        # nothing is ever held while change detection is off.
        self.declare_parameter("confirm_replans", True)
        self._confirm_replans = bool(self.get_parameter("confirm_replans").value)
        self._pending_pub = self.create_publisher(
            String,
            "~/replan_pending",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._pending_pub.publish(String(data=""))
        self.create_subscription(
            String, "~/replan_decision", self._replan_decision_callback, 10
        )
        self.create_subscription(
            SceneChangeMsg, "~/scene_changes", self._scene_changes_callback, 10
        )
        self._settle_timer = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._version_pub = self.create_publisher(Int64, "~/map_version", latched)
        # The map's epoch, which changes when the database is reloaded: the goal
        # manager forgets what was achieved on the old map (~/map_version only
        # goes up, so it cannot tell a reload from an edit).
        self._epoch_pub = self.create_publisher(String, "~/map_epoch", latched)
        self._publish_map_version()
        # A map version no scene change announced within this long is warned
        # about: some edit changed the map without telling anyone (S1).
        self.declare_parameter("announce_grace_s", 3.0)
        self._announce_grace_s = float(self.get_parameter("announce_grace_s").value)

        # A dispatched robot that neither moves this far nor reports an action
        # for progress_timeout_s is reported (<= 0 turns the watchdog off).
        self.declare_parameter("progress_timeout_s", 60.0)
        self.declare_parameter("progress_move_m", 0.5)
        self._progress_timeout_s = float(self.get_parameter("progress_timeout_s").value)
        self._progress_move_m = float(self.get_parameter("progress_move_m").value)
        self._progress_pub = self.create_publisher(String, "~/progress_alert", 10)
        self._path_status_pubs = {}
        self.create_timer(1.0, self._progress_tick)
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
        with self._plan_lock:
            if msg.plan_id and msg.plan_id == self._plan_ids.get(msg.robot_name):
                done = self._actions_done.get(msg.robot_name, 0) + 1
                self._actions_done[msg.robot_name] = done
                sent = self._dispatched.get(msg.robot_name)
                if sent is not None:
                    sent["last_progress"] = self._now_s()
                    if done >= sent["steps"]:
                        # finished; accomplished only if the last action succeeded
                        self._plan_over(msg.robot_name, stopped=not msg.success)
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
            if _note(msg) == OPERATOR_GAVE_UP:
                # The operator watched the pick fail and chose not to try
                # again; the robot ended its action sequence. Replanning would
                # send it straight back to the same pick.
                self._pick_failures.pop(obj, None)
                with self._plan_lock:
                    if msg.plan_id and msg.plan_id == self._plan_ids.get(
                        msg.robot_name
                    ):
                        self._plan_over(msg.robot_name, stopped=True)
                self.get_logger().warning(
                    f"pick of {obj} by {msg.robot_name} was given up by the "
                    "operator: not replanning; its plan is over"
                )
                return
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

    def _plan_over(self, robot: str, stopped: bool = False) -> None:
        """A robot's plan finished or was stopped (call under _plan_lock): it
        is not watched any more, and no later scene change is held against it
        or its goal. It leaves a held replan's question too.

        Only a plan that ran to its end accomplished its goal, which is when
        its robot's visits stop being remembered (see visits_to_forget). A
        stop does not -- every replan stops a Husky first -- and neither does
        a pick the operator gave up."""
        self._release_held({robot}, "its plan is over")
        goals = self.__dict__.setdefault("_visit_goals", {})
        goal = next(
            (g for r, g in self._plan_goals.items() if r.lower() == robot.lower()), None
        )
        if robot in self._running_plans and goal is not None:
            # an empty plan (a robot left out of a replan) has no goal: it
            # changes nothing about what the robot accomplished before
            goals[robot.lower()] = (goal, not stopped)
        plan = self._running_plans.pop(robot, None)
        self._plan_goals.pop(robot, None)
        self._dispatched.pop(robot, None)
        self._route.pop(robot, None)
        others = plan_dependencies(self._running_plans.values())
        for obj in plan_dependencies([plan]):
            if obj not in others:
                self._plan_deps.pop(obj, None)
        self._plan_robots = sorted(self._running_plans)
        self._active_goal = joined_goals(self._plan_goals.values())

    def _stop_seen(self, robot: str) -> None:
        if self._is_path_robot(robot) and self._is_held(robot):
            # An empty Path for a held Husky is the hold's stop (this node's
            # own, seen back, or one sent since: it is stopped either way). It
            # stays held and its plan stays judged, so a further change joins
            # the question; a decline ends its plan.
            return
        with self._plan_lock:
            running = robot in self._running_plans
            self._plan_over(robot, stopped=True)
        if running:
            self.get_logger().info(
                f"{robot} was stopped: its plan is over and is no longer watched"
            )

    def _scene_changes_callback(self, msg: SceneChangeMsg) -> None:
        if msg.source.startswith("executor/"):
            return  # our own pick or place: the plan already accounts for it
        # Only change detection's replans wait for the operator.
        held = bool(getattr(self, "_confirm_replans", False)) and (
            msg.source.startswith(HELD_SOURCE_PREFIX)
        )
        changes = [
            {
                "kind": c.kind,
                "symbol": c.symbol,
                "old": (c.old_position.x, c.old_position.y),
                "new": (c.new_position.x, c.new_position.y),
                "source": msg.source,
                "held": held,
            }
            for c in msg.changes
        ]
        with self._plan_lock:
            self._announced.add(msg.map_version)
            if not self._pending_changes:
                self._pending_epoch = self._map_state()[0]
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
            epoch = self._pending_epoch
            deps, goal = dict(self._plan_deps), self._active_goal
            robots = list(self._plan_robots)
        with self._world_lock:
            state = self.world_tracker.snapshot(self._current_dsg())
        # Changes from change detection wait for the operator (see
        # confirm_replans), the rest replan at once, as without it: judged
        # apart, so that each replan names only its own changes.
        now = [c for c in changes if not c.get("held")]
        later = [c for c in changes if c.get("held")]
        reasons = {}
        for key, group in (("now", now), ("held", later)):
            affected = group and affecting_changes(
                group, deps, goal, state, move_threshold_m=self._move_threshold_m
            )
            if affected:
                reasons[key] = "scene changed: " + "; ".join(r for _, r in affected)
        if not reasons:
            self.get_logger().info(
                f"scene change does not affect the plan: "
                f"{sorted({c['symbol'] for c in changes})}"
            )
            return
        # The robots held are those paused now: a Husky's empty Path, seen
        # back, takes it out of _plan_robots before the map arrives.
        token = None
        pub = getattr(self, "_pending_pub", None)
        if "held" in reasons and not (pub and pub.get_subscription_count()):
            self._replan_held_now(reasons, "nothing listens on ~/replan_pending")
        if "held" in reasons:
            token = self._hold_until_asked(
                robots, later[-1].get("source", ""), reasons["held"]
            )
        reached = self._pause_robots(robots, "; ".join(reasons.values()))
        if token is not None and set(robots) - set(reached):
            # A robot the pause did not reach drives on: nothing to hold.
            with self._hold_state()["lock"]:
                self._hold_state()["waits"].pop(token, None)
            token = None
            missed = ", ".join(sorted(set(robots) - set(reached)))
            self._replan_held_now(reasons, f"the pause did not reach {missed}")
        if "now" in reasons:
            self._wait_for_map_then_replan(version, reasons["now"], epoch=epoch)
        if token is not None:
            self._wait_for_map_then_replan(
                version, reasons["held"], epoch=epoch, hold=token
            )

    def _replan_held_now(self, reasons, why: str) -> None:
        """Replan reasons["held"] at once, with any immediate one."""
        held = reasons.pop("held")
        self.get_logger().warning(f"not holding {held!r} ({why}): replanning")
        reasons["now"] = scene_reason(
            [reasons["now"], held] if "now" in reasons else [held]
        )

    def _is_path_robot(self, robot: str) -> bool:
        """A Husky: its plan is a nav_msgs/Path, and it has no executor to pause."""
        adaptor = (getattr(self, "robot_adaptors", None) or {}).get(robot)
        if adaptor is None:
            return False
        return getattr(adaptor, "PLAN_MSG", None) is Path or (
            str(getattr(adaptor, "robot_type", "")).lower() in PATH_ROBOT_TYPES
        )

    def _pause_robots(self, robots, reason: str) -> list:
        """Stop the robots in the plan now, before waiting for the map.

        A Spot is paused; a Husky is stopped with an empty Path (it can only
        be given a new one). Returns the robots reached: a robot whose topic
        has no subscriber is reported as not reached, not as stopped.
        """
        paused, stopped, missed = [], [], []
        for robot in robots:
            if self._is_path_robot(robot):
                adaptor = self.robot_adaptors[robot]
                stop = Path()
                stop.header.frame_id = self.dsg_frame or ""
                stop.header.stamp = self.get_clock().now().to_msg()
                adaptor.publish_plan(stop)
                pub, done = getattr(adaptor, "plan_pub", None), stopped
                with self._plan_lock:
                    self._dispatched.pop(robot, None)
            else:
                pub, done = self._pause_pub_for(robot), paused
                pub.publish(Bool(data=True))
            count = getattr(pub, "get_subscription_count", None)
            (done if count is None or count() > 0 else missed).append(robot)
        if paused:
            self.get_logger().warning(f"pausing {', '.join(paused)}: {reason}")
        if stopped:
            self.get_logger().warning(
                f"stopping {', '.join(stopped)} (empty path; a Husky cannot "
                f"pause): {reason}"
            )
        if missed:
            self.get_logger().error(
                f"could not reach {', '.join(missed)} to stop it (nothing "
                f"subscribes to its pause or plan topic): {reason}"
            )
        return paused + stopped

    def _pause_pub_for(self, robot: str):
        # Made with the plan, not at pause time: a brand-new publisher's first
        # message can go out before the executor is matched, and be lost.
        pub = self._pause_pubs.get(robot)
        if pub is None:
            topic = self._executor_pause_topic.format(robot=robot)
            pub = self.create_publisher(Bool, topic, 10)
            self._pause_pubs[robot] = pub
        return pub

    def dsg_callback(self, msg):
        super().dsg_callback(msg)
        change = self._note_map_state()
        self._publish_map_version()
        if change is None:
            return
        old, new, reloaded = change
        verdict = self._check_routes()
        if reloaded and not verdict.startswith("route affected"):
            verdict += "; " + self._judge_reload()
        if old is None:
            what = f"first map: map_version {new[1]}"
        elif reloaded:
            what = (
                f"map reloaded (epoch {old[0] or '?'} -> {new[0] or '?'}): "
                f"map_version {old[1]} -> {new[1]}"
            )
        else:
            what = f"map_version {old[1]} -> {new[1]}"
        self.get_logger().info(f"{what} (generation {self._map_generation}): {verdict}")
        if old is not None and not reloaded and new[1] != old[1]:
            self._expect_announcement(new[1], verdict)

    def _note_map_state(self):
        """(old key, new key, reloaded) if the map's (epoch, version) changed."""
        key = self._map_state()
        with self._plan_lock:
            old = self._map_key
            if key == old:
                return None
            self._map_key = key
            self._map_generation += 1
            reloaded = old is not None and key[0] != old[0]
            if reloaded:
                # A whole new map: the route is judged against it afresh, and
                # versions announced in the old epoch mean nothing now.
                self._route_version = None
                self._announced = set()
        return old, key, reloaded

    def _expect_announcement(self, version: int, verdict: str) -> None:
        """Warn if no scene change message announces this map version soon."""
        timer = None

        def check():
            timer.cancel()
            self.destroy_timer(timer)
            with self._plan_lock:
                announced = version in self._announced
                current = self._map_key
            if not announced and current is not None and current[1] == version:
                self.get_logger().warning(
                    f"map_version moved to {version} with no scene change message "
                    f"for it (an edit nobody announced, e.g. an SGET edit); "
                    f"route verdict: {verdict}"
                )

        timer = self.create_timer(self._announce_grace_s, check)

    def _judge_reload(self) -> str:
        """After a reload, judge what the plan acts on against the new map."""
        with self._plan_lock:
            deps, goal = dict(self._plan_deps), self._active_goal
            robots = list(self._plan_robots)
        if not deps:
            return "no objects to recheck"
        dsg = self._current_dsg()
        where = {}
        try:
            for n in dsg.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
                where[n.id.str(True).lower()] = tuple(n.attributes.position[:2])
        except Exception:
            pass
        changes = [
            {"kind": MOVED, "symbol": sym, "old": xy, "new": where[sym]}
            if sym in where
            else {"kind": REMOVED, "symbol": sym, "old": xy, "new": None}
            for sym, (_, xy) in deps.items()
        ]
        with self._world_lock:
            state = self.world_tracker.snapshot(dsg)
        affected = affecting_changes(
            changes, deps, goal, state, move_threshold_m=self._move_threshold_m
        )
        if not affected:
            return f"objects the plan acts on unchanged: {sorted(deps)}"
        reason = "map reloaded: " + "; ".join(r for _, r in affected)
        self._pause_robots(robots, reason)
        self._request_replan(reason)
        return f"objects affected: {[s for s, _ in affected]}"

    @staticmethod
    def _mesh_places(dsg) -> dict:
        """{symbol -> node id} of the places in a map."""
        if dsg is None:
            return {}
        layer = dsg.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
        return {n.id.str(True).lower(): n.id.value for n in layer.nodes}

    def _place_finder(self, dsg):
        """place_at((x, y)): the place a plan's path point is on, or None."""
        if dsg is None:
            return lambda xy: None
        layer = dsg.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
        nodes = list(layer.nodes)
        if not nodes:
            return lambda xy: None
        xy = np.array([n.attributes.position[:2] for n in nodes])
        names = [n.id.str(True).lower() for n in nodes]

        def place_at(p):
            d = np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1])
            i = int(np.argmin(d))
            return names[i] if d[i] < 0.05 else None

        return place_at

    def _check_routes(self) -> str:
        """Replan if the new map cuts the route still ahead of a robot.

        Returns the verdict, for the log.
        """
        version = self._map_version()
        with self._plan_lock:
            if not self._route:
                return "no route to watch"
            if version == self._route_version:
                return "route already checked on this map"
            self._route_version = version
            route, done = dict(self._route), dict(self._actions_done)
            robots = list(self._plan_robots)
        dsg = self._current_dsg()
        places = self._mesh_places(dsg)
        broken = broken_route(
            route,
            done,
            has_place=lambda s: s in places,
            connected=lambda a, b: dsg.has_edge(places[a], places[b]),
        )
        if not broken:
            watched = {p for legs in route.values() for _, ps in legs for p in ps if p}
            return f"route not affected ({len(watched)} place(s) watched)"
        with self._plan_lock:
            self._route = {}  # judged; the new plan brings its own route
        reason = "map changed: " + "; ".join(why for _, why in broken)
        self._pause_robots(robots, reason)
        self._request_replan(reason)
        return f"route affected: {reason}"

    def _publish_map_version(self) -> None:
        if self._version_pub is None:  # a map before __init__ finished
            return
        version = self._map_version()
        if version != self._published_version:
            self._published_version = version
            self._version_pub.publish(Int64(data=version))
        epoch_pub = getattr(self, "_epoch_pub", None)
        if epoch_pub is None:
            return
        epoch = self._map_state()[0]
        if epoch and epoch != getattr(self, "_published_epoch", ""):
            self._published_epoch = epoch
            epoch_pub.publish(String(data=epoch))

    def _map_version(self) -> int:
        """How many times the map's (epoch, version) has changed: goes only up,
        where heracles' own version restarts on a reload."""
        return self._map_generation

    def _map_state(self):
        """(map_epoch, map_version) of the map held now. An older heracles
        writes no epoch: ""."""
        dsg = self._current_dsg()
        if dsg is None:
            return ("", 0)
        try:
            state = dsg.metadata.get().get(MAP_METADATA_KEY, {}) or {}
            return (
                str(state.get("map_epoch") or ""),
                int(state.get("map_version") or 0),
            )
        except Exception as exc:
            self.get_logger().warning(f"cannot read the map version: {exc}")
            return ("", 0)

    def _wait_for_map_then_replan(
        self, version, reason, waited=0.0, epoch=None, hold=None
    ):
        """Replan only once the planner holds a map with the change in it.

        version is heracles' own, from the scene change; it is compared only
        within the epoch the change arrived in. A map of another epoch is a
        reload, which has whatever the change did in it or is newer still.

        With hold (a token from _hold_until_asked) the replan is not sent but
        asked of the operator; once none of its robots is held any more (a
        new plan, or their plans ended) there is nothing to ask, and the wait
        ends.
        """
        if hold is not None and hold not in self._hold_state()["waits"]:
            self.get_logger().info(
                f"not asking about {reason!r}: none of its robots waits any more"
            )
            return
        now_epoch, now_version = self._map_state()
        reached = now_version >= version or (epoch is not None and now_epoch != epoch)
        if reached or waited >= 10.0:
            if waited >= 10.0:
                self.get_logger().warning(
                    f"map_version {version} not received after 10 s; replanning anyway"
                )
            if hold is None:
                self._request_replan(reason)
            else:
                self._ask_operator(hold, reason)
            return
        timer = None

        def again():
            timer.cancel()
            self.destroy_timer(timer)
            self._wait_for_map_then_replan(version, reason, waited + 0.2, epoch, hold)

        timer = self.create_timer(0.2, again)

    def _request_replan(self, reason: str) -> None:
        # A held change is never lost: the replan carries it, and what it
        # held is held no more.
        held = self._take_held() if hasattr(self, "_take_held") else []
        if held and reason.startswith(SCENE_LEAD):
            reason, held = scene_reason([reason, *held]), []
        self.get_logger().warning(f"plan invalidated: {reason}; requesting a replan")
        self._replan_pub.publish(String(data=reason))
        if held:  # another lead: the held changes go right after
            reason = scene_reason(held)
            self.get_logger().warning(f"held: {reason}; requesting a replan")
            self._replan_pub.publish(String(data=reason))

    # ------------- held replans (confirm_replans) -------------

    def _hold_state(self) -> dict:
        """The held replans' state, made on first use: lock (an RLock; take
        it after _plan_lock, never before), question (the one asked, or
        None), waits ({token: {robots, source, decided}}: changes held until
        the map has them, then asked) and decided (how many questions the
        operator has answered)."""
        state = self.__dict__.get("_hold")
        if state is None:
            state = self.__dict__.setdefault(
                "_hold",
                {
                    "lock": threading.RLock(),
                    "question": None,
                    "waits": {},
                    "decided": 0,
                },
            )
        return state

    def _publish_question(self, question) -> None:
        pub = getattr(self, "_pending_pub", None)
        if pub is not None:
            pub.publish(String(data="" if question is None else json.dumps(question)))

    def _hold_until_asked(self, robots, source: str, reason: str):
        """Hold the robots being paused for a change detection replan; the
        token to ask with once the map has the change (None: no robot)."""
        robots = set(robots)
        if not robots:
            self.get_logger().info("change detection replan: no robot to hold")
            return None
        token = uuid.uuid4().hex
        hold = self._hold_state()
        with hold["lock"]:
            hold["waits"][token] = {
                "robots": robots,
                "source": source,
                "reason": reason,
                "decided": hold["decided"],
            }
        return token

    def _ask_operator(self, token: str, reason: str) -> None:
        """Ask the operator to confirm the held replan: a new question (new
        id) with the reasons and robots already asked about, and these."""
        hold = self._hold_state()
        with hold["lock"]:
            wait = hold["waits"].get(token)
            answered = wait is not None and wait["decided"] != hold["decided"]
        if answered:
            # The operator answered a question about these robots since they
            # were paused, and may have resumed them: they are held paused.
            self._pause_robots(sorted(wait["robots"]), reason)
        with hold["lock"]:
            wait = hold["waits"].pop(token, None)
            if not wait or not wait["robots"]:
                self.get_logger().info(
                    f"not asking about {reason!r}: none of its robots waits any more"
                )
                return
            old = hold["question"] or {"robots": [], "reasons": []}
            reasons = old["reasons"] + [reason]
            robots = sorted(set(old["robots"]) | wait["robots"])
            question = {
                "id": uuid.uuid4().hex,
                "robots": robots,
                "reasons": reasons,
                "reason": scene_reason(reasons),
                "source": wait["source"],
                "stopped": [r for r in robots if self._is_path_robot(r)],
                "t": time.time(),
            }
            hold["question"] = question
            self._publish_question(question)
        self.get_logger().warning(
            f"plan invalidated: {question['reason']}; the replan of "
            f"{', '.join(question['robots'])} waits for the operator "
            f"(question {question['id'][:8]})"
        )

    def _release_held(self, robots, why: str, replan=False) -> None:
        """These robots are held no more (a new plan, or their plan ended):
        they leave the question (same id) and the changes waiting for the
        map. A question with no robot left is cleared. With replan (a new
        plan) the changes no robot is held for any more are replanned, not
        dropped."""
        robots, lost = set(robots), []
        hold = self._hold_state()
        with hold["lock"]:
            for token, wait in list(hold["waits"].items()):
                wait["robots"] -= robots
                if not wait["robots"]:
                    lost.append(hold["waits"].pop(token)["reason"])
            question = hold["question"]
            if question is not None and robots & set(question["robots"]):
                left = [r for r in question["robots"] if r not in robots]
                stopped = [r for r in question["stopped"] if r in left]
                hold["question"] = (
                    {**question, "robots": left, "stopped": stopped} if left else None
                )
                self._publish_question(hold["question"])
                lost = ([] if left else question["reasons"]) + lost
            else:
                question = None
        if replan and lost:
            # Each part of a reason starts with the object it is about. The
            # goal just planned answers a part whose object it no longer names
            # (e.g. the console's new goal without the removed object).
            goal = set(re.findall(r"[^\s()]+", (self._planning_goal or "").lower()))
            parts = [p for r in lost for p in r.removeprefix(SCENE_LEAD).split("; ")]
            still = [p for p in parts if p.split(" ", 1)[0] in goal]
            if len(still) < len(parts):
                self.get_logger().info(
                    "held changes answered by the goal just planned: "
                    + "; ".join(p for p in parts if p not in still)
                )
            if still:
                self._request_replan(scene_reason(still))
        if question is None:
            return
        self.get_logger().info(
            f"held replan {question['id'][:8]}: {', '.join(sorted(robots))} "
            f"left it ({why})" + ("" if left else "; nothing is held any more")
        )

    def _take_held(self) -> list:
        """Every held reason (the question's, then those waiting for the
        map), held no more."""
        hold = self._hold_state()
        with hold["lock"]:
            question, hold["question"] = hold["question"], None
            reasons = (question["reasons"] if question else []) + [
                w["reason"] for w in hold["waits"].values()
            ]
            hold["waits"].clear()
            if question is not None:
                self._publish_question(None)
        return reasons

    def _is_held(self, robot: str) -> bool:
        hold = self._hold_state()
        with hold["lock"]:
            question = hold["question"]
            return (question is not None and robot in question["robots"]) or any(
                robot in w["robots"] for w in hold["waits"].values()
            )

    def _replan_decision_callback(self, msg: String) -> None:
        """The operator's answer, JSON {id, decision}. Only an answer to the
        question asked now counts."""
        try:
            data = json.loads(msg.data)
            qid, decision = str(data["id"]), str(data["decision"]).strip().lower()
        except Exception:
            self.get_logger().warning(
                f"replan decision {msg.data!r} ignored: not JSON with id and decision"
            )
            return
        if decision not in ("confirm", "decline"):
            self.get_logger().warning(f"unknown replan decision {msg.data!r}")
            return
        hold = self._hold_state()
        with hold["lock"]:
            question = hold["question"]
            if question is None or question["id"] != qid:
                now = question["id"][:8] if question else "none"
                self.get_logger().warning(
                    f"replan {decision} for question {qid[:8]} ignored: the "
                    f"question asked now is {now}"
                )
                return
            hold["decided"] += 1
            if decision == "confirm":
                # with the changes still waiting for their map
                reasons = self._take_held()
            else:
                hold["question"] = None
                self._publish_question(None)
        if decision == "decline":
            # A held Husky was stopped with an empty Path: nothing resumes it.
            with self._plan_lock:
                for robot in question["robots"]:
                    if self._is_path_robot(robot):
                        self._plan_over(robot, stopped=True)
        self.get_logger().info(
            f"replan {decision}d by the operator (question {qid[:8]}): "
            f"{question['reason']}"
            + ("" if decision == "confirm" else "; nothing is sent")
        )
        if decision == "confirm":
            self._request_replan(scene_reason(reasons))

    def _reset_mission_state_callback(self, request, response):
        with self._world_lock:
            self.world_tracker.reset()
        response.success = True
        response.message = "forgot visited places and inspected objects"
        self.get_logger().info(f"Mission state reset: {response.message}")
        return response

    def world_state_for(self, pddl_goal: str, constraints=None):
        """The state a plan for this goal should start from.

        Objects held but not wanted by the goal are left out -- they are put
        down before the plan runs (see finalize_plans) -- so the planner plans
        with those hands empty.

        With a busy-robots constraint the goal is an additional task: the busy
        robots, and any robot holding an object, are left alone (the grounding
        leaves them out, finalize_plans sends them nothing) unless the goal
        names them; a named one still puts down what it holds first.
        """
        dsg = self._current_dsg()
        assigned = assigned_robots(constraints)
        with self._plan_lock:
            running = {r.lower() for r in getattr(self, "_running_plans", None) or {}}
            goals = self.__dict__.setdefault("_visit_goals", {})
            plan_goals = {r.lower(): g for r, g in self._plan_goals.items()}
            # running: a non-empty plan (an empty one, for a robot left out of a
            # replan, has no goal and runs nothing)
            running = {r for r in running if r in plan_goals}
            for r in running:  # mid-plan: its goal is the one it is running
                goals[r] = (plan_goals[r], False)
            fleet = {r.lower() for r in (getattr(self, "robot_adaptors", None) or {})}
            fleet |= set(goals) | running
            # a robot stopped short is given another goal only by a goal that
            # names it (or limits the plan to it)
            tokens = set(re.findall(r"[^\s()]+", (pddl_goal or "").lower()))
            given = (assigned or set()) | (fleet & tokens)
            over = visits_to_forget(goals, running, pddl_goal, given, fleet)
            for r in over:
                goals.pop(r, None)
        with self._world_lock:
            # A robot's visits are the history of its goal. They are kept
            # until the goal is accomplished -- through every replan and stop
            # on the way -- and forgotten for that robot once it is. A goal
            # sent when every robot is idle (new-mission) needs nothing more:
            # each robot has either accomplished its goal, or is being sent on
            # with it, or is being given another.
            tracked = {*self.world_tracker.visited_by, *self.world_tracker.inspected_by}
            if new_mission(constraints):
                # every robot is idle: visits no goal accounts for go too
                over |= tracked - set(goals)
            forgot = self.world_tracker.forget_visits(keep=tracked - over)
            if new_mission(constraints):
                self.world_tracker.forget_unclaimed()
            if forgot:
                self.get_logger().info(
                    "forgot the visits and inspections of "
                    f"{', '.join(sorted(forgot))}: "
                    + (
                        "a new mission"
                        if new_mission(constraints)
                        else "goal accomplished or dropped"
                    )
                )
            state = self.world_tracker.snapshot(dsg)
        busy = unavailable_robots(constraints, state, goal_tokens(pddl_goal))
        # A use-robots goal is theirs alone: the rest of the fleet is left
        # alone, and keeps holding what it holds.
        outside = set()
        if assigned is not None:
            outside = {
                r.lower() for r in (getattr(self, "robot_adaptors", None) or {})
            } - assigned
        kept = kept_objects(pddl_goal) | {
            o for r in outside for o in state.holding.get(r, [])
        }
        planning, release = split_held(state, kept)
        planning.cannot_manipulate = {
            r
            for r in (getattr(self, "robot_adaptors", None) or {})
            if self._is_path_robot(r)
        }
        release = {r: objs for r, objs in release.items() if r not in busy | outside}
        self._pending_release = release
        self._pending_busy = busy | outside
        self._planning_goal = pddl_goal
        additional = busy_robots(constraints) is not None
        self.get_logger().info(
            f"Planning from mission state: {len(planning.visited)} visited, "
            f"{len(planning.inspected)} inspected, holding={planning.holding}"
            + (f", putting down first: {release}" if release else "")
            + (
                f"; additional task, leaving busy robots alone: {sorted(busy)}"
                if additional
                else ""
            )
            + (f"; use-robots: leaving {sorted(outside)} alone" if outside else "")
        )
        return planning

    def finalize_plans(self, plan_dict, robot_poses):
        plan_dict = self._leave_busy_alone(plan_dict)
        plan_dict = self._put_down_first(plan_dict, robot_poses)
        plan_dict = self._keep_carrying(plan_dict)
        left_alone, self._left_alone = self._left_alone, set()
        if all(plan_is_empty(p) for p in plan_dict.values()):
            return plan_dict  # nothing is dispatched: the running plans stand
        with self._plan_lock:
            interrupted = sorted(set(plan_dict) & set(self._dispatched))
            # Robots an additional task left alone go on with their plans,
            # which are still watched alongside the new ones.
            kept = {
                r: self._running_plans[r]
                for r in left_alone
                if r in self._running_plans and r not in plan_dict
            }
        if interrupted:
            self.get_logger().warning(
                f"interrupting {', '.join(interrupted)} (unfinished plans) for "
                f"the new plan of {self._planning_goal!r}"
            )
        running = {**kept, **plan_dict}
        # Robots given a new plan are held no more; the others stay held.
        if hasattr(self, "_release_held"):  # a stand-in node holds nothing
            self._release_held(plan_dict, "given a new plan", replan=True)
        # What these plans act on, for judging later scene changes.
        with self._plan_lock:
            self._running_plans = running
            self._plan_deps = plan_dependencies(running.values())
            self._plan_robots = sorted(running)
            # Each plan keeps the goal it was made for; an empty plan has none.
            self._plan_goals = {
                **{r: g for r, g in self._plan_goals.items() if r in kept},
                **{
                    r: self._planning_goal
                    for r, p in plan_dict.items()
                    if not plan_is_empty(p)
                },
            }
            self._active_goal = joined_goals(self._plan_goals.values())
            self._route = route_places(running, self._place_finder(self._current_dsg()))
            self._plan_ids = plan_ids(running)
            self._actions_done = {
                r: n for r, n in self._actions_done.items() if r in kept
            }
            self._route_version = self._map_version()
            self._watch_progress(plan_dict, robot_poses, keep=kept)
        for robot in self._plan_robots:
            if not self._is_path_robot(robot):
                self._pause_pub_for(robot)
        watched = {
            r: len({p for _, places in legs for p in places if p})
            for r, legs in self._route.items()
        }
        self.get_logger().info(f"watching route places for map edits: {watched}")
        return plan_dict

    def _leave_busy_alone(self, plan_dict):
        """An additional task sends a busy robot nothing, not even an empty
        plan, which would preempt what it is doing; nor does a goal send
        anything to a robot its use-robots leaves out."""
        busy, self._pending_busy = self._pending_busy, set()
        left = sorted(r for r in plan_dict if r.lower() in busy)
        # A busy robot the planner left out of the problem has no entry in
        # plan_dict at all: it is left alone just the same, and its running
        # plan must stay watched.
        with self._plan_lock:
            unplanned = {r for r in self._running_plans if r.lower() in busy}
        self._left_alone = set(left) | unplanned
        if not left:
            return plan_dict
        self.get_logger().info(
            f"not sending {', '.join(left)} anything (busy, or not one of the "
            "goal's use-robots)"
        )
        return {r: p for r, p in plan_dict.items() if r.lower() not in busy}

    def _keep_carrying(self, plan_dict):
        """Never send an empty plan to a robot holding an object: it would stop
        mid-carry with the object in hand. (_put_down_first gives it a
        put-down whenever it can; this is when it could not, e.g. no pose.)"""
        with self._world_lock:
            holding = self.world_tracker.snapshot(self._current_dsg()).holding
        keep = {}
        for robot, plan in plan_dict.items():
            held = holding.get(robot.lower())
            if held and plan_is_empty(plan):
                self.get_logger().warning(
                    f"not sending {robot} an empty plan: it holds {held}"
                )
                self._left_alone.add(robot)
                continue
            keep[robot] = plan
        return keep if len(keep) < len(plan_dict) else plan_dict

    def _put_down_first(self, plan_dict, robot_poses):
        release, self._pending_release = self._pending_release, {}
        if not release:
            return plan_dict
        plan_id = next(
            (p.plan_id for p in plan_dict.values() if isinstance(p, ActionSequence)),
            "",
        ) or str(uuid.uuid4())
        for robot, objects in release.items():
            name = next((n for n in self.robot_adaptors if n.lower() == robot), robot)
            if not isinstance(
                plan_dict.get(name, ActionSequence(plan_id, name, [])), ActionSequence
            ):
                self.get_logger().error(
                    f"{name} holds {objects}, but its plan is not an action "
                    "sequence: cannot put them down first"
                )
                continue
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

    # ------------- progress -------------

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _watch_progress(self, plan_dict, robot_poses, keep=()) -> None:
        """Start watching each robot given a plan (under _plan_lock); robots
        in keep go on with their running plan, watched as before."""
        now = self._now_s()
        self._dispatched = {r: d for r, d in self._dispatched.items() if r in keep}
        for robot, plan in plan_dict.items():
            if plan_is_empty(plan):
                continue
            path = None
            if not hasattr(plan, "actions"):
                path = np.array(
                    [[p.pose.position.x, p.pose.position.y] for p in plan.poses]
                )
            if robot not in self._path_status_pubs:
                # Made with the plan: a brand-new publisher's first message
                # can go out before the goal manager is matched, and be lost.
                self._path_status_pubs[robot] = self.create_publisher(
                    String, f"/{robot}/omniplanner_node/path_status", 10
                )
            pose = robot_poses.get(robot)
            start = None if pose is None else np.asarray(pose[:2], dtype=float)
            self._dispatched[robot] = {
                "plan_id": getattr(plan, "plan_id", "") or "(path)",
                "since": now,
                "last_progress": now,
                "pose": start,
                "warned": False,
                "path": path,
                "reached": -1,
                "steps": len(plan.actions) if path is None else len(path),
                # where the robot has been since the last waypoint it reached
                # (see waypoints_passed_on_the_way): it starts at the path's
                # start, which the planner put where the robot is
                "track": None
                if path is None or not len(path)
                else [path[0]] + ([] if start is None else [start]),
            }

    def _progress_tick(self) -> None:
        with self._plan_lock:
            robots = list(self._dispatched)
        if not robots or self.dsg_frame is None:
            return
        for robot in robots:
            if robot in self.robot_adaptors:
                self._check_progress(robot, self.robot_pose(robot))

    def _check_progress(self, robot, pose) -> None:
        """Fold a pose in: movement, a Husky's waypoints, and the watchdog."""
        now = self._now_s()
        visited_points = []
        advanced = moved = False
        with self._plan_lock:
            sent = self._dispatched.get(robot)
            if sent is None:
                return
            if pose is not None:
                xy = np.asarray(pose[:2], dtype=float)
                if sent["pose"] is None:
                    sent["pose"] = xy
                elif np.hypot(*(xy - sent["pose"])) > self._progress_move_m:
                    sent["pose"], sent["last_progress"] = xy, now
                    moved = True
                path = sent["path"]
                if path is not None and len(path):
                    track = sent["track"]
                    if np.hypot(*(xy - track[-1])) >= TRACK_STEP_M:
                        track.append(xy)
                        del track[: max(0, len(track) - TRACK_MAX_POINTS)]
                    radius = self.world_tracker.visited_radius_m
                    rest = path[sent["reached"] + 1 :]
                    near = np.hypot(*(rest - xy).T) <= radius
                    if near.any():
                        upto = int(np.flatnonzero(near)[-1]) + 1
                        ok = np.zeros(len(rest), dtype=bool)
                        ok[:upto] = waypoints_passed_on_the_way(
                            rest[:upto], track + [xy], radius
                        )
                        last = last_waypoint_reached(near & ok)
                    else:
                        last = None
                    if last is not None:
                        reached = sent["reached"] + 1 + last
                        visited_points = path[sent["reached"] + 1 : reached + 1]
                        sent["reached"], sent["last_progress"] = reached, now
                        sent["track"] = [path[reached], xy]
                        self._actions_done[robot] = reached
                        advanced = True
            finished = sent["path"] is not None and sent["reached"] >= sent["steps"] - 1
            if finished:
                self._plan_over(robot)
            # A robot held for the operator waits, it does not stall: its time
            # without progress starts once it is let go. A held Spot is
            # reported as progress, so the goal manager does not stall it.
            held = not finished and self._is_held(robot)
            ping = False
            if held:
                sent["last_progress"], sent["warned"] = now, False
                last = sent.get("held_reported")
                ping = last is None or now - last >= HELD_PROGRESS_EVERY_S
                if ping:
                    sent["held_reported"] = now
            stalled = (
                not finished
                and self._progress_timeout_s > 0
                and not sent["warned"]
                and now - sent["last_progress"] > self._progress_timeout_s
            )
            if stalled:
                sent["warned"] = True
        if len(visited_points):
            with self._world_lock:
                new = self.world_tracker.record(
                    "FOLLOW",
                    True,
                    robot,
                    points=visited_points,
                    dsg=self._current_dsg(),
                )
            if new:
                self.get_logger().info(f"{robot} passed: {sorted(new)}")
        if finished:
            self.get_logger().info(f"{robot} reached the end of its path")
        if sent["path"] is not None and (advanced or finished or stalled):
            status = "done" if finished else "stalled" if stalled else "progress"
            self._publish_path_status(robot, sent, status)
        elif sent["path"] is None and (moved or ping):
            # A Spot reports actions, but not during one: say it is moving.
            self._publish_path_status(robot, sent, "progress")
        if stalled:
            text = (
                f"{robot} has made no progress on plan {sent['plan_id']} for "
                f"{now - sent['last_progress']:.0f} s: it has not moved and "
                "reported no action. Check the robot, or send the goal again."
            )
            self.get_logger().warning(text)
            self._progress_pub.publish(String(data=text))

    def _publish_path_status(self, robot, sent, status) -> None:
        pub = self._path_status_pubs.get(robot)
        if pub is None:
            return
        msg = {
            "plan_id": sent["plan_id"],
            "status": status,
            "reached": int(sent["reached"]) + 1,
            "steps": int(sent["steps"]),
        }
        pub.publish(String(data=json.dumps(msg)))

    def status_warnings(self):
        with self._plan_lock:
            return [
                f"{r} stalled on plan {d['plan_id']}"
                for r, d in sorted(self._dispatched.items())
                if d["warned"]
            ]


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
