"""ROS plugin exposing GRSTAPS-X as an omniplanner planner.

Registers like every other planner, so `run-adt4` can drive it: put

    planners:
      grstaps_planner:
        plugin:
          type: Grstaps

in omniplanner_plugins.yaml and publish a GotoPointsGoalMsg to
``~/grstaps_planner/grstaps_goal``. That message already carries exactly what a
single-agent GRSTAPS goal needs (a robot id and the symbols to visit), so no new
message type is required for the proof of concept.

The solver runs in its docker image (it needs Gurobi + OMPL), invoked the same
way pddl_planning invokes fast-downward.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, fields
from typing import overload

import numpy as np
import omniplanner.compile_plan  # NOQA: F401  (registers compile_plan dispatches)
import spark_config as sc
from dsg_pddl.pddl_grounding import ConstraintFact
from omniplanner.grstaps_planner import (
    GrstapsDomain,
    GrstapsGoal,
    GrstapsPlan,
    parse_goal_targets,
    task_location,
)
from omniplanner.omniplanner import (
    MultiRobotWrapper,
    PlanRequest,
    RobotWrapper,
    SymbolicContext,
)
from omniplanner_msgs.msg import (
    ConstrainedPddlGoalMsg,
    GotoPointsGoalMsg,
    ScheduledTaskMsg,
    TaskScheduleMsg,
)
from plum import dispatch
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)
from std_msgs.msg import String

from omniplanner_ros.pddl_planner_ros import ensure_3d

logger = logging.getLogger(__name__)


def _leg_points(transition):
    """One transition as a 2D polyline, or None if it reports no travel.

    Kept per-leg rather than concatenated across the plan. Splicing every
    transition into one polyline loses the boundary at each task, and the
    executor -- which follows with a lookahead and a 2.8 m goal tolerance --
    can then cut the corner across a boundary instead of arriving at the task
    location.
    """
    points = []
    for config in (transition or {}).get("path") or []:
        xy = [config.get("x"), config.get("y")]
        if xy[0] is None or xy[1] is None:
            continue
        if not points or points[-1] != xy:
            points.append(xy)
    return np.array(points, dtype=float) if points else None


def _agent_steps(plan: GrstapsPlan, robot_name: str):
    """(leg, task) pairs for one robot, in the order it executes them.

    transitions[i] is the travel that brings the robot to individual_plan[i]:
    the solver emits exactly one transition per assigned task, and a task the
    robot is already standing at gets a degenerate one-point path rather than
    being omitted. Index alignment is therefore sound, and it is what lets each
    action be placed at the end of the leg that reaches it -- pairing only the
    legs that describe real travel would silently attach actions to the wrong
    task as soon as one task started where the previous one ended.
    """
    by_id = {t.get("id"): t for t in plan.tasks or []}
    for agent in plan.agents or []:
        if agent.get("name") != robot_name:
            continue
        legs = agent.get("transitions") or []
        for i, task_id in enumerate(agent.get("individual_plan") or []):
            leg = _leg_points(legs[i]) if i < len(legs) else None
            yield leg, (by_id.get(task_id) or {})


def _task_actions(geometry, task, frame_id, robot_point):
    """The executor actions one scheduled task becomes.

    A visit is pure navigation and adds nothing beyond the leg that reached it.
    Everything else actuates, and without this the arm never moves and an
    inspection never looks: the fleet would drive the whole mission and
    accomplish none of it.

    `robot_point` is where the robot stands when the task begins -- the end of
    its inbound leg -- which is what the executor's robot_point means and what
    the fast-downward path passes as `last_pose`.
    """
    g = geometry.get(task.get("id"))
    if not g or g["action"] == "visit":
        return []

    kind = g["action"]
    target = g["target"]
    object_point = g["object_point"]
    object_class = g.get("object_class", "")
    here = ensure_3d(robot_point) if robot_point is not None else object_point

    if kind == "inspect":
        return [
            Gaze(
                frame=frame_id,
                robot_point=here,
                gaze_point=object_point,
                stow_after=True,
                object_id=target,
            )
        ]

    pick = Pick(
        frame=frame_id,
        object_class=object_class,
        robot_point=here,
        object_point=object_point,
        object_id=target,
    )
    if kind == "pick":
        return [pick]

    def place_at(robot_at, put_at):
        return Place(
            frame=frame_id,
            object_class=object_class,
            robot_point=robot_at,
            object_point=put_at,
            object_id=target,
        )

    if kind == "place":
        # object_point is already the destination: the task is named after the
        # object but the solver places it where the relocation ends.
        return [place_at(here, object_point)]

    # relocate: pick where it lies, carry, put down at the destination.
    actions = [pick]
    carry = g.get("carry_path")
    destination = g.get("dest_point")
    if destination is None:
        return actions + [place_at(here, object_point)]
    if carry is not None and len(carry) >= 2:
        actions.append(Follow(frame=frame_id, path2d=carry))
        return actions + [place_at(ensure_3d(carry[-1]), destination)]
    return actions + [place_at(destination, destination)]


def _peel_symbolic(x):
    """Strip SymbolicContext / RobotWrapper layers to expose the GrstapsPlan."""
    while isinstance(x, SymbolicContext) or hasattr(x, "value"):
        x = x.value
    return x


def _extract_visited_pois(plan: GrstapsPlan):
    """Return {robot_name -> set(POI ids)} for a solved GRSTAPS-X plan.

    agents[].individual_plan holds task ids, so the per-robot split is exact
    rather than inferred -- unlike the geometry, which is only a polyline.
    """
    by_id = {t.get("id"): t for t in plan.tasks or []}
    result = {}
    for agent in plan.agents or []:
        name = agent.get("name")
        if name is None:
            continue
        pois = set()
        for task_id in agent.get("individual_plan") or []:
            loc = task_location((by_id.get(task_id) or {}).get("name", ""))
            if loc:
                pois.add(loc)
        result[name] = pois
    return result


def _ordering(plan: GrstapsPlan):
    """(predecessors, coalition) lookups keyed by task id, as executor strings.

    The executor is given the ordering rather than the timepoints: the solver's
    times assume the durations it was given, so one slow pick invalidates every
    later one, while "after o4 is out of the way" stays true however long the
    moving took.
    """
    predecessors = {}
    for pair in plan.precedence or []:
        if len(pair) >= 2:
            predecessors.setdefault(str(pair[1]), []).append(str(pair[0]))

    names = [a.get("name", "") for a in plan.agents or []]
    coalitions = {}
    for t in plan.tasks or []:
        members = t.get("coalition") or []
        if len(members) > 1:
            # Solo tasks get no rendezvous: waiting for yourself is a deadlock
            # dressed as a barrier.
            coalitions[str(t.get("id"))] = [
                names[i] for i in members if 0 <= i < len(names)
            ]
    return predecessors, coalitions


def compile_grstaps_plan(plan: GrstapsPlan, plan_id, robot_name, frame_id):
    """One robot's schedule as drive-then-act, task by task.

    Timepoints are still dropped -- ActionSequence has no temporal fields, and
    the schedule goes out separately on TaskScheduleMsg -- but the ordering
    they encoded now travels with the actions.
    """
    predecessors, coalitions = _ordering(plan)
    actions = []
    at = None
    for leg, task in _agent_steps(plan, robot_name):
        task_id = str(task.get("id")) if task.get("id") is not None else ""
        travel = None
        if leg is not None and len(leg) >= 2:
            travel = Follow(frame=frame_id, path2d=leg)
            actions.append(travel)
        if leg is not None and len(leg):
            at = leg[-1]
        acted = _task_actions(plan.task_geometry, task, frame_id, at)
        actions.extend(acted)

        # Gate the task's own action, not the travel to it: arriving early is
        # harmless and it is how a coalition member gets into position before
        # announcing itself ready. A visit has no action of its own -- arriving
        # is the task -- so there the travel is what gets gated.
        for action in filter(None, [travel, *acted]):
            action.task_id = task_id
        gated = acted[0] if acted else travel
        if gated is not None:
            gated.after_task_ids = list(predecessors.get(task_id, []))
            gated.coalition_robots = list(coalitions.get(task_id, []))
        # A relocation moves the robot inside the task, so the next task's
        # robot_point is where the carry ended, not where the inbound leg did.
        # This matters whenever the following task reports no travel of its own.
        for action in acted:
            if isinstance(action, Follow):
                if len(action.path2d):
                    at = action.path2d[-1]
            elif getattr(action, "robot_point", None) is not None:
                at = action.robot_point
    return ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)


@overload
@dispatch
def compile_plan(adaptor, plan_frame: str, p: GrstapsPlan):
    return compile_grstaps_plan(p, str(uuid.uuid4()), adaptor.name, plan_frame)


@overload
@dispatch
def compile_plan(
    adaptors: dict, plan_frame: str, p: MultiRobotWrapper[SymbolicContext[GrstapsPlan]]
):
    """Split one solved fleet plan into a per-robot ActionSequence.

    GRSTAPS-X solves for the whole fleet at once and reports the allocation in
    agents[].name, so the split is the solver's own decision rather than
    anything re-derived here. Every robot in the fleet gets a sequence, even an
    empty one -- a robot the allocator chose not to use still needs a plan
    published, or it keeps executing its previous one.
    """
    plan = p.value.value
    plan_id = str(uuid.uuid4())
    out = []
    for robot_name in p.names:
        adaptor = adaptors.get(robot_name)
        name = adaptor.name if adaptor is not None else robot_name
        out.append(
            RobotWrapper(
                robot_name, compile_grstaps_plan(plan, plan_id, name, plan_frame)
            )
        )
    return out


class GrstapsRos:
    def __init__(self, config: GrstapsConfig):
        self.config = config
        self._node = None
        self._visited_pois_pubs: dict = {}  # {robot_name -> ROS publisher}
        self._schedule_pub = None

    def get_plan_callback(self):
        # Two ways in: a plain list of symbols, or the plan-repair flow's
        # ConstrainedPddlGoalMsg (goal string + constraints), which lets
        # goal_manager drive this planner exactly as it drives the PDDL one.
        if self.config.goal_format == "constrained_pddl":
            return ConstrainedPddlGoalMsg, "pddl_goal", self.constrained_pddl_callback
        return GotoPointsGoalMsg, "grstaps_goal", self.grstaps_callback

    def get_plugin_feedback(self, node):
        self._node = node
        return None

    def constrained_pddl_callback(self, msg, robot_poses):
        """Plan-repair entry point: PDDL goal string + runtime constraints."""
        found = parse_goal_targets(msg.goal.pddl_goal)
        if not any(found.values()):
            logger.warning(
                "No recognised targets in goal %r; nothing for GRSTAPS-X to do.",
                msg.goal.pddl_goal,
            )
        else:
            logger.info(
                "Goal: %d visit, %d inspect, %d manipulate target(s)",
                len(found["visit"]),
                len(found["inspect"]),
                len(found["manipulate"]),
            )

        # Constraints arrive from two places and both must be honoured: the
        # repair node accumulates persistent ones on the node, and the goal
        # message carries whatever the agent attached to this goal. Merging
        # matches MultiRobotPddlConstrained, so switching planners does not
        # silently drop the constraints a user already stated.
        persistent = []
        if self._node is not None:
            persistent = list(getattr(self._node, "active_constraints", []))
        per_msg = [
            ConstraintFact(predicate=c.predicate, symbols=list(c.symbols))
            for c in msg.constraints
        ]

        goal = GrstapsGoal(
            goal_points=found["visit"],
            inspect_points=found["inspect"],
            manipulate_points=found["manipulate"],
            robot_id=msg.goal.robot_id,
            constraints=persistent + per_msg,
        )
        return self._request(goal, robot_poses)

    def _publish_schedule(self, plan, plan_dict):
        """Publish the timings the solver computed, which the plan cannot carry.

        ActionSequenceMsg has no temporal fields, so compiling a plan discards
        every start/finish timepoint -- most of what a temporal planner is for.
        A separate topic makes the schedule observable without changing the
        executor contract; giving the executor timed behaviour is a later,
        cross-repo step.
        """
        if self._schedule_pub is None:
            self._schedule_pub = self._node.create_publisher(
                TaskScheduleMsg, "~/task_schedule", 1
            )

        robot_of_task = {}
        for agent in plan.agents or []:
            for task_id in agent.get("individual_plan") or []:
                robot_of_task[task_id] = agent.get("name", "")

        msg = TaskScheduleMsg()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        # Tie the schedule to the geometry published for the same plan.
        msg.plan_id = next(
            (getattr(p, "plan_id", "") for p in (plan_dict or {}).values()), ""
        )
        msg.makespan = float(plan.makespan)
        for t in sorted(plan.tasks or [], key=lambda t: t.get("start_timepoint", 0.0)):
            st = ScheduledTaskMsg()
            st.robot_name = robot_of_task.get(t.get("id"), "")
            name = t.get("name", "")
            st.action = name.split()[0] if name.split() else ""
            st.target = task_location(name) or ""
            st.start_time = float(t.get("start_timepoint", 0.0))
            st.finish_time = float(t.get("finish_timepoint", 0.0))
            msg.tasks.append(st)
        self._schedule_pub.publish(msg)
        logger.info(
            "Published schedule: makespan=%.2f s, %d task(s)",
            msg.makespan,
            len(msg.tasks),
        )

    def on_plan_compiled(self, plans, plan_dict):
        """Publish the POIs each robot's freshly compiled plan visits.

        goal_manager treats an empty cache as "no plan yet" and replans
        unconditionally, so without this hook the repair flow still behaves
        correctly but never skips a redundant replan -- the whole point of it.
        """
        if self._node is None:
            return
        plan = _peel_symbolic(plans)
        if not isinstance(plan, GrstapsPlan):
            return
        for robot_name, pois in _extract_visited_pois(plan).items():
            pub = self._visited_pois_pubs.get(robot_name)
            if pub is None:
                pub = self._node.create_publisher(
                    String,
                    f"/{robot_name}/omniplanner_node/plan_visited_pois",
                    1,
                )
                self._visited_pois_pubs[robot_name] = pub
            pub.publish(String(data=json.dumps(sorted(pois))))
        self._publish_schedule(plan, plan_dict)

    def grstaps_callback(self, msg, robot_poses):
        goal = GrstapsGoal(
            goal_points=list(msg.point_names_to_visit), robot_id=msg.robot_id
        )
        return self._request(goal, robot_poses)

    def _robot_types(self):
        """{robot_name -> robot_type} from the node's adaptors.

        omniplanner_plugins.yaml already declares a robot_type for every robot
        and omniplanner_node exposes it on each adaptor, so species selection
        needs no new configuration and no change to the node.
        """
        adaptors = getattr(self._node, "robot_adaptors", None) or {}
        return {name: getattr(a, "robot_type", "") for name, a in adaptors.items()}

    def _request(self, goal, robot_poses):
        goal.robot_types = self._robot_types()
        # Only forward what the yaml actually set, so GrstapsDomain's
        # environment-driven defaults still apply to everything else.
        overrides = {
            f.name: getattr(self.config, f.name)
            for f in fields(self.config)
            if getattr(self.config, f.name) is not None
        }
        overrides.pop("goal_format", None)  # plugin-side only; not a domain knob
        domain = GrstapsDomain(**overrides)
        return PlanRequest(domain=domain, goal=goal, robot_states=robot_poses)


@sc.register_config("omniplanner_pipeline", name="Grstaps", constructor=GrstapsRos)
@dataclass
class GrstapsConfig(sc.Config):
    """Mirrors GrstapsDomain's knobs. Anything left None falls back to that
    dataclass's defaults, which read ADT4_GRSTAPS_* / ADT4_GUROBI_* from the
    environment -- so a deployment overrides paths without editing this config.
    """

    # "points" -> GotoPointsGoalMsg on grstaps_goal
    # "constrained_pddl" -> ConstrainedPddlGoalMsg on pddl_goal (repair flow)
    goal_format: str = "points"
    run_mode: str = None  # "native" (Named-User licence) or "docker" (WLS)
    repo_root: str = None
    docker_image: str = None
    license_path: str = None
    native_binary: str = None
    gurobi_home: str = None
    conda_prefix: str = None
    scenario_name: str = None
    solver_params_from: str = None
    # Task durations in seconds. The scheduler places tasks in time from these
    # plus travel, so they are what a makespan is measured against.
    visit_duration_s: float = None
    inspect_duration_s: float = None
    pick_duration_s: float = None
    place_duration_s: float = None
    # How far a forbidden-poi constraint reaches, in metres of navigable path.
    forbidden_radius_m: float = None
    # "pddl"  -> domain/problem.pddl through grstapsx_example
    # "itags" -> itags_input.json straight into the itags binary, which also
    #            honours `before` ordering constraints. Needs the itags target
    #            built; see ADT4_ITAGS_BINARY.
    entry: str = None
    itags_binary: str = None
    # {robot_name: species} to override what robot_type implies, e.g.
    # {hilbert: spot_arm} when the sim spot stands in for one with an arm.
    # Species are spot, spot_arm, uav.
    robot_species: dict = None
