from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib.resources import as_file, files

import dsg_pddl.domains
import dsg_pddl.dsg_pddl_grounding_multirobot  # noqa: F401
import dsg_pddl.dsg_pddl_planning  # noqa: F401
import spark_config as sc
from dsg_pddl.dsg_pddl_planning import PddlPlan
from dsg_pddl.pddl_grounding import ConstraintFact, MultiRobotPddlDomain, PddlGoal
from omniplanner.omniplanner import MultiRobotWrapper, PlanRequest, SymbolicContext
from omniplanner_msgs.msg import ConstrainedPddlGoalMsg, PddlGoalMsg
from std_msgs.msg import String

logger = logging.getLogger(__name__)


def _peel_symbolic(x):
    """Strip SymbolicContext layers (may be nested) to expose the inner value."""
    while isinstance(x, SymbolicContext):
        x = x.value
    return x


def _extract_visited_pois_from_pipeline(plans):
    """Return {robot_name -> set(POI ids)} for a multi-robot PDDL pipeline output.

    Mirrors the per-robot split that compile_multirobot_pddl_plan does so the
    plugin can publish visit-coverage without depending on a side-channel cache.
    """
    result: dict[str, set[str]] = {}
    inner = _peel_symbolic(plans)

    if isinstance(inner, MultiRobotWrapper):
        plan = _peel_symbolic(inner.value)
        if not isinstance(plan, PddlPlan):
            return result
        for rn in inner.names:
            result[rn] = set()
        for sym in plan.symbolic_actions:
            if len(sym) < 2 or sym[0] != "goto-poi":
                continue
            rn = inner.remap_name_to_outer(sym[1])
            if rn in result:
                result[rn].add(sym[-1])
        return result

    if isinstance(inner, PddlPlan) and inner.symbolic_actions:
        # Single-robot fallback.
        rn = inner.symbolic_actions[0][1]
        result[rn] = {sym[-1] for sym in inner.symbolic_actions if sym[0] == "goto-poi"}

    return result


class MultiRobotPddlPlannerRos:
    def __init__(self, config: MultiRobotPddlConfig):
        self.config = config

        with as_file(
            files(dsg_pddl.domains).joinpath(config.domain_name + ".pddl")
        ) as path:
            logger.info(f"Loading domain {path}")
            with open(str(path), "r") as fo:
                # Currently, we have a fixed domain. In the future, could make adjustments based on goal message?
                self.domain = MultiRobotPddlDomain(fo.read())

    def get_plan_callback(self):
        # TODO: topic name should depend on the config (i.e. what domain is specified)
        return PddlGoalMsg, "pddl_goal", self.pddl_callback

    def get_plugin_feedback(self, node):
        return None

    def pddl_callback(self, msg, robot_poses):
        logger.info(f"Received PDDL goal {msg.pddl_goal} for robot {msg.robot_id}")
        goal = PddlGoal(pddl_goal=msg.pddl_goal, robot_id=msg.robot_id)
        robot_domain = self.domain
        req = PlanRequest(
            domain=robot_domain,
            goal=goal,
            robot_states=robot_poses,
        )
        return req


@sc.register_config(
    "omniplanner_pipeline", name="MultiRobotPddl", constructor=MultiRobotPddlPlannerRos
)
@dataclass
class MultiRobotPddlConfig(sc.Config):
    domain_name: str = None


class MultiRobotPddlConstrainedPlannerRos(MultiRobotPddlPlannerRos):
    """Variant of MultiRobotPddlPlannerRos that supports runtime constraints.

    Differences from the base plugin:
      - Subscribes to ConstrainedPddlGoalMsg instead of PddlGoalMsg.
      - Merges per-message constraints with the parent node's
        `active_constraints` (the persistent set, updated via the repair
        node's ``~/constraints`` topic) before grounding.
      - Defines an ``on_plan_compiled`` hook which the omniplanner core
        invokes after a plan is published; it extracts the POIs visited by
        the freshly compiled plan and publishes them, so the goal manager
        can decide whether a subsequent goal needs replanning.
    """

    def __init__(self, config: MultiRobotPddlConstrainedConfig):
        super().__init__(config)
        self._node = None
        self._visited_pois_pubs: dict = {}  # {robot_name -> ROS publisher}

    def get_plan_callback(self):
        return ConstrainedPddlGoalMsg, "pddl_goal", self.pddl_callback

    def get_plugin_feedback(self, node):
        self._node = node
        return None

    def pddl_callback(self, msg: ConstrainedPddlGoalMsg, robot_poses):
        logger.info(
            f"Received constrained PDDL goal {msg.goal.pddl_goal} "
            f"for robot {msg.goal.robot_id} with {len(msg.constraints)} per-msg constraints"
        )

        persistent = []
        if self._node is not None:
            persistent = list(getattr(self._node, "active_constraints", []))

        per_msg = [
            ConstraintFact(predicate=c.predicate, symbols=list(c.symbols))
            for c in msg.constraints
        ]
        merged = persistent + per_msg

        goal = PddlGoal(
            pddl_goal=msg.goal.pddl_goal,
            robot_id=msg.goal.robot_id,
            constraints=merged,
        )
        return PlanRequest(domain=self.domain, goal=goal, robot_states=robot_poses)

    def on_plan_compiled(self, plans, plan_dict):
        """Publish the set of POIs each robot's freshly compiled plan visits."""
        if self._node is None:
            return
        visited = _extract_visited_pois_from_pipeline(plans)
        for robot_name, pois in visited.items():
            pub = self._visited_pois_pubs.get(robot_name)
            if pub is None:
                pub = self._node.create_publisher(
                    String,
                    f"/{robot_name}/omniplanner_node/plan_visited_pois",
                    1,
                )
                self._visited_pois_pubs[robot_name] = pub
            msg = String()
            msg.data = json.dumps(sorted(pois))
            pub.publish(msg)


@sc.register_config(
    "omniplanner_pipeline",
    name="MultiRobotPddlConstrained",
    constructor=MultiRobotPddlConstrainedPlannerRos,
)
@dataclass
class MultiRobotPddlConstrainedConfig(MultiRobotPddlConfig):
    pass
