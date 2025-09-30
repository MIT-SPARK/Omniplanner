from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.resources import as_file, files

import dsg_pddl.domains
import dsg_pddl.dsg_pddl_grounding_multirobot  # noqa: F401
import dsg_pddl.dsg_pddl_planning  # noqa: F401
import spark_config as sc
from dsg_pddl.pddl_grounding import MultiRobotPddlDomain, PddlGoal
from omniplanner.omniplanner import PlanRequest
from omniplanner_msgs.msg import PddlGoalMsg

logger = logging.getLogger(__name__)


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
