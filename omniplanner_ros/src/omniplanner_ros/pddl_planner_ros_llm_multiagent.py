from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.resources import as_file, files

import dsg_pddl.domains
import spark_config as sc
from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from omniplanner.omniplanner import PlanRequest
from omniplanner_msgs.msg import PddlGoalMsgList


logger = logging.getLogger(__name__)


class MultiRobotLlmPddlPlannerRos:
    def __init__(self, config: MultirobotPddlConfig):
        self.config = config

        with as_file(
            files(dsg_pddl.domains).joinpath(config.domain_name + ".pddl")
        ) as path:
            logger.info(f"Loading domain {path}")
            with open(str(path), "r") as fo:
                self.domain = PddlDomain(fo.read())

    def get_plan_callback(self):
        return PddlGoalMsgList, "pddl_goal", self.pddl_callback

    def get_plugin_feedback(self, node):
        return None

    def pddl_callback(self, msg, robot_poses):
        logger.info(f"Received PDDL goals {msg.single_robot_goals}")
        goals = []
        for goal in msg.single_robot_goals:
            logger.info(f"  {goal.robot_id}: {goal.pddl_goal}")
            g = PddlGoal(pddl_goal=goal.pddl_goal, robot_id=goal.robot_id)
            robot_domain = self.domain
            req = PlanRequest(
                domain=robot_domain,
                goal=g,
                robot_states=robot_poses,
            )
            goals.append(req)
        return goals


@sc.register_config(
    "omniplanner_pipeline",
    name="MultiRobotLlmPddl",
    constructor=MultiRobotLlmPddlPlannerRos,
)
@dataclass
class MultirobotPddlConfig(sc.Config):
    domain_name: str = None
