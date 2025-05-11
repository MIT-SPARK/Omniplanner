from __future__ import annotations

from dataclasses import dataclass

import spark_config as sc
from omniplanner.language_planner import LanguageDomain, LanguageGoal
from omniplanner.omniplanner import PlanRequest
from omniplanner_msgs.msg import LanguageGoalMsg
import dsg_pddl


class LanguagePlannerRos:
    def __init__(self, config: LanguagePlannerConfig):
        self.config = config
        self.node = None

        with as_file(
            files(dsg_pddl.domains).joinpath(config.pddl_domain_name + ".pddl")
        ) as path:
            logger.info(f"Loading domain {path}")
            with open(str(path), "r") as fo:
                # Currently, we have a fixed domain. In the future, could make adjustments based on goal message?
                self.domain = PddlDomain(fo.read()) # TODO: import PddlDomain

    def get_plan_callback(self):
        return LanguageGoalMsg, "language_goal", self.language_callback

    def set_plugin_node(self, node):
        self.node = node
        self.pub_a = self.node.create_publisher(...)

    def language_callback(self, msg, robot_poses):
        ### TODO: Any information that we need to add to the LanguageGoalMsg needs to get piped through
        ### to this language goal
        goal = LanguageGoal(command=msg.command, robot_id=msg.robot_id)

        req = PlanRequest(
            domain=LanguageDomain(self.config.domain_type, self.domain),
            goal=goal,
            robot_states=robot_poses,
        )
        self.pub_a()
        return req


@sc.register_config(
    "omniplanner_pipeline", name="LanguagePlanner", constructor=LanguagePlannerRos
)
@dataclass
class LanguagePlannerConfig(sc.Config):
    domain_type: str = "pddl"
    pddl_domain_name: str = "GotoObjectDomain"
