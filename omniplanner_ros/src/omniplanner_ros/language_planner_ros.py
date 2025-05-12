from __future__ import annotations

from dataclasses import dataclass

import spark_config as sc
from omniplanner.language_planner import LanguageDomain, LanguageGoal
from omniplanner.omniplanner import PlanRequest
from omniplanner_msgs.msg import LanguageGoalMsg
from omniplanner_ros.omniplanner_node import PluginFeedbackCollector
import dsg_pddl
from dsg_pddl.dsg_pddl_interface import PddlDomain, PddlGoal, PddlPlan
from std_msgs.msg import String


class LanguagePlannerRos:
    def __init__(self, config: LanguagePlannerConfig):
        self.config = config

        with as_file(
            files(dsg_pddl.domains).joinpath(config.pddl_domain_name + ".pddl")
        ) as path:
            logger.info(f"Loading domain {path}")
            with open(str(path), "r") as fo:
                # Currently, we have a fixed domain. In the future, could make adjustments based on goal message?
                self.domain = PddlDomain(fo.read())

    def get_plan_callback(self):
        return LanguageGoalMsg, "language_goal", self.language_callback

    def get_plugin_feedback(self, node):
        feedback = PluginFeedbackCollector()
        feedback.publishers["llm_response"] = node.create_publisher(
            String, "/rviz2_panel/llm_response", 1
        )
        return feedback

    def language_callback(self, msg, robot_poses):
        ### TODO: Any information that we need to add to the LanguageGoalMsg needs to get piped through
        ### to this language goal
        goal = LanguageGoal(command=msg.command, robot_id=msg.robot_id)

        req = PlanRequest(
            domain=LanguageDomain(self.config.domain_type, self.domain),
            goal=goal,
            robot_states=robot_poses,
        )
        return req


@sc.register_config(
    "omniplanner_pipeline", name="LanguagePlanner", constructor=LanguagePlannerRos
)
@dataclass
class LanguagePlannerConfig(sc.Config):
    domain_type: str = "pddl"
    pddl_domain_name: str = "GotoObjectDomain"
