from __future__ import annotations

from dataclasses import dataclass

import spark_config as sc
from omniplanner.language_planner import LanguageDomain, LanguageGoal
from omniplanner.omniplanner import PlanRequest
from omniplanner_msgs.msg import LanguageGoalMsg


class LanguagePlannerRos:
    def __init__(self, config: LanguagePlannerConfig):
        self.config = config

    def get_plan_callback(self):
        return LanguageGoalMsg, "language_goal", self.language_callback

    def language_callback(self, msg, robot_poses):
        ### TODO: Any information that we need to add to the LanguageGoalMsg needs to get piped through
        ### to this language goal
        goal = LanguageGoal(command=msg.command, robot_id=msg.robot_id)

        req = PlanRequest(
            domain=LanguageDomain(),
            goal=goal,
            robot_states=robot_poses,
        )
        return req


@sc.register_config(
    "omniplanner_pipeline", name="LanguagePlanner", constructor=LanguagePlannerRos
)
@dataclass
class LanguagePlannerConfig(sc.Config):
    pass
