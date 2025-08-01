from __future__ import annotations

import uuid
from dataclasses import dataclass

import spark_config as sc
from omniplanner.omniplanner import PlanRequest
from omniplanner.tsp import FollowPathPlan, TspDomain, TspGoal
from omniplanner_msgs.msg import GotoPointsGoalMsg
from plum import dispatch
from robot_executor_interface.action_descriptions import ActionSequence, Follow


def compile_path_plan(plan: FollowPathPlan, plan_id, robot_name, frame_id):
    actions = []
    for p in plan.steps:
        actions.append(Follow(frame=frame_id, path2d=p.path))

    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq


@dispatch
def compile_plan(adaptor, p: FollowPathPlan):
    return compile_path_plan(p, str(uuid.uuid4()), adaptor.name, adaptor.parent_frame)


class TspRos:
    def __init__(self, config: TspConfig):
        self.config = config

    def get_plan_callback(self):
        return GotoPointsGoalMsg, "solve_tsp_goal", self.tsp_callback

    def get_plugin_feedback(self, node):
        return None

    def tsp_callback(self, msg, robot_poses):
        goal = TspGoal(goal_points=msg.point_names_to_visit, robot_id=msg.robot_id)
        robot_domain = TspDomain(solver=self.config.solver)
        req = PlanRequest(
            domain=robot_domain,
            goal=goal,
            robot_states=robot_poses,
        )
        return req


@sc.register_config("omniplanner_pipeline", name="Tsp", constructor=TspRos)
@dataclass
class TspConfig(sc.Config):
    solver: str = "2opt"
