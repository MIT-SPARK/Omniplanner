from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import omniplanner.compile_plan  # NOQA
import spark_config as sc
from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal, GotoPointsPlan
from omniplanner.omniplanner import PlanRequest
from omniplanner_msgs.msg import GotoPointsGoalMsg
from plum import dispatch
from robot_executor_interface.action_descriptions import ActionSequence, Follow


def compile_points_plan(plan: GotoPointsPlan, plan_id, robot_name, frame_id):
    actions = []
    for p in plan.plan:
        xs = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[0], p.goal[0]])
        ys = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[1], p.goal[1]])
        p_interp = np.vstack([xs, ys])
        actions.append(Follow(frame=frame_id, path2d=p_interp.T))

    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq


@dispatch
def compile_plan(adaptor, p: GotoPointsPlan):
    return compile_points_plan(p, str(uuid.uuid4()), adaptor.name, adaptor.parent_frame)


class GotoPointsRos:
    def __init__(self, config: GotoPointsConfig):
        self.config = config

    def get_plan_callback(self):
        return GotoPointsGoalMsg, "goto_points_goal", self.goto_points_callback

    def get_plugin_feedback(self, node):
        return None

    def goto_points_callback(self, msg, robot_poses):
        goal = GotoPointsGoal(
            goal_points=msg.point_names_to_visit, robot_id=msg.robot_id
        )
        domain = GotoPointsDomain()
        req = PlanRequest(
            domain=domain,
            goal=goal,
            robot_states=robot_poses,
        )
        return req


@sc.register_config(
    "omniplanner_pipeline", name="GotoPoints", constructor=GotoPointsRos
)
@dataclass
class GotoPointsConfig(sc.Config):
    pass
