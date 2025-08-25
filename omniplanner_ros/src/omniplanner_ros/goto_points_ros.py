from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import omniplanner.compile_plan  # NOQA
import spark_config as sc
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal, GotoPointsPlan
from omniplanner.omniplanner import PlanRequest
from omniplanner_msgs.msg import GotoPointsGoalMsg
from plum import dispatch
from rclpy.clock import Clock
from robot_executor_interface_ros.action_descriptions_ros import to_msg


@to_msg.register
def _(action: Path):
    return action


def compile_points_plan(plan: GotoPointsPlan, plan_id, robot_name, frame_id):
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = Clock().now().to_msg()

    for p in plan.plan:
        xs = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[0], p.goal[0]])
        ys = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[1], p.goal[1]])

        for x, y in zip(xs, ys):
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = (
                Clock().now().to_msg()
            )  # Same timestamp is okay, or could vary
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # Assuming no orientation is needed or use a default quaternion
            pose.pose.orientation.w = 1.0  # Neutral rotation

            path_msg.poses.append(pose)

    return path_msg


@dispatch
def compile_plan(adaptor, plan_frame: str, p: GotoPointsPlan):
    return compile_points_plan(p, str(uuid.uuid4()), adaptor.name, plan_frame)


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
