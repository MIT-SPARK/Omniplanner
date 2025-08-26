from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import overload

import dsg_pddl.domains
import numpy as np
import spark_config as sc
from dsg_pddl.dsg_pddl_planning import PddlPlan
from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from omniplanner.omniplanner import PlanRequest, SymbolicContext
from omniplanner_msgs.msg import PddlGoalMsg
from plum import dispatch
from rclpy.clock import Clock
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)

from omniplanner_ros.omniplanner_node import PhoenixPlanningAdaptor

logger = logging.getLogger(__name__)


def ensure_3d(pt):
    if len(pt) < 2:
        raise Exception("Expected 2d or 3d point, got point length: {len(pt}")
    if len(pt) == 2:
        new_pt = np.zeros(3)
        new_pt[:2] = pt
        return new_pt
    return pt


@overload
@dispatch
def compile_plan(adaptor, plan_frame: str, plan: PddlPlan):
    return compile_plan(adaptor, plan_frame, SymbolicContext({}, plan))


@overload
@dispatch
def compile_plan(adaptor, plan_frame: str, p: SymbolicContext[PddlPlan]):
    return compile_pddl_plan(p, str(uuid.uuid4()), adaptor.name, plan_frame)


def compile_pddl_plan(
    contextualized_plan: SymbolicContext[PddlPlan], plan_id, robot_name, frame_id
):
    plan = contextualized_plan.value
    context = contextualized_plan.context
    actions = []
    for symbolic_action, parameters in zip(
        plan.symbolic_actions, plan.parameterized_actions
    ):
        match symbolic_action[0]:
            case "goto-poi":
                actions.append(Follow(frame=frame_id, path2d=parameters))
            case "inspect":
                robot_point, gaze_point = parameters
                actions.append(
                    Gaze(
                        frame=frame_id,
                        robot_point=ensure_3d(robot_point),
                        gaze_point=ensure_3d(gaze_point),
                        stow_after=True,
                    )
                )
            case "pick-object":
                robot_point, pick_point = parameters
                object_class = ""
                if symbolic_action[1] in context:
                    attrs = context[symbolic_action[1]]
                    if "semantic_label" in attrs:
                        object_class = attrs["semantic_label"]
                actions.append(
                    Pick(
                        frame=frame_id,
                        object_class=object_class,
                        robot_point=ensure_3d(robot_point),
                        object_point=ensure_3d(pick_point),
                    )
                )
            case "place-object":
                robot_point, place_point = parameters
                actions.append(
                    Place(
                        frame=frame_id,
                        object_class=object_class,
                        robot_point=ensure_3d(robot_point),
                        object_point=ensure_3d(place_point),
                    )
                )
            case _:
                raise NotImplementedError(
                    f"I don't know how to compile {symbolic_action[0]}"
                )

    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq


@dispatch
def compile_plan(
    adaptor: PhoenixPlanningAdaptor, plan_frame: str, p: SymbolicContext[PddlPlan]
):
    return compile_phoenix_pddl_plan(p, adaptor.name, plan_frame)


def compile_phoenix_pddl_plan(contextualized_plan, robot_name, frame_id):
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = Clock().now().to_msg()
    plan = contextualized_plan.value
    for symbolic_action, parameters in zip(
        plan.symbolic_actions, plan.parameterized_actions
    ):
        match symbolic_action[0]:
            case "goto-poi":
                for pose in parameters:
                    p = PoseStamped()
                    p.header.frame_id = frame_id
                    p.pose.position.x = pose[0]
                    p.pose.position.y = pose[1]
                    p.pose.orientation.w = 1.0
                    path_msg.poses.append(p)
            case "inspect":
                logger.error("Phoenix doesn't know how to inspect :(. Skipping!")
            case "pick-object":
                logger.error("Phoenix doesn't know how to pick-object :(. Skipping!")
            case "place-object":
                logger.error("Phoenix doesn't know how to place-object :(. Skipping!")
            case _:
                raise NotImplementedError(
                    f"I don't know how to compile {symbolic_action[0]}"
                )
    return path_msg


class PddlPlannerRos:
    def __init__(self, config: PddlConfig):
        self.config = config

        with as_file(
            files(dsg_pddl.domains).joinpath(config.domain_name + ".pddl")
        ) as path:
            logger.info(f"Loading domain {path}")
            with open(str(path), "r") as fo:
                # Currently, we have a fixed domain. In the future, could make adjustments based on goal message?
                self.domain = PddlDomain(fo.read())

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


@sc.register_config("omniplanner_pipeline", name="Pddl", constructor=PddlPlannerRos)
@dataclass
class PddlConfig(sc.Config):
    domain_name: str = None
