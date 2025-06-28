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
from omniplanner.omniplanner import PlanRequest, SymbolicContext
from omniplanner_msgs.msg import PddlGoalMsg
from plum import dispatch
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)

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
def compile_plan(adaptor, plan: PddlPlan):
    return compile_plan(adaptor, SymbolicContext({}, plan))


# TODO: if we wanted to compile a pddl plan to a different target (e.g. Phoenix macroactions), we would dispatch
# on the type of `adaptor`
@dispatch
def compile_plan(adaptor, p: SymbolicContext[PddlPlan]):
    return compile_pddl_plan(p, str(uuid.uuid4()), adaptor.name, adaptor.parent_frame)


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
