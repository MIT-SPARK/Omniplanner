from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.resources import as_file, files

import dsg_pddl.domains
import spark_config as sc
from dsg_pddl.dsg_pddl_interface import PddlDomain, PddlGoal, PddlPlan
from omniplanner.omniplanner import PlanRequest, compile_plan
from omniplanner_msgs.msg import PddlGoalMsg
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Gaze

logger = logging.getLogger(__name__)


@compile_plan.register
def compile_plan(plan: PddlPlan, plan_id, robot_name, frame_id):
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
                        robot_point=robot_point,
                        gaze_point=gaze_point,
                        stow_after=True,
                    )
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

    def pddl_callback(self, msg, robot_poses):
        logger.info(f"Received PDDL goal {msg.pddl_goal} for robot {msg.robot_id}")
        goal = PddlGoal(pddl_goal=msg.pddl_goal, robot_id=msg.robot_id)
        req = PlanRequest(
            domain=self.domain,
            goal=goal,
            robot_states=robot_poses,
        )
        return req


@sc.register_config("omniplanner_pipeline", name="Pddl", constructor=PddlPlannerRos)
@dataclass
class PddlConfig(sc.Config):
    domain_name: str = None
