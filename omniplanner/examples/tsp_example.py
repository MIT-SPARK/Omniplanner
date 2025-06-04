import logging

import numpy as np
from robot_executor_interface.action_descriptions import ActionSequence, Follow
from utils import build_test_dsg

from omniplanner.omniplanner import (
    PlanRequest,
    full_planning_pipeline,
)
from omniplanner.tsp import FollowPathPlan, TspDomain, TspGoal

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


def compile_plan(plan: FollowPathPlan, plan_id, robot_name, frame_id):
    actions = []
    for p in plan:
        actions.append(Follow(frame=frame_id, path2d=p.path))
    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq


print("==========================")
print("== TSP Domain           ==")
print("==========================")
print("")

goal = TspGoal(goal_points=["o(0)", "o(1)"], robot_id="spot")

robot_domain = TspDomain(solver="2opt")

robot_poses = {"spot": np.array([0.0, 0.1])}

req = PlanRequest(
    domain=robot_domain,
    goal=goal,
    robot_states=robot_poses,
)

G = build_test_dsg()
robot_plan = full_planning_pipeline(req, G)

print("Plan from planning domain:")
print(robot_plan)

compiled_plan = compile_plan(
    robot_plan.value, "abc123", robot_plan.name, "a_coordinate_frame"
)
print("compiled plan:")
print(compiled_plan)
