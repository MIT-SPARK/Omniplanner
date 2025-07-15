import logging

import numpy as np
from utils import DummyRobotPlanningAdaptor, build_test_dsg

from omniplanner.omniplanner import (
    PlanRequest,
    full_planning_pipeline,
)
from omniplanner.tsp import TspDomain, TspGoal
from omniplanner_ros.goto_points_ros import compile_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")

print("==========================")
print("== TSP Domain           ==")
print("==========================")
print("")

goal = TspGoal(goal_points=["O(0)", "O(1)"], robot_id="spot")

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

compiled_plan = compile_plan(adaptor, robot_plan)
print("compiled plan:")
print(compiled_plan)
