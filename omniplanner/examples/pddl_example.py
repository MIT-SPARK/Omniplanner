import logging
from importlib.resources import as_file, files

import dsg_pddl.domains
import numpy as np
import spark_dsg
from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from ruamel.yaml import YAML
from utils import DummyRobotPlanningAdaptor

from omniplanner.compile_plan import collect_plans
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner_ros.pddl_planner_ros import compile_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

yaml = YAML(typ="safe")


# G = build_test_dsg()
# goal = PddlGoal(robot_id="euclid", pddl_goal="(and (visited-object o0) (visited-object o1))")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(object-in-place o1 p0)")
G = spark_dsg.DynamicSceneGraph.load(
    "/home/ubuntu/lxc_datashare/west_point_fused_map_wregions_labelspace.json"
)


adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
adaptors = {"euclid": adaptor}

robot_poses = {"euclid": np.array([0.0, 0.1])}

print("================================")
print("==   PDDL Domain (Simple)     ==")
print("================================")
print("")

# TODO: Currently the simple domain has no notion of regions. I intend to add regions here,
# but in a simple way that doesn't reflect the fact that a place is "in" a region.
goal = PddlGoal(
    robot_id="euclid", pddl_goal="(and (visited-place p1042) (visited-object o94))"
)

# Load the PDDL domain you want to use
with as_file(files(dsg_pddl.domains).joinpath("GotoObjectDomain.pddl")) as path:
    print(f"Loading domain {path}")
    with open(str(path), "r") as fo:
        domain = PddlDomain(fo.read())


# Build the plan request
req = PlanRequest(
    domain=domain,
    goal=goal,
    robot_states=robot_poses,
)


plan = full_planning_pipeline(req, G)

print("Plan from planning domain:")
print(plan)

compiled_plan = compile_plan(adaptors, plan)
print(compiled_plan)

collected_plans = collect_plans(compiled_plan)
print("Collected plans:")
print(collected_plans)


print("================================")
print("==   PDDL Domain (Pick/Place) ==")
print("================================")
print("")
goal = PddlGoal(robot_id="euclid", pddl_goal="(and (object-in-place o94 p2157))")

# Load the PDDL domain you want to use
with as_file(
    files(dsg_pddl.domains).joinpath("ObjectRearrangementDomain.pddl")
) as path:
    print(f"Loading domain {path}")
    with open(str(path), "r") as fo:
        domain = PddlDomain(fo.read())


# Build the plan request
req = PlanRequest(
    domain=domain,
    goal=goal,
    robot_states=robot_poses,
)


plan = full_planning_pipeline(req, G)

print("Plan from planning domain:")
print(plan)

collected_plan = collect_plans(compile_plan(adaptors, plan))
print("collected plan: ", collected_plan)


print("================================")
print("==   PDDL Domain (Regions)    ==")
print("================================")
print("")


# goal = PddlGoal(robot_id="euclid", pddl_goal="(or (visited-place r116) (and (visited-place r69) (visited-place r83)))")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(object-in-place o94 r5)")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(visited-poi o61)")
goal = PddlGoal(
    robot_id="euclid",
    pddl_goal="(and (visited-region r70) (at-place p1042) (object-in-place o94 p2157))",
)

# Load the PDDL domain you want to use
with as_file(
    files(dsg_pddl.domains).joinpath("RegionObjectRearrangementDomain.pddl")
) as path:
    print(f"Loading domain {path}")
    with open(str(path), "r") as fo:
        domain = PddlDomain(fo.read())


# Build the plan request
req = PlanRequest(
    domain=domain,
    goal=goal,
    robot_states=robot_poses,
)


plan = full_planning_pipeline(req, G)

print("Plan from planning domain:")
print(plan)


collected_plan = collect_plans(compile_plan(adaptors, plan))
print("collected plan: ", collected_plan)
