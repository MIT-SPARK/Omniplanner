import logging
from importlib.resources import as_file, files
import numpy as np
import os
import time

import dsg_pddl.domains
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

# NOTE: To get the symbolic plan cost from Fast Downward, you would need to:
# 1. Modify the solve_pddl function in dsg_pddl_planning.py to capture stdout/stderr
# 2. Parse the output for lines like "Plan cost: X" or "Solution cost: X"
# 3. Return both the plan and the cost
# 
# The current implementation only shows geometric costs from the parameterized actions.
# Fast Downward typically reports costs like:
# [t=0.001198s, 11068 KB] Plan cost: 53

def extract_plan_from_wrapper(plan):
    """Extract the actual plan from OmniPlanner wrappers"""
    # Handle SymbolicContext wrapper
    if hasattr(plan, 'value'):
        inner_plan = plan.value
        # Handle RobotWrapper
        if hasattr(inner_plan, 'value'):
            return inner_plan.value
        return inner_plan
    return plan

def calculate_geometric_cost(parameterized_actions):
    """Calculate the total geometric cost from parameterized actions"""
    total_cost = 0.0
    for action_path in parameterized_actions:
        if len(action_path) > 1:
            # Calculate path length for this action
            path_cost = 0.0
            for i in range(1, len(action_path)):
                # Euclidean distance between consecutive waypoints
                diff = np.array(action_path[i]) - np.array(action_path[i-1])
                path_cost += np.linalg.norm(diff)
            total_cost += path_cost
    return total_cost

def extract_symbolic_cost_from_plan(plan):
    """Extract symbolic cost from the plan object if available"""
    # The plan cost is typically embedded in the plan structure
    # For now, we'll return None as the symbolic cost extraction
    # would require modifying the solve_pddl function to capture Fast Downward output
    return None

def calculate_cost_from_plan_file(plan_file_path):
    """Calculate cost from a plan file if available"""
    try:
        if os.path.exists(plan_file_path):
            with open(plan_file_path, 'r') as f:
                lines = f.readlines()
                # Look for cost information in the plan file
                for line in lines:
                    if 'cost =' in line:
                        try:
                            cost_str = line.split('cost =')[1].split()[0]
                            return float(cost_str)
                        except (ValueError, IndexError):
                            pass
                # If no cost found, count actions (unit cost assumption)
                action_lines = [line for line in lines if line.strip() and not line.startswith(';')]
                return len(action_lines)
    except Exception as e:
        print(f"Warning: Could not read plan file {plan_file_path}: {e}")
    return None

def print_plan_with_costs(plan, actual_plan):
    """Print plan information including costs"""
    print("✓ Single-robot planning successful!")
    print(f"Plan type: {type(plan)}")
    print(f"Actual plan type: {type(actual_plan)}")
    
    if hasattr(actual_plan, 'symbolic_actions'):
        print(f"Number of actions: {len(actual_plan.symbolic_actions)}")
        print("Plan actions:")
        for i, action in enumerate(actual_plan.symbolic_actions):
            print(f"  {i+1}: {action}")
        
        # Calculate geometric cost from parameterized actions
        if hasattr(actual_plan, 'parameterized_actions'):
            geometric_cost = calculate_geometric_cost(actual_plan.parameterized_actions)
            print(f"\n📊 Plan Cost Analysis:")
            print(f"  • Number of actions: {len(actual_plan.symbolic_actions)}")
            print(f"  • Geometric path cost: {geometric_cost:.3f} units")
            
            # Try to get symbolic cost from plan file
            plan_file_path = "pddl_dumps/plan_latest.txt"
            symbolic_cost = calculate_cost_from_plan_file(plan_file_path)
            if symbolic_cost is not None:
                print(f"  • Symbolic plan cost: {symbolic_cost}")
            else:
                print(f"  • Symbolic plan cost: Not available (requires Fast Downward output capture)")
                print(f"  • Note: Fast Downward typically reports 'Plan cost: X' in its output")
            
            # Show individual action costs
            print(f"\n  • Action-by-action breakdown:")
            for i, (symbolic_action, parameterized_action) in enumerate(zip(actual_plan.symbolic_actions, actual_plan.parameterized_actions)):
                if len(parameterized_action) > 1:
                    action_cost = 0.0
                    for j in range(1, len(parameterized_action)):
                        diff = np.array(parameterized_action[j]) - np.array(parameterized_action[j-1])
                        action_cost += np.linalg.norm(diff)
                    print(f"    {i+1}. {symbolic_action[0]}: {action_cost:.3f} units")
                else:
                    print(f"    {i+1}. {symbolic_action[0]}: 0.000 units (no movement)")
            
            # Summary and analysis
            print(f"\n📈 Plan Efficiency Analysis:")
            print(f"  • Actions per goal: {len(actual_plan.symbolic_actions)} actions")
            print(f"  • Average cost per action: {geometric_cost/len(actual_plan.symbolic_actions):.3f} units")
            print(f"  • Total geometric distance: {geometric_cost:.3f} units")
            
            if symbolic_cost is not None:
                print(f"  • Symbolic vs Geometric cost ratio: {symbolic_cost/geometric_cost:.3f}")
                if symbolic_cost > geometric_cost:
                    print(f"  • Note: Symbolic cost ({symbolic_cost}) > Geometric cost ({geometric_cost:.3f})")
                    print(f"    This suggests the PDDL domain has non-unit action costs")
                else:
                    print(f"  • Note: Symbolic cost ({symbolic_cost}) ≤ Geometric cost ({geometric_cost:.3f})")
                    print(f"    This suggests the PDDL domain uses unit costs or underestimates")
    else:
        print("Plan object structure:")
        print(f"  Plan attributes: {dir(actual_plan)}")
        print(f"  Plan: {actual_plan}")

# G = build_test_dsg()
# goal = PddlGoal(robot_id="euclid", pddl_goal="(and (visited-object o0) (visited-object o1))")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(object-in-place o1 p0)")
G = spark_dsg.DynamicSceneGraph.load(
    "/home/jaeyoun-choi/colcon_ws/src/awesome_dcist_t4/omniplanner/omniplanner/examples/scenegraph/west_point_fused_map_wregions_labelspace.json"
)


adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
adaptors = {"euclid": adaptor}

robot_poses = {"euclid": np.array([-15.0, -15.1])}

print("================================")
print("==   PDDL Domain (Simple)     ==")
print("================================")
print("")

# TODO: Currently the simple domain has no notion of regions. I intend to add regions here,
# but in a simple way that doesn't reflect the fact that a place is "in" a region.




# goal = PddlGoal(
#     # robot_id="euclid", pddl_goal="(and (visited-place p1042) (visited-object o94) (visited-place p1338) (visited-place p2290) (visited-object o62) (visited-object o2) (visited-object o4) (visited-object o18))"
#     robot_id="euclid", pddl_goal="(and (visited-place p1338) (visited-place p2290) )"
# )
# # goal = PddlGoal(
# #     robot_id="euclid", pddl_goal="(and (visited-object o18) (visited-object o21))"
# # )
# # goal_string ="(and (safe o2)(safe o3)(safe o4)(safe o10)(safe o21)(safe o18))"
# # Load the PDDL domain you want to use
# with as_file(files(dsg_pddl.domains).joinpath("GotoObjectDomain.pddl")) as path:
#     print(f"Loading domain {path}")
#     with open(str(path), "r") as fo:
#         domain = PddlDomain(fo.read())


# # Build the plan request
# req = PlanRequest(
#     domain=domain,
#     goal=goal,
#     robot_states=robot_poses,
# )


# plan = full_planning_pipeline(req, G)

# print("Plan from planning domain:")
# print(plan)
# # Load the PDDL domain you want to use
# with as_file(files(dsg_pddl.domains).joinpath("GotoObjectDomain.pddl")) as path:
#     print(f"Loading domain {path}")
#     with open(str(path), "r") as fo:
#         domain = PddlDomain(fo.read())


# actual_plan = extract_plan_from_wrapper(plan)
        
# print("✓ Single-robot planning successful!")
# print(f"Plan type: {type(plan)}")
# print(f"Actual plan type: {type(actual_plan)}")
# if hasattr(actual_plan, 'symbolic_actions'):
#             print(f"Number of actions: {len(actual_plan.symbolic_actions)}")
#             print("Plan actions:")
#             for i, action in enumerate(actual_plan.symbolic_actions):
#                 print(f"  {i+1}: {action}")
# else:
#     print("Plan object structure:")
#     print(f"  Plan attributes: {dir(actual_plan)}")
#     print(f"  Plan: {actual_plan}")
# compiled_plan = compile_plan(adaptors, plan)
# print(compiled_plan)

# collected_plans = collect_plans(compiled_plan)
# print("Collected plans:")
# print(collected_plans)


# print("================================")
# print("==   PDDL Domain (Pick/Place) ==")
# print("================================")
# print("")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(and (object-in-place o94 p2157))")

# # Load the PDDL domain you want to use
# with as_file(
#     files(dsg_pddl.domains).joinpath("ObjectRearrangementDomain.pddl")
# ) as path:
#     print(f"Loading domain {path}")
#     with open(str(path), "r") as fo:
#         domain = PddlDomain(fo.read())


# # Build the plan request
# req = PlanRequest(
#     domain=domain,
#     goal=goal,
#     robot_states=robot_poses,
# )


# plan = full_planning_pipeline(req, G)

# # print("Plan from planning domain:")
# # print(plan)

# actual_plan = extract_plan_from_wrapper(plan)
        
# print("✓ Single-robot planning successful!")
# print(f"Plan type: {type(plan)}")
# print(f"Actual plan type: {type(actual_plan)}")
# if hasattr(actual_plan, 'symbolic_actions'):
#             print(f"Number of actions: {len(actual_plan.symbolic_actions)}")
#             print("Plan actions:")
#             for i, action in enumerate(actual_plan.symbolic_actions):
#                 print(f"  {i+1}: {action}")
# else:
#     print("Plan object structure:")
#     print(f"  Plan attributes: {dir(actual_plan)}")
#     print(f"  Plan: {actual_plan}")

# # collected_plan = collect_plans(compile_plan(adaptors, plan))
# # print("collected plan: ", collected_plan)


# print("================================")
# print("==   PDDL Domain (Regions)    ==")
# print("================================")
# print("")


# goal = PddlGoal(robot_id="euclid", pddl_goal="(or (visited-place r116) (and (visited-place r69) (visited-place r83)))")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(object-in-place o94 r5)")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(visited-poi o61)")
goal = PddlGoal(
    robot_id="euclid",
    # pddl_goal="(and (visited-region r70) (at-place p1042) (object-in-place o94 p2157))",
    pddl_goal ="(and (object-in-place o5 p91) (object-in-place o94 p2157))"

)

# Track PDDL domain loading time
print("\n🔄 Loading PDDL domain...")
domain_start_time = time.time()

# Load the PDDL domain you want to use
with as_file(
    files(dsg_pddl.domains).joinpath("RegionObjectRearrangementDomain.pddl")
) as path:
    print(f"Loading domain {path}")
    with open(str(path), "r") as fo:
        domain = PddlDomain(fo.read())

domain_end_time = time.time()
domain_loading_time = domain_end_time - domain_start_time

print(f"✓ Domain loaded: {domain.domain_name}")
print(f"⏱️  Domain loading time: {domain_loading_time:.3f} seconds")

# Build the plan request
req = PlanRequest(
    domain=domain,
    goal=goal,
    robot_states=robot_poses,
)

# Track planning execution time
print("\n🔄 Running single-robot planning pipeline...")
planning_start_time = time.time()

plan = full_planning_pipeline(req, G) 
## req: PlanRequest -> req.domain=domain, G: map_context, dsg

planning_end_time = time.time()
planning_execution_time = planning_end_time - planning_start_time

actual_plan = extract_plan_from_wrapper(plan)

# Print plan with cost information
print_plan_with_costs(plan, actual_plan)

# Print execution time summary
total_time = domain_loading_time + planning_execution_time
print(f"\n⏱️  Total execution time: {total_time:.3f} seconds (Domain: {domain_loading_time:.3f}s + Planning: {planning_execution_time:.3f}s)")
print(f"✓ Single-robot planning completed successfully!")
