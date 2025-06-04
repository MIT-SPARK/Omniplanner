import logging
from importlib.resources import as_file, files

import dsg_pddl.domains
import numpy as np
import spark_dsg
from dsg_pddl.dsg_pddl_planning import PddlPlan
from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)
from ruamel.yaml import YAML

from omniplanner.omniplanner import (
    PlanRequest,
    full_planning_pipeline,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

yaml = YAML(typ="safe")


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
            case "pick-object":
                robot_point, pick_point = parameters
                actions.append(
                    Pick(
                        frame=frame_id,
                        object_class="",
                        robot_point=robot_point,
                        object_point=pick_point,
                    )
                )
            case "place-object":
                robot_point, place_point = parameters
                actions.append(
                    Place(
                        frame=frame_id,
                        object_class="",
                        robot_point=robot_point,
                        object_point=place_point,
                    )
                )
            case _:
                raise NotImplementedError(
                    f"I don't know how to compile {symbolic_action[0]}"
                )

    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq


print("================================")
print("==   PDDL Domain              ==")
print("================================")
print("")

# goal = PddlGoal(robot_id="euclid", pddl_goal="(and (visited-object o0) (visited-object o1))")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(object-in-place o1 p0)")

# goal = PddlGoal(robot_id="euclid", pddl_goal="(or (visited-place r116) (and (visited-place r69) (visited-place r83)))")
# goal = PddlGoal(robot_id="euclid", pddl_goal="(object-in-place o94 r5)")
goal = PddlGoal(robot_id="euclid", pddl_goal="(visited-poi o61)")

# (object-in-place o0 p0)
# (object-in-place o1 p1) )


robot_poses = {"euclid": np.array([0.0, 0.1])}

# Load the PDDL domain you want to use
# with as_file(files(dsg_pddl.domains).joinpath("GotoObjectDomain.pddl")) as path:
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


# G = build_test_dsg()

G = spark_dsg.DynamicSceneGraph.load(
    "/home/ubuntu/lxc_datashare/west_point_fused_map_wregions.json"
)

plan = full_planning_pipeline(req, G)

print("Plan from planning domain:")
print(plan)


compiled_plan = compile_plan(plan.value, "abc123", plan.name, "a_coordinate_frame")
print("compiled plan:")
print(compiled_plan)
