import logging
from importlib.resources import as_file, files

import dsg_pddl.domains
import nlu_interface.resources
import numpy as np
from dsg_pddl.dsg_pddl_interface import PddlDomain, PddlPlan
from nlu_interface.llm_interface import OpenAIWrapper
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Gaze
from ruamel.yaml import YAML
from utils import build_test_dsg

from omniplanner.language_planner import LanguageDomain, LanguageGoal
from omniplanner.omniplanner import (
    PlanRequest,
    full_planning_pipeline,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

yaml = YAML(typ="safe")


def compile_plan_goto(plan, plan_id, robot_name, frame_id):
    """This function turns the output of the planner into a ROS message that is ingestible by a robot"""
    actions = []
    for p in plan.plan:
        xs = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[0], p.goal[0]])
        ys = np.interp(np.linspace(0, 1, 10), [0, 1], [p.start[1], p.goal[1]])
        p_interp = np.vstack([xs, ys])
        actions.append(Follow(frame=frame_id, path2d=p_interp.T))

    seq = ActionSequence(plan_id=plan_id, robot_name=robot_name, actions=actions)
    return seq


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


print("================================")
print("== Goto Point Language Domain ==")
print("================================")
print("")

goal = LanguageGoal("spot", "O(0) O(1)")
robot_domain = LanguageDomain("goto_points", None, None)

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

compiled_plan = compile_plan_goto(
    robot_plan.value, "abc123", robot_plan.name, "a_coordinate_frame"
)
print("compiled plan:")
print(compiled_plan)


print("================================")
print("==   PDDL Language Domain     ==")
print("================================")
print("")

goal = LanguageGoal(command="Euclid, go to objects O(0) and O(1)", robot_id="")
domain_type = "Pddl"
robot_poses = {"euclid": np.array([0.0, 0.1])}

# Load the PDDL domain you want to use
with as_file(files(dsg_pddl.domains).joinpath("GotoObjectDomain.pddl")) as path:
    print(f"Loading domain {path}")
    with open(str(path), "r") as fo:
        domain = PddlDomain(fo.read())

# Load the LLM config to use
with open("llm_config.yaml", "r") as file:
    llm_config = yaml.load(file)


# Load the LLM prompt
with as_file(
    files(nlu_interface.resources).joinpath(llm_config["prompt"] + ".yaml")
) as path:
    print(f'Loading prompt from "{path}"')
    with open(str(path), "r") as file:
        prompt = yaml.load(file)

llm_interface = OpenAIWrapper(
    model=llm_config["model"],
    mode=llm_config["mode"],
    prompt=prompt,
    num_incontext_examples=llm_config["num_incontext_examples"],
    temperature=llm_config["temperature"],
    api_timeout=llm_config["api_timeout"],
    seed=llm_config["seed"],
    api_key_env_var=llm_config["api_key_env_var"],
    debug=llm_config["debug"],
)

# Build the plan request
req = PlanRequest(
    domain=LanguageDomain(domain_type, domain, llm_interface),
    goal=goal,
    robot_states=robot_poses,
)

G = build_test_dsg()
plan = full_planning_pipeline(req, G)

print("Plan from planning domain:")
print(plan)


compiled_plan = compile_plan(
    plan[0].value, "abc123", plan[0].name, "a_coordinate_frame"
)
print("compiled plan:")
print(compiled_plan)
