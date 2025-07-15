import logging
from importlib.resources import as_file, files

import dsg_pddl.domains
import nlu_interface.resources
import numpy as np
from dsg_pddl.pddl_grounding import PddlDomain
from nlu_interface.llm_interface import OpenAIWrapper
from ruamel.yaml import YAML
from utils import DummyRobotPlanningAdaptor, build_test_dsg

from omniplanner.language_planner import LanguageDomain, LanguageGoal
from omniplanner.omniplanner import (
    PlanRequest,
    full_planning_pipeline,
)
from omniplanner_ros.pddl_planner_ros import compile_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

yaml = YAML(typ="safe")

adaptor = DummyRobotPlanningAdaptor("spot", "spot", "map", "body")
adaptors = {"euclid": adaptor}

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

compiled_plan = compile_plan(adaptor, robot_plan)
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


compiled_plan = compile_plan(adaptors, plan)
print("compiled plan:")
print(compiled_plan)
