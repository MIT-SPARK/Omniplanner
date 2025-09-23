import logging
import os
from importlib.resources import as_file, files

import dsg_pddl.domains
import nlu_interface.resources
import numpy as np
from dsg_pddl.pddl_grounding import PddlDomain
from nlu_interface.llm_interface import OpenAIWrapper
from ruamel.yaml import YAML
from utils import DummyRobotPlanningAdaptor, build_test_dsg

from omniplanner.language_planner import LanguageDomain, LanguageGoal
from omniplanner.hybrid_language_planner import HybridLanguageDomain, DCGInterface
from omniplanner.omniplanner import (
    PlanRequest,
    full_planning_pipeline,
)
from omniplanner_ros.pddl_planner_ros import compile_plan

def replace_env_var(string, env_var):
    return string.replace("$" + env_var, os.getenv(env_var))

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

yaml = YAML(typ="safe")

adaptor = DummyRobotPlanningAdaptor("spot", "spot", "map", "body")
adaptors = {"euclid": adaptor}

print("================================")
print("==   PDDL Language Domain     ==")
print("================================")
print("")

goal = LanguageGoal(command="Euclid, go to objects O(0) and O(1)", robot_id="")
domain_type = "Pddl"
robot_poses = {"euclid": np.array([0.0, 0.1])}

# Load the example program config
with open("config/hybrid_language_example.yaml", "r") as file:
    example_config = yaml.load(file)

# Load the PDDL domain
pddl_domain_filename = replace_env_var(example_config["pddl_domain"], "OMNIPLANNER_PATH")
with open(pddl_domain_filename, "r") as file:
    print(f"Loading the PDDL domain file: {pddl_domain_filename}")
    domain = PddlDomain(file.read())

# Load the DCG config to use
dcg_config_filename = replace_env_var(example_config["dcg_config"], "OMNIPLANNER_PATH")
with open(dcg_config_filename, "r") as file:
    print(f"Loading the DCG config file: {dcg_config_filename}")
    dcg_config = yaml.load(file)

# Load the LLM config to use
llm_config_filename = replace_env_var(example_config["llm_config"], "OMNIPLANNER_PATH")
with open(llm_config_filename, "r") as file:
    print(f"Loading the LLM config file: {llm_config_filename}")
    llm_config = yaml.load(file)

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
language_domain = LanguageDomain(domain_type, domain, llm_interface)
dcg_interface = DCGInterface()
hybrid_language_domain = HybridLanguageDomain(language_domain, dcg_interface)
req = PlanRequest(
    domain=hybrid_language_domain,
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
