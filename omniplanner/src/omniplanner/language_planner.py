from dataclasses import dataclass

from multipledispatch import dispatch

from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal
from dsg_pddl.dsg_pddl_interface import PddlDomain, PddlGoal

import ast
from nlu_interface.llm_interface import LLMInterface

from std_msgs.msg import String


@dataclass
class LanguageDomain:
    domain_type: str
    pddl_domain: PddlDomain = None
    llm_interface: LLMInterface = None

@dataclass
class LanguageGoal:
    robot_id: str
    command: str

def inject_labelspaces(dsg):
  import yaml
  labelspace_metadata = {}
  with open("/colcon_ws/src/awesome_dcist_t4/nlu_interface/nlu_interface/resources/labelspaces/ade20k_mit_label_space.yaml", 'r') as file:
    object_labelspace = yaml.safe_load(file)
  id_to_label = {item["label"] : item["name"] for item in object_labelspace["label_names"]}
  labelspace_metadata["object_labelspace"] = id_to_label
  with open("/colcon_ws/src/awesome_dcist_t4/nlu_interface/nlu_interface/resources/labelspaces/scene_camp_buckner_label_space.yaml", 'r') as file:
    region_labelspace = yaml.safe_load(file)
  id_to_label = {item["label"] : item["name"] for item in region_labelspace["label_names"]}
  labelspace_metadata["region_labelspace"] = id_to_label
  dsg.metadata.set( labelspace_metadata )
  return dsg

@dispatch(LanguageDomain, object, dict, LanguageGoal, object)
def ground_problem(domain, dsg, robot_states, goal, feedback=None):

    if domain.domain_type == "goto_point":
        language_grounded_goal = GotoPointsGoal(
            goal_points=goal.command.split(" "), robot_id=goal.robot_id
        )
        problem_type = GotoPointsDomain()
        return ground_problem(problem_type, dsg, robot_states, language_grounded_goal)
    elif domain.domain_type == "Pddl":
        # Inject labelspaces -- This is temporary for testing purposes only.
        dsg = inject_labelspaces(dsg)
        # Query the LLM & Parse the response
        response = domain.llm_interface.request_plan_specification(
            goal.command,
            dsg
        )
        goal_dict = ast.literal_eval(response)
        # Publish feedback to the rviz interface
        publish = feedback.plugin_feedback_collectors["language_planner"].publish["llm_response"]
        publish( str(goal_dict) )
        # Construct the PddlGoal object for the PDDL planner
        pddl_language_grounded_goal = PddlGoal(pddl_goal=goal_dict["spot"], robot_id="spot")
        return ground_problem(domain.pddl_domain, dsg, robot_states, pddl_language_grounded_goal, feedback)
    else:
      raise Exception(f"Unexpected domain_type: {domain.domain_type}")

