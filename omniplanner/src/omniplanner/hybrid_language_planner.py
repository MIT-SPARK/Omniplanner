import ast
from dataclasses import dataclass
from typing import Any

from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from nlu_interface.llm_interface import LLMInterface
from plum import dispatch

from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal

from omniplanner.language_planner import LanguageGoal, LanguageDomain

# TODO - depend on h2sl pybindings
#from h2sl import DCGInterface

class HybridArchitectureException(Exception):
    pass

class DCGInterface:
    """A placeholder class implementation for an interface to a Distributed Correspondence Graph"""
    def __init__(self, debug):
        self.debug = debug

    def request_plan_specification(self, command, dsg) -> tuple[bool, str]:
        return False, None

@dataclass
class HybridLanguageDomain:

    language_domain: LanguageDomain = None
    dcg_interface: DCGInterface = None
    # For Reference, LanguageDomain:
    #   -domain_type: str
    #   -pddl_domain: PddlDomain = None
    #   -llm_interface: LLMInterface = None


@dispatch
def ground_problem(
    domain: HybridLanguageDomain,
    dsg: Any,
    robot_states: dict,
    goal: LanguageGoal,
    feedback: Any = None,
):
    if domain.language_domain.domain_type == "Pddl":
        # Handle langauge grounding via Distributed Correspondence Graphs
        success, response = domain.dcg_interface.request_plan_specification(goal.command, dsg)
        # Handle routing if failure
        if not success:
            try:
                problems = ground_problem(
                    domain.language_domain,
                    dsg,
                    robot_states,
                    goal,
                    feedback,
                )
            except Exception as e:
                raise HybridArchitectureException(f"Hybrid Architecture -- Routed to the LLM. Received an exception {e}")
            # If not debug, return problems; otherwise raise an exception
            if not domain.dcg_interface.debug:
                return problems
            raise HybridArchitectureException("Hybrid Architecture -- Successfully called LLM. Debugging, so not returning planning problems.")
        raise HybridArchitectureException("Hybrid Architecture -- Successfully called DCG.")

        ############### Below is code that should not run
        # TODO - remove the HybridArchitectureException raises once the DCGInterface exists

        # Handle return if success
        # Parse the response? 
        goal_dict = ast.literal_eval(response) # TODO -- handle the output appropriately
        # Publish feedback to the rviz interface
        if feedback is not None:
            publish = feedback.plugin_feedback_collectors["hybrid_language_planner"].publish[
                "llm_response"
            ]
            publish(str(goal_dict))

        problems = []
        for robot_name, goal in goal_dict.items():
            # Construct the PddlGoal object for the PDDL planner
            pddl_language_grounded_goal = PddlGoal(pddl_goal=goal, robot_id=robot_name)

            grounded_problem = ground_problem(
                domain.language_domain.pddl_domain,
                dsg,
                robot_states,
                pddl_language_grounded_goal,
                feedback,
            )
            problems.append(grounded_problem)

        return problems

    else:
        raise Exception(f"Unexpected domain_type: {domain.language_domain.domain_type}")
