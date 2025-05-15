import ast
from dataclasses import dataclass

from dsg_pddl.dsg_pddl_interface import PddlDomain, PddlGoal
from multipledispatch import dispatch
from nlu_interface.llm_interface import LLMInterface

from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal
from omniplanner.omniplanner import RobotProblem, RobotProblems


@dataclass
class LanguageDomain:
    domain_type: str
    pddl_domain: PddlDomain = None
    llm_interface: LLMInterface = None


@dataclass
class LanguageGoal:
    robot_id: str
    command: str


@dispatch(LanguageDomain, object, dict, LanguageGoal, object)
def ground_problem(domain, dsg, robot_states, goal, feedback=None):
    if domain.domain_type == "goto_points":
        language_grounded_goal = GotoPointsGoal(
            goal_points=goal.command.split(" "), robot_id=goal.robot_id
        )
        problem_type = GotoPointsDomain()
        return ground_problem(
            problem_type, dsg, robot_states, language_grounded_goal, feedback
        )
    elif domain.domain_type == "Pddl":
        # Query the LLM & Parse the response
        response = domain.llm_interface.request_plan_specification(goal.command, dsg)
        goal_dict = ast.literal_eval(response)
        # Publish feedback to the rviz interface
        publish = feedback.plugin_feedback_collectors["language_planner"].publish[
            "llm_response"
        ]
        publish(str(goal_dict))

        problems = RobotProblems()
        for robot_name, goal in goal_dict.items():
            # Construct the PddlGoal object for the PDDL planner
            pddl_language_grounded_goal = PddlGoal(pddl_goal=goal, robot_id=robot_name)

            grounded_problem = ground_problem(
                domain.pddl_domain,
                dsg,
                robot_states,
                pddl_language_grounded_goal,
                feedback,
            )
            problems.append(RobotProblem(robot_name, grounded_problem))

        return problems

    else:
        raise Exception(f"Unexpected domain_type: {domain.domain_type}")
