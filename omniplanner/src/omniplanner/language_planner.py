from dataclasses import dataclass

from multipledispatch import dispatch

from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal


@dataclass
class LanguageDomain:
    domain_type: str
    pddl_domain: PddlDomain = None


@dataclass
class LanguageGoal:
    robot_id: str
    command: str


@dispatch(LanguageDomain, object, dict, LanguageGoal, object)
def ground_problem(domain, dsg, robot_states, goal, feedback=None):

    if domain.domain_type == "goto_point":
        language_grounded_goal = GotoPointsGoal(
            goal_points=goal.command.split(" "), robot_id=goal.robot_id
        )
        problem_type = GotoPointsDomain()
        return ground_problem(problem_type, dsg, robot_states, language_grounded_goal)
    elif domain.domain_type == "pddl":

        # TODO: make a pddl domain
        return ground_problem(domain.pddl_domain, dsg, robot_states, pddl_language_grounded_goal)

