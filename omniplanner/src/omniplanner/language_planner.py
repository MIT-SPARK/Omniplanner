from dataclasses import dataclass
from multipledispatch import dispatch
from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal



class LanguageDomain:
    pass

@dataclass
class LanguageGoal:
    robot_id: str
    command: str


@dispatch(LanguageDomain, object, dict, LanguageGoal)
def ground_problem(domain, dsg, robot_states, goal):


    ##############################
    ### TODO: this block is where we take the goal string from the goal, and any other goal or scene graph information,
    ### and turn it into a specific planning problem. The LLM can decide to set the problem type, which
    ### implicitly controls which downstream planner is used.

    language_grounded_goal = GotoPointsGoal(goal_points=goal.command.split(' '), robot_id=goal.robot_id)
    problem_type = GotoPointsDomain()
    ##############################

    return ground_problem(problem_type, dsg, robot_states, language_grounded_goal)

