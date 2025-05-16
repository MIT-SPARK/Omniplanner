# examples of planning combinations:
import logging
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Dict, overload

from multipledispatch import dispatch

logger = logging.getLogger(__name__)


# 0. GotoPoints Domain, ListOfPoints goal, PointLocations
# 0.5 GotoPoints Domain, ListOfSymbols goal, DSG
# 1. PDDL domain, PDDL goal, DSG
# 2. PDDLStream domain, PDDL goal, DSG
# 3. TSP domain, PDDL goal, DSG
# 4. TSP domain, PDDL goal, Dictionary of {symbol: position}
# 5. LTL domain, LTL goal, DSG
# 6. LTL domain, LTL goal, networkx graph
# 7. BSP domain, BSP goal, DSG

# And then, all of the above but the goal is given in natural language


def tokenize_lisp(string):
    return string.replace("\n", "").replace("(", "( ").replace(")", " )").split()


def get_lisp_ast(toks):
    t = toks.pop(0)
    if t == "(":
        exp = ()
        while toks[0] != ")":
            exp += (get_lisp_ast(toks),)
        toks.pop(0)
        return exp
    elif t != ")":
        return t
    else:
        raise SyntaxError("Unexpected )")


def lisp_string_to_ast(string):
    return get_lisp_ast(tokenize_lisp(string))


class PddlGoal(tuple):
    def __new__(self, t):
        if isinstance(t, str):
            t = lisp_string_to_ast(t)
        return tuple.__new__(PddlGoal, t)


class NaturalLanguageGoal(str):
    pass


class PlanningDomain:
    pass


class PlanningGoal:
    pass


class ExecutionInterface:
    pass


@dataclass
class PlanRequest:
    domain: PlanningDomain
    goal: PlanningGoal
    robot_states: dict


@dataclass
class GroundedProblem:
    pass
    # initial_state: Any
    # goal_states: Any


@dataclass
class Plan:
    pass


@dataclass
class RobotPlanningDomain:
    robot_name: str
    domain: Any


@dataclass
class RobotProblem:
    robot_name: str
    problem: Any


class RobotProblems(list):
    pass


class DispatchException(Exception):
    def __init__(self, function_name, *objects):
        arg_type_string = ", ".join(map(lambda x: x.__name__, map(type, objects)))
        super().__init__(
            f"No matching specialization for {function_name}({arg_type_string})"
        )


@overload
@dispatch(PlanningDomain, object, object, PlanningGoal, object)
def ground_problem(
    domain, map_context, intial_state, goal, feedback=None
) -> GroundedProblem:
    raise DispatchException(ground_problem, domain, map_context, goal, feedback)


@overload
@dispatch(GroundedProblem, object)
def make_plan(grounded_problem, map_context) -> Plan:
    raise DispatchException(make_plan, grounded_problem, map_context)


@dispatch(RobotPlanningDomain, object, object, object, object)
def ground_problem(
    domain, map_context, initial_state, goal, feedback=None
) -> RobotProblem:
    logger.warning("grounding RobotPlanningDomain")
    grounded_problem = ground_problem(
        domain.domain, map_context, initial_state, goal, feedback
    )
    return RobotProblem(domain.robot_name, grounded_problem)


@overload
@dispatch(RobotProblems, object)
def make_plan(problems, map_context) -> Dict[str, Plan]:
    logger.warning("Solving List of {type(problems[0])}")
    name_to_plan = {}
    for p in problems:
        logger.warning(f"Making plan for {p.robot_name}")
        name_to_plan |= make_plan(p, map_context)
    return name_to_plan


@dispatch(RobotProblem, object)
def make_plan(grounded_problem, map_context) -> Dict[str, Plan]:
    logger.warning("Solving RobotProblem")
    plan = make_plan(grounded_problem.problem, map_context)
    return {grounded_problem.robot_name: plan}


def full_planning_pipeline(plan_request: PlanRequest, map_context: Any, feedback=None):
    grounded_problem = ground_problem(
        plan_request.domain,
        map_context,
        plan_request.robot_states,
        plan_request.goal,
        feedback,
    )
    plan = make_plan(grounded_problem, map_context)
    return plan


@singledispatch
def compile_plan(plan, plan_id, robot_name, frame_id):
    raise NotImplementedError(f"No `compile_plan` implementation for {type(plan)}")
