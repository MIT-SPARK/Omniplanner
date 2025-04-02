# examples of planning combinations:
from dataclasses import dataclass
from typing import Any
from multipledispatch import dispatch

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


class DispatchException(Exception):
    def __init__(self, function_name, *objects):
        arg_type_string = ", ".join(map(lambda x: x.__name__, map(type, objects)))
        super().__init__(
            f"No matching specialization for {function_name}({arg_type_string})"
        )


@dispatch(PlanningDomain, object, object, PlanningGoal)
def ground_problem(domain, map_context, intial_state, goal) -> GroundedProblem:
    raise DispatchException(ground_problem, domain, map_context, goal)


@dispatch(GroundedProblem, object)
def make_plan(grounded_problem, map_context) -> Plan:
    raise DispatchException(make_plan, grounded_problem, map_context)


def full_planning_pipeline(plan_request: PlanRequest, map_context: Any):
    grounded_problem = ground_problem(
        plan_request.domain, map_context, plan_request.initial_state, plan_request.goal
    )
    plan = make_plan(grounded_problem, map_context)
    return plan
