# examples of planning combinations:
import logging
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable, overload

from plum import dispatch

from omniplanner.functor import Functor, FunctorTrait, dispatchable_parametric

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


@dispatchable_parametric
class RobotWrapper[T](FunctorTrait):
    name: str
    value: T


@dispatch
def fmap(fn: Callable, robot_wrapper: RobotWrapper):
    return RobotWrapper(robot_wrapper.name, fn(robot_wrapper.value))


class DispatchException(Exception):
    def __init__(self, function_name, *objects):
        arg_type_string = ", ".join(map(lambda x: x.__name__, map(type, objects)))
        super().__init__(
            f"No matching specialization for {function_name}({arg_type_string})"
        )


@dispatch
def ground_problem(
    domain: PlanningDomain,
    map_context: Any,
    intial_state: Any,
    goal: PlanningGoal,
    feedback: Any = None,
) -> GroundedProblem:
    raise DispatchException(ground_problem, domain, map_context, goal, feedback)


@overload
@dispatch
def make_plan(grounded_problem: GroundedProblem, map_context: Any) -> Plan:
    raise DispatchException(make_plan, grounded_problem, map_context)


@dispatch
def make_plan(grounded_problem: Functor, map_context: Any):
    return fmap(lambda e: make_plan(e, map_context), grounded_problem)


def full_planning_pipeline(plan_request: PlanRequest, map_context: Any, feedback=None):
    grounded_problem = ground_problem(
        plan_request.domain,
        map_context,
        plan_request.robot_states,
        plan_request.goal,
        feedback,
    )
    logger.debug("Grounded Problem")
    plan = make_plan(grounded_problem, map_context)
    logger.debug(f"Made plan {plan}")
    return plan


@singledispatch
def compile_plan(plan, plan_id, robot_name, frame_id):
    raise NotImplementedError(f"No `compile_plan` implementation for {type(plan)}")
