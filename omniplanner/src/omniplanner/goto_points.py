from dataclasses import dataclass
from typing import List, overload

import numpy as np
import parse
import spark_dsg
from multipledispatch import dispatch

from omniplanner.omniplanner import PlanningDomain


def str_to_ns_value(s):
    p = parse.parse("{}({})", s)
    key = p.fixed[0]
    idx = int(p.fixed[1])
    ns = spark_dsg.NodeSymbol(key, idx)
    return ns.value


class GotoPointsDomain(PlanningDomain):
    pass


class GroundedGotoPointsProblem:
    def __init__(self, start_point, points):
        self.start_point = start_point
        self.goal_points = points


@dataclass
class GotoPointsPlan:
    plan: list


@dataclass
class GotoPointPrimitive:
    start: np.ndarray
    goal: np.ndarray


@dataclass
class GotoPointsGoal:
    goal_points: List[str]
    robot_id: str


# @dispatch(GotoPointsDomain, spark_dsg.DSG_TYPE, np.ndarray, list)
@overload
@dispatch(GotoPointsDomain, object, np.ndarray, list)
def ground_problem(domain, dsg, start, goal) -> GroundedGotoPointsProblem:
    def get_loc(symbol):
        node = dsg.find_node(str_to_ns_value(symbol))
        if node is None:
            raise Exception(f"Requested symbol {symbol} not in scene graph")
        return node.attributes.position[:2]

    referenced_points = np.array([get_loc(symbol) for symbol in goal])
    return ground_problem(
        domain, referenced_points, start, [i for i in range(len(goal))]
    )


@overload
@dispatch(GotoPointsDomain, np.ndarray, np.ndarray, list)
def ground_problem(domain, map_context, start, goal) -> GroundedGotoPointsProblem:
    point_sequence = map_context[goal]
    return GroundedGotoPointsProblem(start, point_sequence)


@dispatch(GroundedGotoPointsProblem, object)
def make_plan(grounded_problem, map_context) -> GotoPointsPlan:
    plan = []
    p = GotoPointPrimitive(
        grounded_problem.start_point, grounded_problem.goal_points[0]
    )
    plan.append(p)
    for idx in range(len(grounded_problem.goal_points) - 1):
        p = GotoPointPrimitive(
            grounded_problem.goal_points[idx], grounded_problem.goal_points[idx + 1]
        )
        plan.append(p)

    return GotoPointsPlan(plan)
