# examples of planning combinations:
import logging
from collections import UserDict
from dataclasses import dataclass
from typing import Any, Callable, List, overload

from dsg_pddl.pddl_utils import pddl_char_to_dsg_char
from plum import dispatch
from spark_dsg import NodeSymbol

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


class Wrapper(FunctorTrait):
    pass


@overload
@dispatch
def fmap(fn: Callable, wrapper: Wrapper):
    return with_new_value(wrapper, fn(extract(wrapper)))


@dispatch
def push(x: Wrapper):
    """Really we want to specify Wrapper[Wrapper[Any]], but I couldn't get type inference to work"""
    inner_wrapper = extract(x)
    if not isinstance(inner_wrapper, Wrapper) and not isinstance(inner_wrapper, List):
        raise TypeError("Can only push type Wrapper[Wrapper[T]]")
    return fmap(lambda val: with_new_value(x, val), inner_wrapper)


@dispatchable_parametric
# class RobotWrapper[T](FunctorTrait):
class RobotWrapper[T](Wrapper):
    name: str
    value: T


@overload
@dispatch
def fmap(fn: Callable, robot_wrapper: RobotWrapper):
    return RobotWrapper(robot_wrapper.name, fn(robot_wrapper.value))


@overload
@dispatch
def extract(x: RobotWrapper):
    return x.value


@overload
@dispatch
def with_new_value(x: RobotWrapper, value):
    return RobotWrapper(x.name, value)


def string_as_nodesymbol(string):
    try:
        c = pddl_char_to_dsg_char(string[0])
        return NodeSymbol(c, int(string[1:]))
    except Exception:
        return None


@dispatchable_parametric
class SymbolicContext[T](Wrapper):
    context: dict
    value: T


@dispatch
def extract(x: SymbolicContext):
    return x.value


@dispatch
def with_new_value(x: SymbolicContext, value):
    return SymbolicContext(x.context, value)


@dispatch
def fmap(fn: Callable, v: SymbolicContext):
    return SymbolicContext(v.context, fn(v.value))


class DsgNodeContext(UserDict):
    def __init__(self, dsg_dict, external_dict):
        super().__init__()
        self.dsg_dict = dsg_dict
        self.external_dict = external_dict
        self.data = dsg_dict | external_dict

    def __setitem__(self, key, value):
        if key in self.dsg_dict:
            raise Exception("Cannot override DSG context")
        self.external_dict[key] = value
        self.data[key] = value


class DsgContextProvider(dict):
    def __init__(self, dsg):
        self.dsg = dsg

    def __getitem__(self, key):
        ns = string_as_nodesymbol(key)
        symbol_info = {}
        if ns is not None:
            node = self.dsg.find_node(ns)
            if node is not None:
                symbol_info["position"] = node.attributes.position
                node_layer = node.layer.layer
                node_partition = node.layer.partition
                category = self.dsg.get_labelspace(
                    node_layer, node_partition
                ).get_node_category(node)
                symbol_info["semantic_label"] = category

        try:
            explicit_symbols = dict.__getitem__(self, key)
        except KeyError:
            dict.__setitem__(self, key, {})
            explicit_symbols = dict.__getitem__(self, key)
        return DsgNodeContext(symbol_info, explicit_symbols)

    def __setitem__(self, key, val):
        if not isinstance(val, dict):
            raise TypeError(
                "Context value must be a dictionary, e.g., {context_key: context_value}"
            )
        dict.__setitem__(self, key, val)

    def __contains__(self, key):
        if dict.__contains__(self, key) and len(dict.__getitem__(self, key)) > 0:
            return True

        ns = string_as_nodesymbol(key)
        if ns is None:
            return False
        return self.dsg.find_node(ns) is not None


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

    # TODO: it would be nice if we could incorporate additional symbol context
    # added during grounding...
    
    dsg_context = DsgContextProvider(map_context)
    contextualized_problem = SymbolicContext(dsg_context, grounded_problem)
    import dsg_pddl.dsg_pddl_planning  # noqa: F401
    plan = make_plan(contextualized_problem, map_context)
    logger.debug(f"Made plan {plan}")
    return plan