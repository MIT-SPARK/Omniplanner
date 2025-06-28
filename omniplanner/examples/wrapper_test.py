from dataclasses import dataclass
from functools import partial
from typing import Any, List, overload

from plum import dispatch

from omniplanner.functor import fmap
from omniplanner.omniplanner import RobotWrapper, SymbolicContext, Wrapper

# Wrapper typeclass:
# extract -- return the wrapped value
# with_new_value -- set the wrapper's value to a new value
# push -- (get automatically from extract and with_new_value) -- swap the order of wrappers
# You can choose which "depths" to swap in a stack of wrappers by composing fmap: fmap(push, wrapper) will leave the top-layer wrapper alone and swap the next two layers


@overload
@dispatch
def extract(x: RobotWrapper):
    return x.value


@dispatch
def extract(x: SymbolicContext):
    return x.value


@overload
@dispatch
def with_new_value(x: RobotWrapper, value):
    return RobotWrapper(x.name, value)


@dispatch
def with_new_value(x: SymbolicContext, value):
    return SymbolicContext(x.context, value)


@dispatch
def push(x: Wrapper):
    """Really we want to specify Wrapper[Wrapper[Any]], but I couldn't get type inference to work"""
    inner_wrapper = extract(x)
    if not isinstance(inner_wrapper, Wrapper) and not isinstance(inner_wrapper, List):
        raise TypeError("Can only push type Wrapper[Wrapper[T]]")
    return fmap(lambda val: with_new_value(x, val), inner_wrapper)


a = RobotWrapper("jasper", 1)
print(a)

b = SymbolicContext({"jasper": {"robot_type": "spot"}}, a)

print(b)

c = push(b)

print(c)

t = RobotWrapper("jasper", [1, 2, 3])
print(push(t))


@dataclass
class TempPddlPlan:
    fake_action: int


@dataclass
class FakeTsp:
    order: list


@dataclass
class FakeAction:
    name: str
    action: int


@dataclass
class Adaptor:
    robot_name: str


@dataclass
class Plan:
    pass


@dataclass
class MyListPlan(list, Plan):
    val: 1


@overload
@dispatch
def compile_plan(adaptors, p: SymbolicContext[List[Any]]) -> List[RobotWrapper[Any]]:
    return fmap(partial(compile_plan, adaptors), push(p))


@overload
@dispatch
def compile_plan(adaptors, p: SymbolicContext[RobotWrapper[Any]]) -> RobotWrapper[Any]:
    robot_symbolic = push(p)
    adaptor = adaptors[robot_symbolic.name]
    return fmap(partial(compile_plan, adaptor), robot_symbolic)


@overload
@dispatch
def compile_plan(adaptor, p: SymbolicContext[int]):
    return str(p.value)


# NOTE: this is the signature that you would modify to introduce a compiler for
# a new robot or Abstract Plan type. If we wanted to support different kinds of
# adaptors, that we would dispatch on adaptor here too (currently we assume all
# adaptors are for interfacing with RobotExecutor interface, but if we had e.g.
# a Phoenix interface, then that would be implemented as a separate adaptor
# type.
@overload
@dispatch
def compile_plan(adaptor, p: SymbolicContext[TempPddlPlan]):
    return FakeAction(adaptor.robot_name, p.value.fake_action)


# Test analogous to the compiler for TspPlan
@overload
@dispatch
def compile_plan(adaptor, p: Wrapper):
    return compile_plan(adaptor, extract(p))


@overload
@dispatch
def compile_plan(adaptor, p: FakeTsp):
    return FakeAction(adaptor.robot_name + "_tsp", p.order)


@dispatch
def compile_plan(adaptor, p: SymbolicContext[MyListPlan]):
    return FakeAction(adaptor.robot_name, p.value.val)


@overload
@dispatch
def collect_plans(p: List[RobotWrapper[Any]]):
    # TODO: should we warn about duplicate names in robot wrappers?
    return {r.name: r.value for r in p}


@dispatch
def collect_plans(p: RobotWrapper[Any]):
    return {p.name: p.value}


adaptors = {"jasper": Adaptor("jasper")}

plan1 = SymbolicContext({"p1": {"class": "tree"}}, 1)

print("compiled_plan 1: ")
print(compile_plan(adaptors["jasper"], plan1))
#
# plan2 = SymbolicContext({"p1": {"class": "tree"}}, [RobotWrapper("jasper", 1)])
#
# print("compiled_plan 2: ")
# print(compile_plan(plan2, adaptors))
# print("collected_plan 2: ")
# print(collect_plans(compile_plan(adaptors, plan2)))


plan3 = SymbolicContext(
    {"p1": {"class": "tree"}}, RobotWrapper("jasper", TempPddlPlan(42))
)
print(compile_plan(adaptors, plan3))
print(collect_plans(compile_plan(adaptors, plan3)))

# Test "skipping" the symbolic context wrapper
plan4 = SymbolicContext(
    {"p1": {"class": "tree"}}, RobotWrapper("jasper", FakeTsp([1, 2, 3]))
)
print(compile_plan(adaptors, plan4))
print(collect_plans(compile_plan(adaptors, plan4)))


# Tricky edge case where the plan is derived from a list
plan5 = SymbolicContext(
    {"p1": {"class": "tree"}}, RobotWrapper("jasper", MyListPlan(12))
)
print(compile_plan(adaptors, plan5))
print(collect_plans(compile_plan(adaptors, plan5)))
