import logging
from functools import partial
from typing import Any, List, overload

from plum import dispatch

from omniplanner.omniplanner import (
    MultiRobotWrapper,
    RobotWrapper,
    SymbolicContext,
    Wrapper,
    extract,
    fmap,
    push,
)

logger = logging.getLogger(__name__)


@overload
@dispatch
def compile_plan(
    adaptors, plan_frame: str, p: SymbolicContext[List[Any]]
) -> List[RobotWrapper[Any]]:
    logger.warning(f"SymbolicContext[List[Any]] with: {type(p)}")
    return fmap(partial(compile_plan, adaptors, plan_frame), push(p))


@overload
@dispatch
def compile_plan(
    adaptors: dict, plan_frame: str, p: SymbolicContext[RobotWrapper[Any]]
) -> RobotWrapper[Any]:
    logger.warning(f"SymbolicContext[RobotWrapper[Any]] with: {type(p)}")
    robot_symbolic = push(p)
    adaptor = adaptors[robot_symbolic.name]
    return fmap(partial(compile_plan, adaptor, plan_frame), robot_symbolic)


@overload
@dispatch
def compile_plan(adaptors: dict, plan_frame: str, p: list):
    logger.warning(f"list[] with: {type(p)}")
    return fmap(partial(compile_plan, adaptors, plan_frame), p)


@overload
@dispatch
def compile_plan(adaptors: dict, plan_frame: str, p: RobotWrapper[Any]):
    return compile_plan(adaptors[p.name], plan_frame, p)


@overload
@dispatch
def compile_plan(
    adaptors: dict, plan_frame: str, p: SymbolicContext[MultiRobotWrapper[Any]]
):
    logger.warning(f"SymbolicContext[MultiRobotWrapper[Any]] with: {type(p)}")
    multirobot_symbolic = push(p)
    return compile_plan(adaptors, plan_frame, multirobot_symbolic)


@overload
@dispatch
def compile_plan(
    adaptors: dict, plan_frame: str, p: MultiRobotWrapper[SymbolicContext[Any]]
) -> MultiRobotWrapper[Any]:
    logger.warning(f"MultiRobotWrapper[SymbolicContext[Any]] with: {type(p)}")

    remapped_adaptors = {p.remap_name_to_inner(k): v for k, v in adaptors.items()}
    return fmap(partial(compile_plan, remapped_adaptors, plan_frame), p)


# This implementation lets the compilation process transparently "skip" unused wrappers
@dispatch
def compile_plan(adaptor, plan_frame: str, p: Wrapper):
    logger.debug(f"Skipping wrapper of type {type(p)}")
    return compile_plan(adaptor, plan_frame, extract(p))


# NOTE: this is the signature that you would modify to introduce a compiler for
# a new robot or Abstract Plan type. If we wanted to support different kinds of
# adaptors, that we would dispatch on adaptor here too (currently we assume all
# adaptors are for interfacing with RobotExecutor interface, but if we had e.g.
# a Phoenix interface, then that would be implemented as a separate adaptor
# type.
# @dispatch
# def compile_plan(adaptor, p: SymbolicContext[PddlPlan]):
#    return FakeExecutorAction(adaptor.robot_name, p.value.fake_action)


@overload
@dispatch
def collect_plans(p: List[RobotWrapper[Any]]):
    # TODO: should we warn about duplicate names in robot wrappers?
    return {r.name: r.value for r in p}


@dispatch
def collect_plans(p: RobotWrapper[Any]):
    return {p.name: p.value}
