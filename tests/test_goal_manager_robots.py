"""The goal manager pauses every robot of the running plan, by name, and
releases them when planning fails."""

import json
from types import SimpleNamespace as NS

from std_msgs.msg import String

from omniplanner_msgs.msg import ConstrainedPddlGoalMsg, ConstraintFact
from omniplanner_ros.goal_manager_node import GoalManager


def _manager():
    sent = []  # (robot, verb)
    gm = NS(
        _plan_robots=set(),
        _paused=set(),
        _executor_cmd=lambda robot, verb: sent.append((robot, verb)),
        get_logger=lambda: NS(info=lambda *_: None, warning=lambda *_: None),
        _notify_operator=lambda text: None,
        _pending_goal=None,
        _running_constraints=frozenset(),
        _plan_constraints=frozenset(),
        _cache_valid=True,
    )
    for name in (
        "_plan_robots_cb",
        "_pause_executor",
        "_resume_executor",
        "_stop_executor",
        "_planner_failed_cb",
    ):
        setattr(gm, name, getattr(GoalManager, name).__get__(gm))
    return gm, sent


def _plan(gm, *robots):
    gm._plan_robots_cb(String(data=json.dumps(list(robots))))


def test_nothing_to_pause_before_the_first_plan():
    gm, sent = _manager()
    gm._pause_executor()
    gm._resume_executor()
    assert sent == []


def test_pauses_and_resumes_every_robot_of_the_plan():
    gm, sent = _manager()
    _plan(gm, "hamilton", "euclid")
    gm._pause_executor()
    assert sorted(sent) == [("euclid", "pause"), ("hamilton", "pause")]
    sent.clear()
    gm._resume_executor()
    assert sorted(sent) == [("euclid", "resume"), ("hamilton", "resume")]
    sent.clear()
    gm._resume_executor()  # nothing is paused any more
    assert sent == []


def test_new_plan_stops_a_paused_robot_it_leaves_out():
    gm, sent = _manager()
    _plan(gm, "hamilton", "euclid")
    gm._pause_executor()
    sent.clear()
    _plan(gm, "hamilton")
    # hamilton's new sequence preempts it; euclid's old plan is superseded.
    assert sent == [("euclid", "stop")]
    sent.clear()
    gm._pause_executor()
    assert sent == [("hamilton", "pause")]


def test_new_plan_without_a_pause_stops_nobody():
    gm, sent = _manager()
    _plan(gm, "hamilton", "euclid")
    _plan(gm, "hamilton")
    assert sent == []


def _goal(*forbidden):
    msg = ConstrainedPddlGoalMsg()
    msg.goal.pddl_goal = "(visited-object o14)"
    msg.constraints = [
        ConstraintFact(predicate="forbidden-poi", symbols=[s]) for s in forbidden
    ]
    return msg


def test_failed_goal_under_the_same_constraints_resumes_the_plan():
    gm, sent = _manager()
    _plan(gm, "hamilton")
    gm._pause_executor()
    sent.clear()
    gm._pending_goal = _goal()
    gm._planner_failed_cb(String(data="multi_robot_pddl_constrained: no plan"))
    assert sent == [("hamilton", "resume")]


def test_failed_goal_with_new_constraints_stops_the_plan():
    # "Avoid o14" could not be planned: the running plan may go through o14.
    gm, sent = _manager()
    _plan(gm, "hamilton")
    gm._pause_executor()
    sent.clear()
    gm._pending_goal = _goal("o14")
    gm._planner_failed_cb(String(data="multi_robot_pddl_constrained: no plan"))
    assert sent == [("hamilton", "stop")]
    assert not gm._cache_valid
