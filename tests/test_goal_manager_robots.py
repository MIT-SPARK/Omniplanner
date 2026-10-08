"""The goal manager pauses every robot of the running plan, by name."""

import json
from types import SimpleNamespace as NS

from std_msgs.msg import String

from omniplanner_ros.goal_manager_node import GoalManager


def _manager():
    sent = []  # (robot, verb)
    gm = NS(
        _plan_robots=set(),
        _paused=set(),
        _executor_cmd=lambda robot, verb: sent.append((robot, verb)),
        get_logger=lambda: NS(info=lambda *_: None, warning=lambda *_: None),
    )
    for name in ("_plan_robots_cb", "_pause_executor", "_resume_executor"):
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
