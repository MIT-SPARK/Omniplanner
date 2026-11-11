"""The goal manager pauses every robot of the running plan, by name, and
stops them when planning fails."""

import json

import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from std_msgs.msg import String  # noqa: E402


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _plan(h, *robots):
    h.node._plan_robots_cb(String(data=json.dumps(list(robots))))


def test_nothing_to_pause_before_the_first_plan(h):
    h.node._pause_executor()
    h.node._resume_executor()
    assert h.sent == []


def test_pauses_and_resumes_every_robot_of_the_plan(h):
    _plan(h, "hamilton", "euclid")
    h.node._pause_executor()
    assert sorted(h.sent) == [("euclid", "pause"), ("hamilton", "pause")]
    h.sent.clear()
    h.node._resume_executor()
    assert sorted(h.sent) == [("euclid", "resume"), ("hamilton", "resume")]
    h.sent.clear()
    h.node._resume_executor()  # nothing is paused any more
    assert h.sent == []


def test_new_plan_stops_a_paused_robot_it_leaves_out(h):
    _plan(h, "hamilton", "euclid")
    h.node._pause_executor()
    h.sent.clear()
    _plan(h, "hamilton")
    # hamilton's new sequence preempts it; euclid's old plan is superseded.
    assert h.sent == [("euclid", "stop")]
    h.sent.clear()
    h.node._pause_executor()
    assert h.sent == [("hamilton", "pause")]


def test_new_plan_without_a_pause_stops_nobody(h):
    _plan(h, "hamilton", "euclid")
    _plan(h, "hamilton")
    assert h.sent == []


def test_failed_goal_stops_the_plan_and_waits(h):
    # gauss is a Husky: it cannot pause, but it is stopped.
    _plan(h, "hamilton", "gauss")
    h.send("(visited-object o14)")
    assert h.sent == [("hamilton", "pause")]
    h.sent.clear()
    h.failed("no plan")
    assert sorted(h.sent) == [("gauss", "stop"), ("hamilton", "stop")]
    assert "Stopped gauss and hamilton; waiting for a new goal" in h.texts[-1]


def test_failed_goal_resumes_when_configured_to(tmp_path):
    h = Harness(tmp_path, resume_on_plan_failure=True)
    _plan(h, "hamilton")
    h.send("(visited-object o14)")
    h.sent.clear()
    h.failed("no plan")
    assert h.sent == [("hamilton", "resume")]
    h.close()


def test_failed_goal_with_new_constraints_stops_even_when_resuming(tmp_path):
    h = Harness(tmp_path, resume_on_plan_failure=True)
    _plan(h, "hamilton")
    h.send("(visited-object o14)", forbidden=["o14"])
    h.sent.clear()
    h.failed("no plan")
    assert h.sent == [("hamilton", "stop")]
    h.close()


def test_failed_goal_with_new_constraints_stops_the_plan(h):
    # "Avoid o14" could not be planned: the running plan may go through o14.
    _plan(h, "hamilton")
    h.send("(visited-object o14)", forbidden=["o14"])
    h.sent.clear()
    h.failed("no plan")
    assert h.sent == [("hamilton", "stop")]
    assert "made without these constraints" in h.texts[-1]
