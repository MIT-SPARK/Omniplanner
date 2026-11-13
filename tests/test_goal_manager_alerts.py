"""The goal manager stops and tells the operator when the world breaks the plan."""

import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from std_msgs.msg import String  # noqa: E402

GOAL = "(visited-object o20)"


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _running(h):
    """hilbert is driving a plan for GOAL."""
    goal = h.send(GOAL, robot_id="hilbert")
    h.plan(hilbert=("p1", 3, ["p4", "o20"]))
    h.goals.clear()
    h.sent.clear()
    return goal


def test_failed_operator_goal_stops_and_says_why(h):
    _running(h)
    h.send("(visited-object o21)", robot_id="hilbert")
    h.failed("Cannot satisfy the goal and constraints: o21 cannot be reached")
    assert ("hilbert", "stop") in h.sent and ("hilbert", "resume") not in h.sent
    text = h.texts[-1]
    assert "hilbert's goal '(visited-object o21)'" in text
    assert "o21 cannot be reached" in text
    assert "Stopped hilbert; waiting for a new goal" in text
    assert "multi_robot_pddl_constrained" not in text


def test_failed_scene_change_replan_stops_and_tells_the_operator(h):
    _running(h)
    h.node._replan_request_cb(
        String(data="scene changed: o20 removed before the visit")
    )
    assert [g.goal.pddl_goal for g in h.goals] == [GOAL]
    h.node._planner_failed_cb(
        String(data="multi_robot_pddl_constrained: Goal symbol(s) not in the DSG: o20")
    )
    assert ("hilbert", "stop") in h.sent  # the old plan drives to o20
    assert ("hilbert", "resume") not in h.sent
    text = h.texts[-1]
    assert "o20 removed" in text and "not in the DSG: o20" in text
    assert "Stopped hilbert" in text and "multi_robot_pddl_constrained" not in text


def test_a_scene_change_stops_the_robots_even_with_resume_on_failure(tmp_path):
    h = Harness(tmp_path, resume_on_plan_failure=True)
    _running(h)
    h.node._replan_request_cb(
        String(data="scene changed: o20 removed before the visit")
    )
    h.failed("no plan")
    assert ("hilbert", "stop") in h.sent and ("hilbert", "resume") not in h.sent
    h.close()


def test_a_replan_that_succeeded_is_not_held_against_a_later_failure(h):
    _running(h)
    h.node._replan_request_cb(String(data="scene changed: o4 moved"))
    h.plan(hilbert=("p2", 3, ["o20"]))  # the new plan arrived
    h.send("(visited-object o21)", robot_id="hilbert")
    h.failed("no plan")
    assert "Cannot repair" not in h.texts[-1]


def test_the_same_goal_is_not_replanned_on_the_same_map(h):
    h.map_version(4)
    _running(h)
    h.send(GOAL, robot_id="hilbert")
    assert h.goals == [] and h.sent == []  # in progress: the running plan goes on
    assert "already in progress (hilbert)" in h.texts[-1]


def test_the_same_goal_is_replanned_once_the_map_changed(h):
    h.map_version(4)
    _running(h)
    h.map_version(5)  # e.g. "SGET: delete t444"
    h.send(GOAL, robot_id="hilbert")
    assert len(h.goals) == 1  # planned again, on the edited map
    assert any("planned on an older map" in m for m in h.logs)
    # The new plan is on the edited map, so the next resend is in progress again.
    h.plan(hilbert=("p2", 3, ["o20"]))
    h.send(GOAL, robot_id="hilbert")
    assert len(h.goals) == 1
