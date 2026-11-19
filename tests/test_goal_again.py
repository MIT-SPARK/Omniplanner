"""A goal the operator sends again, in every form it takes.

"Already done" is the right answer to one thing only: the rest of a mission
that is being replanned. An operator who names a place again wants a robot
there again -- whether the goal is the same one, has a new part next to the
old one, names a region or an object, or arrives while other robots work.
"""

import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from std_msgs.msg import String  # noqa: E402


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _done(h, robot):
    h.node._path_status_cb(
        robot, String(data='{"plan_id": "(path)", "status": "done"}')
    )


def _gauss_did(h, goal, coverage):
    """euclid is busy on a long task; gauss gets `goal` and finishes it."""
    h.send("(visited-object-by euclid o1)")
    h.plan(euclid=("pe", 3, ["o1"]))
    h.send(goal)
    h.plan(gauss=("", 0, coverage))
    _done(h, "gauss")
    assert "gauss" not in h.node._robots and "euclid" in h.node._robots
    h.goals.clear()
    h.texts.clear()


def _planned(h):
    return [g.goal.pddl_goal for g in h.goals]


AGAIN = [
    ("(visited-place-by gauss t5)", ["t5"]),
    ("(visited-object-by gauss o3)", ["o3"]),
    ("(visited-region-by gauss r2)", ["r2", "t7"]),
    ("(explored-region-by gauss r2)", ["r2", "t7", "t8"]),
    ("(and (visited-place-by gauss t5) (visited-place-by gauss t6))", ["t5", "t6"]),
    ("(or (visited-place-by gauss t5) (visited-place-by gauss t6))", ["t5"]),
]


@pytest.mark.parametrize("goal,coverage", AGAIN, ids=[g for g, _ in AGAIN])
def test_the_same_goal_again_is_planned_again_while_another_robot_works(
    h, goal, coverage
):
    _gauss_did(h, goal, coverage)
    h.send(goal)
    assert not any("already done" in t for t in h.texts), h.texts
    assert h.goals, "nothing was sent to the planner"
    assert _planned(h)[-1] == goal


def test_an_old_place_next_to_a_new_one_is_planned_with_it(h):
    """As on main: "t5 and t9" with t5 done plans t9; only a goal made of
    finished robots' named visits alone is "go again"."""
    _gauss_did(h, "(visited-place-by gauss t5)", ["t5"])
    h.send("(and (visited-place-by gauss t5) (visited-place-by gauss t9))")
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(visited-place-by gauss t9)"


def test_an_old_place_next_to_another_robots_new_task_is_planned_with_it(h):
    _gauss_did(h, "(visited-place-by gauss t5)", ["t5"])
    h.send("(and (visited-place-by gauss t5) (visited-object-by hamilton o2))")
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(visited-object-by hamilton o2)"


def test_a_replan_still_leaves_out_what_the_mission_has_done(h):
    """The one place "already done" belongs: a replan of the running mission."""
    h.send("(and (visited-object o1) (visited-place-by gauss t5))")
    h.plan(euclid=("pe", 3, ["o1"]), gauss=("", 0, ["t5"]))
    _done(h, "gauss")
    h.goals.clear()
    h.node._replan_request_cb(String(data="a place on euclid's route was removed"))
    (again,) = h.goals
    assert "t5" not in again.goal.pddl_goal and "o1" in again.goal.pddl_goal


def test_the_running_mission_sent_again_does_not_send_back_a_robot_that_finished(h):
    """The console's "Replan same goal" after a scene change re-sends the
    running mission: hamilton, done with its part, is not sent back to it."""
    goal = "(and (visited-object-by hamilton o2) (visited-object-by euclid o5))"
    h.send(goal)
    h.plan(hamilton=("ph", 1, ["o2"]), euclid=("pe", 3, ["o5"]))
    h.done("hamilton", "ph")
    h.goals.clear()
    h.send(goal)
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(visited-object-by euclid o5)"


def test_a_finished_robots_goal_sent_again_after_the_mission_moved_on(h):
    """hamilton's own goal, finished, sent again while euclid works on
    something else: hamilton goes again."""
    h.send("(visited-object-by hamilton o2)")
    h.plan(hamilton=("ph", 1, ["o2"]))
    h.done("hamilton", "ph")
    h.send("(visited-object-by euclid o5)")
    h.plan(euclid=("pe", 3, ["o5"]))
    h.goals.clear()
    h.send("(visited-object-by hamilton o2)")
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(visited-object-by hamilton o2)"


def test_a_visit_no_robot_is_named_for_stays_done_while_the_fleet_works(h):
    """As on main: a team visit sent again while robots are still on the
    mission is already done; only a robot named for a visit goes again."""
    _gauss_did(h, "(visited-place t5)", ["t5"])
    h.send("(visited-place t5)")
    assert h.goals == [] and any("already done" in t for t in h.texts)
