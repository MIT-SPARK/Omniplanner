"""Goals for different robots sent back to back (2026-11-11 rig test): the
planner drops a plan still being computed when a newer goal arrives, so the
goal manager hands it one goal at a time when they are for different robots."""

import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402

EUCLID, HAMILTON = "(visited-object-by euclid o0)", "(visited-object-by hamilton o2)"


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _plan(h, **robots):
    """The planner's answer, then the goal manager's next timer tick."""
    h.plan(**robots)
    h.node._send_queued()


def _goals(h):
    return [(g.goal.pddl_goal, h.use(g)) for g in h.goals]


def test_two_goals_for_different_robots_back_to_back_both_get_a_plan(h):
    h.send(EUCLID)
    h.send(HAMILTON)  # before euclid's plan is back
    h.node._send_queued()  # a tick while euclid's goal is still being planned
    # only euclid's is with the planner: a second one would drop its plan
    assert _goals(h) == [(EUCLID, ["euclid"])]
    assert any("waits" in m and "hamilton" in m for m in h.logs)

    _plan(h, euclid=("pe", 2, ["o0"]))
    assert _goals(h) == [(EUCLID, ["euclid"]), (HAMILTON, ["hamilton"])]
    assert h.sent == []  # euclid is not paused for hamilton's goal
    _plan(h, hamilton=("ph", 2, ["o2"]))

    assert h.node._robots["euclid"].plan_id == "pe"
    assert h.node._robots["hamilton"].plan_id == "ph"
    assert h.node._standing == {"euclid": [EUCLID], "hamilton": [HAMILTON]}
    assert set(h.node._mission) == {EUCLID, HAMILTON}
    assert h.node._queued == [] and h.node._pending is None


def test_a_waiting_goal_keeps_its_robots_coverage_apart(h):
    """euclid's coverage comes after its plan, when hamilton's goal is
    already with the planner: it is still judged against euclid's goal."""
    h.send("(visited-object o0)", use=["euclid"])
    h.send(HAMILTON)
    _plan(h, euclid=("pe", 2, ["o0"]))
    assert h.node._pending.robots == ["hamilton"]
    assert h.node._robots["euclid"].conjuncts == ["(visited-object o0)"]


def test_two_goals_for_the_same_robot_back_to_back_only_the_second_is_planned(h):
    h.send(EUCLID)
    h.send("(visited-object-by euclid o5)")
    # as before: the newer goal goes straight out and replaces the older
    assert [g.goal.pddl_goal for g in h.goals] == [
        EUCLID,
        "(visited-object-by euclid o5)",
    ]
    assert h.node._queued == []
    _plan(h, euclid=("p5", 2, ["o5"]))
    assert h.node._standing == {"euclid": ["(visited-object-by euclid o5)"]}


def test_a_team_goal_supersedes_a_scoped_goal_being_planned_and_the_queue(h):
    h.send(EUCLID)
    h.send(HAMILTON)  # waits
    h.send("(visited-object o7)")  # the team's
    assert [g.goal.pddl_goal for g in h.goals] == [EUCLID, "(visited-object o7)"]
    assert h.node._queued == []
    _plan(h, hilbert=("p7", 2, ["o7"]))
    assert len(h.goals) == 2  # hamilton's goal was replaced, not sent after
    assert h.node._mission == ["(visited-object o7)"]


def test_a_newer_goal_for_a_waiting_robot_replaces_the_waiting_one(h):
    h.send(EUCLID)
    h.send(HAMILTON)
    h.send("(visited-object-by hamilton o9)")
    _plan(h, euclid=("pe", 2, ["o0"]))
    assert [g.goal.pddl_goal for g in h.goals] == [
        EUCLID,
        "(visited-object-by hamilton o9)",
    ]


def test_a_waiting_goal_is_sent_when_the_goal_before_it_fails(h):
    h.send(EUCLID)
    h.send(HAMILTON)
    h.failed("Goal symbol(s) not in the DSG: o0")
    h.node._send_queued()
    assert "Could not plan" in h.texts[-1] and "o0" in h.texts[-1]
    assert _goals(h)[-1] == (HAMILTON, ["hamilton"])
    _plan(h, hamilton=("ph", 2, ["o2"]))
    assert h.node._robots["hamilton"].plan_id == "ph"


def test_three_robots_back_to_back_go_one_after_another(h):
    h.send(EUCLID)
    h.send(HAMILTON)
    h.send("(visited-object-by hilbert o4)")
    assert len(h.goals) == 1
    _plan(h, euclid=("pe", 2, ["o0"]))
    assert len(h.goals) == 2
    _plan(h, hamilton=("ph", 2, ["o2"]))
    assert len(h.goals) == 3
    _plan(h, hilbert=("pi", 2, ["o4"]))
    assert sorted(h.node._standing) == ["euclid", "hamilton", "hilbert"]


def test_a_planner_that_never_answers_does_not_hold_the_queue_for_ever(h):
    h.send(EUCLID)
    h.send(HAMILTON)
    h.t += 30
    h.node._watchdog_cb()
    assert len(h.goals) == 1  # still waiting
    h.t += 31
    h.node._watchdog_cb()
    assert _goals(h)[-1] == (HAMILTON, ["hamilton"])
    assert any("no answer from the planner" in t.lower() for t in h.texts)


def test_holding_goals_can_be_turned_off(tmp_path):
    h = Harness(tmp_path, queue_timeout_s=0.0)
    h.send(EUCLID)
    h.send(HAMILTON)
    assert len(h.goals) == 2  # as before this change
    h.close()


def test_the_waiting_goal_is_decided_after_the_plan_not_inside_its_callback(h):
    """plan_robots can be handled before the plan's action sequences. Decided
    inside that callback, hamilton's goal would find every robot idle, start a
    new mission and forget euclid's goal."""
    h.send(EUCLID)
    h.send(HAMILTON)
    h.plan(euclid=("pe", 2, ["o0"]))
    assert len(h.goals) == 1  # not yet: on the next tick
    h.node._send_queued()
    (_, second) = h.goals
    assert not any(c.predicate == "new-mission" for c in second.constraints)
    assert h.node._standing["euclid"] == [EUCLID]
