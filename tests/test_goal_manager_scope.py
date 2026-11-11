"""Willow's goal manager, from the 2026-11-11 robot test: a goal naming some
robots changes only theirs; a pause never sticks to a newer plan; where a robot
is is never remembered as done."""

import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def test_a_robot_position_goal_is_planned_again_after_the_robot_moved(h):
    """(at-place euclid t7) held once (empty plans), then euclid was sent
    elsewhere on the same map: asking for it again must go to the planner."""
    h.send("(at-place euclid t7)")
    h.plan(euclid=("p0", 0, []))  # euclid is at t7: nothing to do
    assert "already holds" in h.texts[-1]
    stamp = h.node._stamp

    h.send("(visited-place-by euclid t9)")
    h.plan(euclid=("p1", 1, ["t9"]))
    h.done("euclid", "p1")  # euclid drove to t9
    assert h.node._stamp == stamp  # same map
    h.goals.clear()

    h.send("(at-place euclid t7)")
    assert len(h.goals) == 1, h.texts[-1:]
    assert h.goals[0].goal.pddl_goal == "(at-place euclid t7)"


def test_a_position_the_planner_reports_satisfied_is_not_remembered(h):
    h.send("(holding hamilton o1)")
    h.failed("goal already satisfied by the current state")
    h.goals.clear()
    h.send("(holding hamilton o1)")
    assert len(h.goals) == 1


def test_a_placement_the_planner_found_holding_is_still_remembered(h):
    h.send("(object-in-place o58 t2260)")
    h.plan(euclid=("pe", 0, []), hamilton=("ph", 0, []))
    h.goals.clear()
    h.send("(object-in-place o58 t2260)")
    assert h.goals == []
    assert "already done" in h.texts[-1]


# ---------- robot-scoped goals (2026-11-11 robot test) ----------


def _use(msg):
    return [
        s for c in msg.constraints if c.predicate == "use-robots" for s in c.symbols
    ]


def _euclid_to_o20(h):
    h.send("(visited-object-by euclid o20)")
    h.plan(euclid=("438a5115", 2, ["t3", "o20"]))
    h.goals.clear()
    h.sent.clear()


def test_a_goal_naming_one_robot_leaves_the_other_robots_plan_alone(h):
    """(a): (holding hamilton o1) stopped euclid on its way to o20."""
    _euclid_to_o20(h)
    h.send("(holding hamilton o1)")
    (sent,) = h.goals
    assert _use(sent) == ["hamilton"]
    assert h.busy(sent) == []
    assert h.sent == []  # euclid is not paused (hamilton was idle)
    assert any("replanning for hamilton only" in m for m in h.logs)

    # the planner answers for hamilton alone; a coverage message for euclid
    # from that planning (empty) must not wipe what euclid's plan covers
    h.plan(hamilton=("ph", 2, ["o1"]))
    h.node._robot_coverage_cb("euclid", String(data="[]"))
    e = h.node._robots["euclid"]
    assert (e.plan_id, e.status) == ("438a5115", "running")
    assert e.remaining == {"t3", "o20"}
    assert not any(
        "euclid preempted" in m or "euclid its new plan" in m for m in h.logs
    )
    assert h.sent == []  # nobody stopped
    # both robot goals stand, and compose into the mission
    assert h.node._standing == {
        "euclid": ["(visited-object-by euclid o20)"],
        "hamilton": ["(holding hamilton o1)"],
    }
    assert set(h.node._mission) == {
        "(visited-object-by euclid o20)",
        "(holding hamilton o1)",
    }


def test_a_later_goal_for_the_same_robot_replaces_only_its_own(h):
    _euclid_to_o20(h)
    h.send("(holding hamilton o1)")
    h.plan(hamilton=("ph", 2, ["o1"]))
    h.send("(visited-place-by hamilton t9)")
    h.plan(hamilton=("ph2", 1, ["t9"]))
    assert h.node._standing["euclid"] == ["(visited-object-by euclid o20)"]
    assert h.node._standing["hamilton"] == ["(visited-place-by hamilton t9)"]
    assert "(holding hamilton o1)" not in h.node._mission


def _euclid_carrying_o1_to_t4(h):
    """(b): euclid picked o1 and is placing it at t4 (a team goal)."""
    h.send("(object-in-place o1 t4)")
    h.plan(euclid=("d6ed6800", 2, ["t4"]))
    h.done("hamilton", "", action="PICK", obj="o20")  # hamilton holds o20
    h.node._held["euclid"] = {"o1"}
    h.goals.clear()
    h.sent.clear()


def test_a_hamilton_goal_does_not_make_euclid_put_down_its_cone(h):
    _euclid_carrying_o1_to_t4(h)
    assert h.node._held == {"hamilton": {"o20"}, "euclid": {"o1"}}
    h.send("(object-in-place o20 t4)", robot_id="hamilton", use=["hamilton"])
    (sent,) = h.goals
    # hamilton only: the planner (repair node) leaves euclid and its o1 alone
    assert _use(sent) == ["hamilton"]
    assert sent.goal.pddl_goal == "(object-in-place o20 t4)"
    assert h.sent == []  # euclid is not paused
    assert not any(t.startswith("the fleet goal") or "needs what" in t for t in h.texts)
    h.plan(hamilton=("de908928", 2, ["t4"]))
    assert h.node._robots["euclid"].plan_id == "d6ed6800"
    assert set(h.node._mission) == {
        "(object-in-place o1 t4)",
        "(object-in-place o20 t4)",
    }


def test_a_team_goal_still_replans_everyone(h):
    _euclid_carrying_o1_to_t4(h)
    h.send("(object-in-place o20 t4)", robot_id="hamilton", use=["hamilton"])
    h.plan(hamilton=("de908928", 2, ["t4"]))
    h.goals.clear()
    h.sent.clear()
    h.send("(and (object-in-place o1 t5) (object-in-place o20 t5))")
    (sent,) = h.goals
    assert _use(sent) == [] and h.busy(sent) == []  # a repair: anyone
    assert {("euclid", "pause"), ("hamilton", "pause")} <= set(h.sent)
    h.plan(euclid=("pe", 2, ["t5"]), hamilton=("ph", 2, ["t5"]))
    assert h.node._standing == {}  # the team goal replaces every robot goal
    assert h.node._mission == ["(object-in-place o1 t5)", "(object-in-place o20 t5)"]


def test_a_goal_needing_another_robots_object_replans_it_too_and_says_so(h):
    h.send("(and (object-placed-by euclid o1 t4) (visited-place-by euclid t2))")
    h.plan(euclid=("pe", 3, ["t2", "t4"]))
    h.node._held["euclid"] = {"o1"}
    h.goals.clear()
    h.send("(object-in-place o1 t5)", robot_id="hamilton", use=["hamilton"])
    (sent,) = h.goals
    assert _use(sent) == ["euclid", "hamilton"]
    # euclid keeps its own visit; its placement of o1 is replaced
    assert sent.goal.pddl_goal == (
        "(and (object-in-place o1 t5) (visited-place-by euclid t2))"
    )
    (text,) = [t for t in h.texts if "needs what" in t]
    assert "euclid's running plan acts on or holds o1" in text
    assert "dropping (object-placed-by euclid o1 t4)" in text
    assert ("euclid", "pause") in h.sent
    h.plan(euclid=("pe2", 1, ["t2"]), hamilton=("ph", 2, ["t5"]))
    assert "(object-placed-by euclid o1 t4)" not in h.node._mission


def test_visiting_the_same_object_is_no_conflict(h):
    _euclid_to_o20(h)
    h.send("(visited-object-by hamilton o20)")
    (sent,) = h.goals
    assert _use(sent) == ["hamilton"] and h.sent == []


def test_a_goal_naming_every_robot_is_the_teams(tmp_path):
    h = Harness(tmp_path, spots=("euclid", "hamilton"), huskies=())
    try:
        h.send("(visited-object-by euclid o20)")
        h.plan(euclid=("pe", 2, ["o20"]))
        h.goals.clear()
        h.send("(holding hamilton o1)", use=["euclid", "hamilton"])
        assert any(m.startswith("replanning (repair)") for m in h.logs)
    finally:
        h.close()


def test_a_scoped_goal_that_fails_stops_only_its_robots(h):
    _euclid_to_o20(h)
    h.send("(holding hamilton o1)")
    h.plan(hamilton=("ph", 2, ["o1"]))
    h.sent.clear()
    h.send("(visited-place-by hamilton t9)")
    assert h.sent == [("hamilton", "pause")]
    h.failed("no plan")
    assert h.sent[-1] == ("hamilton", "stop")
    assert ("euclid", "stop") not in h.sent
    assert h.node._robots["euclid"].status == "running"


def test_a_replan_request_drops_one_goals_use_robots(h):
    _euclid_to_o20(h)
    h.send("(holding hamilton o1)", use=["hamilton"])
    h.plan(hamilton=("ph", 2, ["o1"]))
    h.goals.clear()
    h.node._replan_request_cb(String(data="scene changed"))
    (sent,) = h.goals
    assert _use(sent) == []
    assert set(sent.goal.pddl_goal[5:-1].replace(") (", ")|(").split("|")) == {
        "(visited-object-by euclid o20)",
        "(holding hamilton o1)",
    }


# ---------- a pause must not outlive the plan it was for ----------


def _replanned_after_pause(h):
    """euclid runs a plan; a repair pauses it, and the new plan comes back."""
    h.send("(visited-object o1)")
    h.plan(euclid=("p1", 2, ["o1"]))
    h.sent.clear()
    h.send("(visited-object o1)", forbidden=["o7"])  # a repair: pauses euclid
    assert h.sent == [("euclid", "pause")]
    h.t += 1.4
    h.plan(euclid=("p2", 2, ["o1"]))
    h.sent.clear()


def test_a_robot_paused_before_its_new_plan_is_resumed_until_the_pause_is_stale(h):
    """(c): a pause that reaches the robot after its new plan (fleetlink
    retries it for up to 10 s) must not stop the new plan for good."""
    _replanned_after_pause(h)
    h.node._unpause_tick()
    assert h.sent == []  # not before the plan has had time to arrive
    h.t += 0.5
    h.node._unpause_tick()
    assert h.sent == [("euclid", "resume")]
    paused_at = h.node._last_pause["euclid"]
    times = []
    for _ in range(40):  # 20 s
        h.t += 0.5
        before = len(h.sent)
        h.node._unpause_tick()
        if len(h.sent) > before:
            times.append(h.t)
    assert all(b - a <= 2.0 for a, b in zip(times, times[1:]))  # every 2 s ...
    assert times[-1] >= paused_at + 12.0  # ... the last once the pause is stale
    assert times[-1] < paused_at + 13.0
    assert "euclid" not in h.node._unpause


def test_a_newer_pause_is_not_undone(h):
    _replanned_after_pause(h)
    h.t += 1.0
    # the planner pauses euclid for a scene change on the new plan
    h.node._pause_seen_cb("euclid", Bool(data=True))
    for _ in range(30):
        h.t += 0.5
        h.node._unpause_tick()
    assert ("euclid", "resume") not in h.sent


def test_our_own_pause_coming_back_is_not_a_new_one(h):
    h.send("(visited-object o1)")
    h.plan(euclid=("p1", 2, ["o1"]))
    h.send("(visited-object o1)", forbidden=["o7"])
    h.t += 0.2
    h.node._pause_seen_cb("euclid", Bool(data=True))  # its echo
    h.t += 1.0
    h.plan(euclid=("p2", 2, ["o1"]))
    assert "euclid" in h.node._unpause


def test_no_resume_without_a_recent_pause_or_after_the_plan_changed(h):
    h.send("(visited-object o1)")
    h.plan(euclid=("p1", 2, ["o1"]))
    assert h.node._unpause == {}
    h.send("(visited-object o1)", forbidden=["o7"])
    h.t += 30.0  # the pause is long stale by the time the plan comes
    h.plan(euclid=("p2", 2, ["o1"]))
    assert h.node._unpause == {}
    _replanned_after_pause(h)
    h.done("euclid", "p2")
    h.done("euclid", "p2")  # finished
    h.t += 1.0
    h.node._unpause_tick()
    assert ("euclid", "resume") not in h.sent
