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
    # not hamilton's use-robots alone: every robot of the mission's goals
    assert sorted(_use(sent)) == ["euclid", "hamilton"]
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


# ---------- mixed goals: a robot-free task is the team's ----------


def test_a_mixed_goal_is_the_teams(h):
    """(inspected-by hamilton o58) with a robot-free move: o12's move must
    not be forced onto hamilton."""
    h.send("(and (inspected-by hamilton o58) (object-in-place o12 t1253))")
    (sent,) = h.goals
    assert _use(sent) == []
    assert not any("robot-scoped" in m for m in h.logs)


def _euclid_visiting_cones(h):
    h.send("(and (visited-object o1) (visited-object o2) (visited-object o3))")
    h.plan(euclid=("cones1", 3, ["o1", "o2", "o3"]))
    h.goals.clear()
    h.sent.clear()


CONES_AND_HAMILTON = (
    "(and (visited-object o1) (visited-object o2) (visited-object o3) "
    "(visited-object-by hamilton o58))"
)


def test_resending_the_cones_with_a_hamilton_task_replans_the_team(h):
    """The cone visits must not all go to hamilton while euclid keeps its
    old plan for them too."""
    _euclid_visiting_cones(h)
    h.send(CONES_AND_HAMILTON)
    (sent,) = h.goals
    assert _use(sent) == [] and h.busy(sent) == []  # a repair: anyone
    assert ("euclid", "pause") in h.sent
    h.plan(euclid=("cones2", 2, ["o1", "o2"]), hamilton=("ph", 2, ["o3", "o58"]))
    assert h.node._robots["euclid"].plan_id == "cones2"
    assert h.node._standing == {}
    assert set(h.node._mission) == set(
        CONES_AND_HAMILTON[5:-1].replace(") (", ")|(").split("|")
    )


def test_euclids_stale_cone_plan_is_stopped_if_the_team_plan_leaves_it_out(h):
    _euclid_visiting_cones(h)
    h.send(CONES_AND_HAMILTON)
    h.plan(hamilton=("ph", 4, ["o1", "o2", "o3", "o58"]))
    assert ("euclid", "stop") in h.sent
    assert "euclid" not in h.node._running()


def test_a_mixed_goal_replaces_standing_robot_goals(h):
    h.send("(holding hamilton o1)")
    h.plan(hamilton=("ph", 2, ["o1"]))
    assert h.node._standing == {"hamilton": ["(holding hamilton o1)"]}
    h.send("(and (visited-object-by hamilton o2) (object-in-place o1 t5))")
    (_, sent) = h.goals
    assert _use(sent) == []
    assert ("hamilton", "pause") in h.sent
    h.plan(hamilton=("ph2", 1, ["o2"]), euclid=("pe", 2, ["t5"]))
    assert h.node._standing == {}
    assert "(holding hamilton o1)" not in h.node._mission


def test_explicit_use_robots_still_scopes_a_mixed_goal(h):
    _euclid_visiting_cones(h)
    h.send(
        "(and (visited-object-by hamilton o58) (visited-object o9))", use=["hamilton"]
    )
    (sent,) = h.goals
    assert _use(sent) == ["hamilton"]
    assert h.sent == []  # euclid is left alone


def test_a_goal_whose_every_open_task_names_a_robot_stays_scoped(h):
    """The robot-free part is already achieved: what is left is hilbert's,
    and euclid and hamilton are left alone."""
    h.send("(and (visited-object o1) (visited-object o2))")
    h.plan(euclid=("pe", 1, ["o1"]), hamilton=("ph", 2, ["o2"]))
    h.done("euclid", "pe")  # o1 visited; hamilton still busy
    h.goals.clear()
    h.sent.clear()
    h.send("(and (visited-object o1) (visited-object-by hilbert o9))")
    (sent,) = h.goals
    assert _use(sent) == ["hilbert"]
    assert sent.goal.pddl_goal == "(visited-object-by hilbert o9)"
    assert h.sent == []


# ---------- the fleet is every robot in the plugin config ----------

EIGHT = ("euclid", "hamilton", "hilbert", "gauss2", "noether", "fermat", "lagrange")


def test_a_mixed_goal_is_the_teams_with_eight_robots_configured(tmp_path):
    """The fleet is every robot in the plugin config, online or not: a mixed
    goal must not be scoped to its one named robot because six configured
    robots are offline."""
    h = Harness(tmp_path, spots=EIGHT, huskies=("gauss",))
    try:
        h.send("(and (visited-object-by euclid o1) (visited-object o3))")
        (sent,) = h.goals
        assert _use(sent) == []
    finally:
        h.close()


def test_replanning_another_robot_keeps_only_its_own_part_of_a_shared_goal(h):
    """2026-11-12 Killian test: gauss and hamilton shared a goal, hamilton
    finished its visit; a euclid-only goal for gauss's cone replanned gauss and
    carried hamilton's visit along, with hamilton left out of the problem:
    "goal references symbols absent from the scene graph: ['hamilton']"."""
    h.send(
        "(and (visited-object-by gauss o58) (visited-place-by hamilton t1472))",
        robot_id="hamilton",
    )
    h.plan(hamilton=("ph", 1, ["t1472", "t1473"]), gauss=("pg", 0, ["o58", "t2181"]))
    h.done("hamilton", "ph")
    h.goals.clear()
    h.sent.clear()

    h.send("(holding euclid o58)", robot_id="euclid", use=["euclid"])
    (sent,) = h.goals
    assert _use(sent) == ["euclid", "gauss"]
    assert (
        sent.goal.pddl_goal
        == "(and (holding euclid o58) (visited-object-by gauss o58))"
    )
    assert "hamilton" not in sent.goal.pddl_goal


def test_a_finished_visit_by_a_named_robot_is_done(h):
    h.send("(visited-place-by hamilton t1472)")
    h.plan(hamilton=("ph", 1, ["t1472"]))
    h.done("hamilton", "ph")
    assert h.node._is_done("(visited-place-by hamilton t1472)")
    # another robot's visit does not count for this one
    assert not h.node._is_done("(visited-place-by euclid t1472)")


# ---------- willow, 2026-11-12: a kept task named a robot outside use-robots ----------

SHARED = "(and (visited-object-by gauss o58) (visited-place-by hamilton t1472))"


def _gauss_and_hamilton(h):
    h.send(SHARED, robot_id="hamilton")
    h.plan(hamilton=("192a7f09", 1, ["t1472"]), gauss=("", 0, ["o58"]))
    h.goals.clear()
    h.texts.clear()


def _sent(h):
    (sent,) = h.goals
    from omniplanner_ros.goal_manager_node import goal_conjuncts

    return set(goal_conjuncts(sent.goal.pddl_goal)), sorted(h.use(sent))


def test_a_conflict_keeps_only_the_tasks_of_the_robots_it_plans_for(h):
    """(holding euclid o58) needs o58, which gauss's plan acts on, so gauss is
    replanned too with the rest of its goal. That goal was shared with
    hamilton: hamilton's task went along, hamilton was not in use-robots, and
    the planner refused the goal ("symbols absent ...: ['hamilton']")."""
    _gauss_and_hamilton(h)  # hamilton still on its way
    h.send("(holding euclid o58)", robot_id="euclid")
    goal, use = _sent(h)
    assert use == ["euclid", "gauss"]
    assert goal == {"(holding euclid o58)", "(visited-object-by gauss o58)"}
    assert ("hamilton", "pause") not in h.sent
    # hamilton's own task is still its own, and still in the mission
    h.plan(euclid=("pe", 2, ["o58"]), gauss=("", 0, ["o58"]))
    assert "(visited-place-by hamilton t1472)" in h.node._mission
    assert h.node._robots["hamilton"].plan_id == "192a7f09"


def test_the_willow_sequence_with_hamilton_finished_plans(h):
    _gauss_and_hamilton(h)
    h.done("hamilton", "192a7f09")  # hamilton reached t1472
    h.send("(holding euclid o58)", robot_id="euclid")
    goal, use = _sent(h)
    assert use == ["euclid", "gauss"]
    assert goal == {"(holding euclid o58)", "(visited-object-by gauss o58)"}
    assert "keeping (visited-object-by gauss o58)." in h.texts[-1]


# ---------- a replan keeps the mission's robot scope (2026-11-17 bench test) ----------


def _hamilton_and_euclid_to_objects(h):
    """The console's team goal from the bench test: two robots named, gauss idle."""
    h.send("(and (visited-object-by hamilton o536) (visited-object-by euclid o710))")
    (sent,) = h.goals
    assert sorted(_use(sent)) == ["euclid", "hamilton"]
    h.plan(hamilton=("ph", 1, ["t1", "o536"]), euclid=("pe", 1, ["t3349", "o710"]))
    h.goals.clear()
    h.sent.clear()


def test_a_replan_of_a_robot_scoped_goal_stays_scoped_to_its_robots(h):
    """Removing t3349 replanned with no use-robots: gauss, not in the goal, was
    sent an empty Path (stopped) and every robot was paused."""
    _hamilton_and_euclid_to_objects(h)
    h.node._replan_request_cb(String(data="place t3349 on euclid's route was removed"))
    (sent,) = h.goals
    assert sorted(_use(sent)) == ["euclid", "hamilton"]
    assert not any(r == "gauss" for r, _ in h.sent)


def test_robot_goals_still_compose_after_a_replan(h):
    """After the replan's plan arrives, each robot keeps its own goal: a later
    goal for gauss leaves hamilton and euclid alone."""
    _hamilton_and_euclid_to_objects(h)
    h.node._replan_request_cb(String(data="scene changed"))
    h.plan(hamilton=("ph2", 1, ["t1", "o536"]), euclid=("pe2", 1, ["t431", "o710"]))
    assert set(h.node._standing) == {"euclid", "hamilton"}
    h.goals.clear()
    h.sent.clear()
    h.send("(visited-object-by gauss o30)")
    (sent,) = h.goals
    assert _use(sent) == ["gauss"]
    assert not any(r in ("euclid", "hamilton") for r, _ in h.sent)


def test_a_replan_leaves_alone_a_robot_whose_own_goal_is_achieved(h):
    """hamilton reached o536 and is done; a change on euclid's route replans
    euclid only: hamilton is neither paused nor planned for."""
    _hamilton_and_euclid_to_objects(h)
    h.done("hamilton", "ph")
    h.node._replan_request_cb(String(data="place t3349 on euclid's route was removed"))
    (sent,) = h.goals
    assert _use(sent) == ["euclid"]
    assert sent.goal.pddl_goal == "(visited-object-by euclid o710)"
    assert [r for r, _ in h.sent] == ["euclid"]


def test_separate_robot_goals_are_replanned_together_and_stay_separate(h):
    """Three goals sent one by one, each for its robot: a replan names all
    three and no one else, and each robot keeps its own goal afterwards."""
    _euclid_to_o20(h)
    h.send("(holding hamilton o1)", use=["hamilton"])
    h.plan(hamilton=("ph", 2, ["o1"]))
    h.send("(visited-place-by gauss t9)")
    h.plan(gauss=("pg", 1, ["t9"]))
    h.goals.clear()
    h.sent.clear()
    h.node._replan_request_cb(String(data="scene changed"))
    (sent,) = h.goals
    assert sorted(_use(sent)) == ["euclid", "gauss", "hamilton"]
    assert sorted(h.sent) == [
        ("euclid", "pause"),
        ("hamilton", "pause"),
    ]  # gauss: a husky
    h.plan(
        euclid=("pe2", 2, ["t3", "o20"]),
        hamilton=("ph2", 2, ["o1"]),
        gauss=("pg2", 1, ["t9"]),
    )
    assert h.node._standing == {
        "euclid": ["(visited-object-by euclid o20)"],
        "hamilton": ["(holding hamilton o1)"],
        "gauss": ["(visited-place-by gauss t9)"],
    }


def test_a_team_task_added_to_robot_goals_makes_the_replan_the_teams(h):
    """euclid has its own goal; a task naming no robot was planned for the free
    robots as an additional task. Its robot is not one of the scoped ones, so
    scoping the replan to euclid would leave that robot on its old plan and
    hand its task to euclid as well."""
    _euclid_to_o20(h)
    h.send("(visited-object o77)")
    (extra,) = h.goals
    assert h.busy(extra) == [["euclid"]]
    h.plan(hamilton=("ph", 1, ["o77"]))
    assert "(visited-object o77)" in h.node._mission
    h.goals.clear()
    h.sent.clear()
    h.node._replan_request_cb(String(data="scene changed"))
    (sent,) = h.goals
    assert _use(sent) == []
    assert sorted(h.sent) == [("euclid", "pause"), ("hamilton", "pause")]
    assert "(visited-object o77)" in sent.goal.pddl_goal
    assert "(visited-object-by euclid o20)" in sent.goal.pddl_goal


def test_a_team_mission_is_still_replanned_for_the_whole_team(h):
    h.send("(and (visited-object o20) (visited-object o21))")
    h.plan(euclid=("pe", 1, ["o20"]), hamilton=("ph", 1, ["o21"]))
    h.goals.clear()
    h.sent.clear()
    h.node._replan_request_cb(String(data="scene changed"))
    (sent,) = h.goals
    assert _use(sent) == []
    assert sorted(h.sent) == [("euclid", "pause"), ("hamilton", "pause")]


def test_a_robot_whose_goal_was_dropped_with_a_removed_object_is_stopped(h):
    """o536 was removed: hamilton's visit is dropped and the replan is euclid's
    alone. hamilton is in no plan to come, and must not walk on to o536."""
    _hamilton_and_euclid_to_objects(h)
    h.node._replan_request_cb(
        String(data="scene changed: o536 removed before the visit the goal asks for")
    )
    (sent,) = h.goals
    assert _use(sent) == ["euclid"]
    assert sent.goal.pddl_goal == "(visited-object-by euclid o710)"
    assert ("hamilton", "stop") in h.sent and ("hamilton", "pause") not in h.sent
    assert ("euclid", "pause") in h.sent
    assert "hamilton" not in h.node._standing


def test_a_scoped_replan_that_fails_leaves_the_outside_robot_running(h):
    """A replan the planner finds no plan for stops the robots it was for;
    gauss was never part of it."""
    _hamilton_and_euclid_to_objects(h)
    h.node._replan_request_cb(String(data="scene changed"))
    h.failed("no plan")
    assert not any(r == "gauss" for r, _ in h.sent)


def test_a_second_replan_after_the_first_keeps_the_same_scope(h):
    _hamilton_and_euclid_to_objects(h)
    for plan in ("2", "3"):
        h.node._replan_request_cb(String(data="scene changed"))
        assert sorted(_use(h.goals[-1])) == ["euclid", "hamilton"]
        h.plan(
            hamilton=("ph" + plan, 1, ["t1", "o536"]),
            euclid=("pe" + plan, 1, ["t431", "o710"]),
        )
    assert not any(r == "gauss" for r, _ in h.sent)
    assert set(h.node._standing) == {"euclid", "hamilton"}
