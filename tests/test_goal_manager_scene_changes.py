"""Scene changes under a multi-robot mission (2026-11-11 rig test): one
removed object drops only its own task, and a plan that was stopped or has
finished is never replanned by a later change."""

import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from omniplanner.plan_validity import (  # noqa: E402
    ADDED,
    MOVED,
    REMOVED,
    affecting_changes,
    removed_symbols,
)

HOLD_O0, HOLD_O2 = "(holding euclid o0)", "(holding hamilton o2)"
O0_GONE = (
    "scene changed: o21 removed before the visit the goal asks for; "
    "o0 removed, and the plan still has to pick it"
)


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _replan(h, reason):
    h.node._replan_request_cb(String(data=reason))


def _two_holds(h):
    h.send(f"(and {HOLD_O0} {HOLD_O2})", robot_id="euclid")
    h.plan(euclid=("pe", 2, ["o0"]), hamilton=("ph", 2, ["o2"]))
    h.goals.clear()
    h.sent.clear()
    h.texts.clear()


# ---------- the reason text names what was removed ----------


def test_removed_symbols_reads_the_reasons_affecting_changes_writes():
    changes = [
        {"kind": REMOVED, "symbol": "O0", "old": (0, 0), "new": None},
        {"kind": REMOVED, "symbol": "o21", "old": (0, 0), "new": None},
        {"kind": REMOVED, "symbol": "o30", "old": (0, 0), "new": None},
        {"kind": REMOVED, "symbol": "o31", "old": (0, 0), "new": None},
        {"kind": MOVED, "symbol": "o5", "old": (0, 0), "new": (3, 0)},
        {"kind": ADDED, "symbol": "o6", "old": None, "new": (1, 0)},
    ]
    goal = (
        "(and (visited-object o21) (safe o30) (object-in-place o31 t1) "
        "(visited-object o5) (visited-object o6))"
    )
    affected = affecting_changes(changes, {"o0": ("pick", (0.0, 0.0))}, goal)
    assert len(affected) == 6
    for prefix in ("scene changed: ", "map reloaded: "):
        reason = prefix + "; ".join(r for _, r in affected)
        assert removed_symbols(reason) == {"o0", "o21", "o30", "o31"}
    assert removed_symbols("map changed: place t5 on euclid's route was removed") == (
        set()
    )
    assert removed_symbols("pick of o3 by euclid failed (attempt 1 of 2)") == set()
    assert removed_symbols("") == set()


# ---------- item 3: one removed object drops only its own task ----------


def test_one_removed_object_drops_its_task_and_keeps_the_other_robots(h):
    _two_holds(h)
    _replan(h, O0_GONE)
    (sent,) = h.goals
    assert sent.goal.pddl_goal == HOLD_O2
    (text,) = h.texts
    assert text == (f"o0 was removed: dropped {HOLD_O0}; continuing with {HOLD_O2}.")
    assert h.node._mission == [HOLD_O2]
    assert all(cs == [HOLD_O2] for cs in h.node._standing.values())

    h.plan(hamilton=("ph2", 2, ["o2"]))
    assert ("euclid", "stop") in h.sent  # nothing is left for it
    assert ("hamilton", "stop") not in h.sent
    assert h.node._robots["hamilton"].plan_id == "ph2"
    assert "euclid" not in h.node._robots
    assert not any("Cannot repair" in t for t in h.texts)


def test_a_goal_with_nothing_left_still_fails_and_says_so(h):
    h.send("(holding hamilton o20)")
    h.plan(hamilton=("ph", 2, ["o20"]))
    h.goals.clear()
    _replan(h, "scene changed: o20 removed, and the plan still has to pick it")
    assert [g.goal.pddl_goal for g in h.goals] == ["(holding hamilton o20)"]
    h.failed("Goal symbol(s) not in the DSG: o20")
    text = h.texts[-1]
    assert "Cannot repair the plan" in text and "o20 removed" in text
    assert "Stopped hamilton" in text and "dropped" not in text


def test_every_task_removed_fails_as_before(h):
    _two_holds(h)
    _replan(
        h,
        "scene changed: o0 removed, and the plan still has to pick it; "
        "o2 removed, and the plan still has to pick it",
    )
    assert [g.goal.pddl_goal for g in h.goals] == [f"(and {HOLD_O0} {HOLD_O2})"]
    h.failed("The goal could not be read: it names o0, o2")
    assert "Cannot repair the plan" in h.texts[-1]
    assert "Stopped euclid and hamilton" in h.texts[-1]


def test_a_moved_object_drops_nothing(h):
    _two_holds(h)
    _replan(h, "scene changed: o0 moved 2.0 m, and the plan still has to pick it")
    assert [g.goal.pddl_goal for g in h.goals] == [f"(and {HOLD_O0} {HOLD_O2})"]
    assert h.texts == []


def test_a_choice_naming_the_removed_object_is_not_dropped(h):
    """(or ...) may still hold another way: only plain tasks are dropped."""
    goal = "(and (or (visited-object o0) (visited-object o1)) (visited-object o2))"
    h.send(goal)
    h.plan(euclid=("pe", 2, ["o0"]), hamilton=("ph", 2, ["o2"]))
    h.goals.clear()
    h.texts.clear()
    _replan(h, "scene changed: o0 removed before the visit the goal asks for")
    (sent,) = h.goals
    assert "(or (visited-object o0) (visited-object o1))" in sent.goal.pddl_goal
    assert h.texts == []


# ---------- item 4: a stopped or finished plan is not replanned ----------


def test_a_stopped_plan_is_not_replanned_by_a_later_scene_change(h):
    """o20 removed: hamilton stopped, 'give a new instruction'. o20 restored:
    the planner asked for a replan, and hamilton went for it again."""
    h.send("(visited-object-by euclid o0)")
    h.plan(euclid=("pe", 1, ["o0"]))
    h.done("euclid", "pe")  # euclid's goal is finished
    h.send("(holding hamilton o20)")
    h.plan(hamilton=("ph", 2, ["o20"]))
    _replan(h, "scene changed: o20 removed, and the plan still has to pick it")
    # only hamilton's goal: euclid's finished one is not planned again
    assert h.goals[-1].goal.pddl_goal == "(holding hamilton o20)"
    h.failed("Goal symbol(s) not in the DSG: o20")
    assert "Stopped hamilton" in h.texts[-1]
    assert h.node._mission == [] and h.node._standing == {}
    h.goals.clear()
    h.sent.clear()

    _replan(h, "scene changed: o20 added, and the plan still has to pick it")
    assert h.goals == [] and h.sent == []  # nothing replanned, nobody paused
    assert any("no mission is running" in m for m in h.logs)


def test_a_stopped_team_mission_is_not_replanned_either(h):
    h.send("(visited-object o20)", robot_id="hilbert")
    h.plan(hilbert=("p1", 3, ["p4", "o20"]))
    _replan(h, "scene changed: o20 removed before the visit the goal asks for")
    h.failed("Goal symbol(s) not in the DSG: o20")
    assert "Stopped hilbert" in h.texts[-1]
    h.goals.clear()
    _replan(h, "scene changed: o20 added before the visit the goal asks for")
    assert h.goals == []


def test_a_replan_is_for_the_robots_still_at_work_not_a_finished_goal(h):
    """Two plans; the change invalidates hamilton's: its goal is replanned,
    not euclid's finished one."""
    h.send("(visited-object-by euclid o0)")
    h.plan(euclid=("pe", 1, ["o0"]))
    h.send("(holding hamilton o17)")
    h.plan(hamilton=("ph", 2, ["o17"]))
    h.done("euclid", "pe")
    assert h.node._mission == ["(holding hamilton o17)"]
    h.goals.clear()
    _replan(h, "scene changed: o17 moved 2.0 m, and the plan still has to pick it")
    assert [g.goal.pddl_goal for g in h.goals] == ["(holding hamilton o17)"]


def test_a_shared_goal_stays_until_its_last_robot_finishes(h):
    h.send("(and (visited-object o4) (visited-object o8))", use=["euclid", "hamilton"])
    h.plan(euclid=("p", 1, ["o4"]), hamilton=("p", 1, ["o8"]))
    h.node._achieved.clear()  # as if the visits were not judged done
    h.done("euclid", "p")
    h.node._achieved.clear()
    assert set(h.node._mission) == {"(visited-object o4)", "(visited-object o8)"}
    h.done("hamilton", "p")
    assert h.node._mission == []


def test_a_failed_goal_for_one_robot_ends_only_its_goal(h):
    """hamilton's new goal cannot be planned: hamilton stops, and its old
    goal is not brought back by a change under euclid's plan."""
    h.send("(visited-object-by euclid o0)")
    h.plan(euclid=("pe", 2, ["o0"]))
    h.send("(visited-object-by hamilton o2)")
    h.plan(hamilton=("ph", 2, ["o2"]))
    h.send("(holding hamilton o99)")
    h.failed("Goal symbol(s) not in the DSG: o99")
    assert "Stopped hamilton" in h.texts[-1]
    assert h.node._mission == ["(visited-object-by euclid o0)"]
    h.goals.clear()
    _replan(h, "scene changed: o0 moved 2.0 m before the visit the goal asks for")
    assert [g.goal.pddl_goal for g in h.goals] == ["(visited-object-by euclid o0)"]


# ---------- a robot's own tasks follow that robot (rig, 2026-11-12) ----------


def test_a_new_goal_for_a_robot_replaces_its_task_after_a_repair(h):
    """Rig: (holding hamilton o2) was replanned once (a repair forgets who
    stood for what), then hamilton was told (holding hamilton o20). With o20
    removed the goal manager 'continued with (holding hamilton o2)', a goal
    the operator had replaced."""
    _two_holds(h)
    _replan(h, O0_GONE)
    h.plan(hamilton=("ph2", 2, ["o2"]))
    assert h.node._mission == [HOLD_O2]

    h.send("(holding hamilton o20)")
    h.plan(hamilton=("ph3", 2, ["o20"]))
    assert h.node._mission == ["(holding hamilton o20)"]

    h.goals.clear()
    h.texts.clear()
    _replan(h, "scene changed: o20 removed, and the plan still has to pick it")
    assert [g.goal.pddl_goal for g in h.goals] == ["(holding hamilton o20)"]
    assert not any("continuing with" in t for t in h.texts)
    h.failed("The goal could not be read: it names o20")
    assert "Cannot repair the plan" in h.texts[-1] and "Stopped hamilton" in h.texts[-1]
    assert h.node._mission == []


def test_a_new_goal_for_one_robot_of_a_shared_goal_replaces_only_its_task(h):
    _two_holds(h)
    h.send("(holding hamilton o20)")
    h.plan(hamilton=("ph3", 2, ["o20"]))
    assert set(h.node._mission) == {HOLD_O0, "(holding hamilton o20)"}
    assert HOLD_O2 not in [c for cs in h.node._standing.values() for c in cs]


def test_a_finished_robots_own_task_leaves_a_shared_goal(h):
    """euclid finished its -by visit of a goal it shares with hamilton: a
    later replan is for hamilton's task only."""
    h.send("(and (visited-object-by euclid o0) (visited-object-by hamilton o2))")
    h.plan(euclid=("pe", 1, ["o0"]), hamilton=("ph", 3, ["o2"]))
    h.done("euclid", "pe")
    assert h.node._mission == ["(visited-object-by hamilton o2)"]
    h.goals.clear()
    _replan(h, "scene changed: o2 moved 2.0 m before the visit the goal asks for")
    assert [g.goal.pddl_goal for g in h.goals] == ["(visited-object-by hamilton o2)"]


def test_a_robot_free_task_is_not_taken_from_the_robot_still_doing_it(h):
    """Shared robot-free tasks stay until the last robot standing for them
    finishes: who does which is the planner's choice."""
    h.send("(and (visited-object o4) (visited-object o8))", use=["euclid", "hamilton"])
    h.plan(euclid=("p", 1, []), hamilton=("p", 2, ["o4", "o8"]))
    h.done("euclid", "p")  # euclid's part was nothing
    assert set(h.node._mission) == {"(visited-object o4)", "(visited-object o8)"}
