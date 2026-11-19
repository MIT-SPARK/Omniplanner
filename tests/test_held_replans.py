"""Held replans (confirm_replans, protocol v4): a change detection scene
change (source awcd/) that invalidates the running goal pauses no robot and
replans nothing until the operator confirms; a map edit (sget/, console/), a
cut route, a map reload and a failed pick replan at once, as without it.

The repair node is the faked one of test_willow_repair_node.
"""

import inspect
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import test_willow_repair_node as W
from test_willow_repair_node import (
    CARRY_GOAL,
    PLACES,
    _about_to_pick,
    _carry,
    _done,
    _map,
    _node,
    _path,
    _replans,
    _visit,
)

from omniplanner.plan_validity import ADDED, MOVED, REMOVED, removed_symbols

repair = pytest.importorskip("omniplanner_ros.omniplanner_repair_node")

GOAL2 = "(and (object-in-place o58 t4) (visited-object testbot_b o60))"
SCENE = "scene changed: "
QUESTION_KEYS = {"id", "robots", "reasons", "reason", "source", "stopped", "t"}


@pytest.fixture(autouse=True)
def _no_ros_map_parsing(monkeypatch):
    monkeypatch.setattr(repair.OmniPlannerRos, "dsg_callback", lambda self, msg: None)


def _confirming(node, on=True):
    node._confirm_replans = on
    node._pending_pub = node.create_publisher(None, "~/replan_pending", 1)
    return node


def _questions(node):
    pub = node.f_pubs.get("~/replan_pending")
    return [m.data for m in pub.sent] if pub else []


def _pending(node):
    sent = _questions(node)
    return json.loads(sent[-1]) if sent and sent[-1] else None


def _decide(node, decision, qid=None):
    if qid is None:
        qid = _pending(node)["id"]
    node._replan_decision_callback(
        NS(data=json.dumps({"id": qid, "decision": decision}))
    )


def _xy(p):
    return NS(x=0.0, y=0.0) if p is None else NS(x=p[0], y=p[1])


def _msg(changes, source, version):
    return NS(
        source=source,
        map_version=version,
        changes=[
            NS(kind=kind, symbol=sym, old_position=_xy(old), new_position=_xy(new))
            for kind, sym, old, new in changes
        ],
    )


def _scene(node, changes, source="awcd/hamilton", version=1):
    """A scene change message, judged once it settled. The map already has
    version 1; a later version is waited for (see _map_arrives)."""
    node._scene_changes_callback(_msg(changes, source, version))
    node._judge_changes()


def _rm(sym, xy=(2.0, 0.2)):
    return (REMOVED, sym, xy, None)


def _map_arrives(node, version):
    node.dsg_last = _map("e1", version)
    while True:  # the 0.2 s map waits, run to their end
        waits = [
            t for t in node.f_timers if t.period == 0.2 and not t.__dict__.get("f")
        ]
        if not waits:
            return
        waits[-1].f = True
        waits[-1].cb()


def _stops(node):
    """Every pause and empty Path sent to a robot."""
    pauses = [
        m
        for topic, pub in node.f_pubs.items()
        if topic.endswith("/spot_executor_node/pause")
        for m in pub.sent
    ]
    paths = [
        m
        for a in node.robot_adaptors.values()
        for m in a.plan_pub.sent
        if not getattr(m, "poses", True)
    ]
    return pauses + paths


def _carrying(node):
    node._planning_goal = CARRY_GOAL
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)


def _two(node):
    node._planning_goal = GOAL2
    node.finalize_plans(
        {"testbot_a": _carry(), "testbot_b": _visit("testbot_b", "v")}, node.f_poses
    )


def _team(node):
    """testbot_a carries o58, the Husky drives a Path: both act on o58."""
    node._planning_goal = CARRY_GOAL
    node.finalize_plans(
        {"testbot_a": _carry(), "testhusky": _path((1, 0), (2, 0))}, node.f_poses
    )


# ---------- a change detection report: asked, nobody paused ----------


def test_a_change_detection_change_is_asked_and_pauses_nobody():
    node = _confirming(_node(_map(), poses={"testhusky": np.zeros(4)}))
    _team(node)
    _scene(node, [_rm("O58")])
    assert _replans(node) == [] and _stops(node) == []
    assert sorted(node._running_plans) == ["testbot_a", "testhusky"]
    q = _pending(node)
    assert set(q) == QUESTION_KEYS
    # robots: those whose plans, or their goals, name o58
    assert q["robots"] == ["testbot_a", "testhusky"] and q["stopped"] == []
    assert q["source"] == "awcd/hamilton" and len(q["id"]) == 32
    assert q["reasons"] == [q["reason"]] and "o58 removed" in q["reason"]
    assert isinstance(q["t"], float)
    _decide(node, "confirm")
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58"} and _questions(node)[-1] == ""
    assert _stops(node) == []  # the replan's new plans replace the old ones


def test_a_declined_change_sends_nothing_and_cannot_be_confirmed_after():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    qid = _pending(node)["id"]
    _decide(node, "decline")
    assert _replans(node) == [] and _stops(node) == [] and _pending(node) is None
    _decide(node, "confirm", qid)
    assert _replans(node) == []
    # still watched: a later change asks again, with a new id
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (5.0, 0.2))])
    assert _pending(node)["id"] != qid


def test_a_confirm_waits_for_the_map_that_has_the_change():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")], version=5)
    assert _pending(node) is not None  # asked at once
    _decide(node, "confirm")
    assert _replans(node) == []
    _map_arrives(node, 5)
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58"}


# ---------- which replans are held ----------


@pytest.mark.parametrize("source", ["sget/chat", "console/map_edit", "other"])
def test_a_map_edit_pauses_and_replans_at_once(source):
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")], source=source)
    (reason,) = _replans(node)
    assert "o58 removed" in reason and _questions(node) == []
    assert [m.data for m in _stops(node)] == [True]


def test_without_confirm_replans_a_detection_replans_at_once():
    node = _confirming(_node(_map()), on=False)
    _carrying(node)
    _scene(node, [_rm("O58")])
    assert len(_replans(node)) == 1 and _questions(node) == []


def test_nothing_listening_on_replan_pending_replans_at_once():
    node = _confirming(_node(_map()))
    node._pending_pub.subscribers = 0
    _carrying(node)
    _scene(node, [_rm("O58")])
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58"} and _questions(node) == []
    assert len(_stops(node)) == 1  # as on main


def test_an_executor_placement_is_still_ignored():
    node = _confirming(_node(_map()))
    _carrying(node)
    node._scene_changes_callback(NS(source="executor/x", map_version=1, changes=[]))
    assert node._pending_changes == [] and _questions(node) == []


def test_a_reload_replans_at_once():
    node = _confirming(_node(_map("e1", 4)))
    _carrying(node)
    node._note_map_state()
    node.dsg_last = _map("e2", 1, o58=(3.5, 0.2))  # o58 moved 1.5 m
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    (reason,) = _replans(node)
    assert reason.startswith("map reloaded: ") and _questions(node) == []


def test_a_cut_route_replans_at_once():
    node = _confirming(_node(_map("e1", 4)))
    _carrying(node)
    node._note_map_state()
    node.dsg_last = _map("e1", 5, places={k: v for k, v in PLACES.items() if k != "t3"})
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    (reason,) = _replans(node)
    assert "t3" in reason and _questions(node) == []


def test_moves_within_move_threshold_are_not_changes():
    node = _confirming(_node(_map()))
    _carrying(node)
    for dx in (0.05, 0.08, 0.3):
        _scene(node, [(MOVED, "O58", (2.0, 0.2), (2.0 + dx, 0.2))])
    assert _questions(node) == []
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (2.8, 0.2))])
    assert "o58 moved 0.8 m" in _pending(node)["reason"]


def test_an_added_object_asks_nothing():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [(ADDED, "O99", None, (1.0, 1.0))])
    assert _questions(node) == [] and _replans(node) == []


def test_no_running_plan_no_question_and_judged_as_on_main():
    node = _confirming(_node(_map()))
    node._active_goal = CARRY_GOAL  # a goal, but no plan running
    _scene(node, [_rm("O58")])
    assert _questions(node) == [] and len(_replans(node)) == 1


# ---------- the question and the decision ----------


def _confirmed_two_removals():
    """Two removed goal objects, asked in two questions, confirmed: the one
    replan request it sent."""
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    first = _pending(node)["id"]
    _scene(node, [_rm("O60", (1.0, 0.2))])
    q = _pending(node)
    assert q["id"] != first and len(q["reasons"]) == 2
    assert q["reason"].count("scene changed: ") == 1
    assert removed_symbols(q["reason"]) == {"o58", "o60"}
    _decide(node, "confirm", first)  # the replaced question
    assert _replans(node) == []
    _decide(node, "confirm")
    (reason,) = _replans(node)
    assert _pending(node) is None
    return reason


def test_a_confirm_sends_one_replan_request_naming_every_removed_object():
    reason = _confirmed_two_removals()
    assert reason.count("scene changed: ") == 1
    assert removed_symbols(reason) == {"o58", "o60"}


def test_the_goal_manager_drops_every_removed_object_of_a_confirmed_hold(tmp_path):
    pytest.importorskip("robot_executor_msgs.msg")
    from goal_manager_harness import Harness
    from std_msgs.msg import String

    reason = _confirmed_two_removals()
    h = Harness(tmp_path)
    try:
        h.send("(and (visited-object o58) (visited-object o60) (visited-place t1))")
        h.plan(euclid=("pe", 3, []))
        h.goals.clear()
        h.node._replan_request_cb(String(data=reason))
        (out,) = h.goals
        goal = out.goal.pddl_goal
        assert "o58" not in goal and "o60" not in goal and "t1" in goal
    finally:
        h.close()


def test_decisions_without_the_questions_id_are_ignored():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    for data in ("confirm", "decline", "{}", '{"decision": "confirm"}', "not json"):
        node._replan_decision_callback(NS(data=data))
    _decide(node, "confirm", qid="0" * 32)
    _decide(node, "maybe")
    assert _replans(node) == [] and _pending(node) is not None


def test_an_edit_waiting_for_its_map_takes_the_held_change_with_it():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    _scene(node, [_rm("O60", (1.0, 0.2))], source="console/map_edit", version=5)
    assert _questions(node)[-1] == "" and _replans(node) == []
    _map_arrives(node, 5)
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58", "o60"}


# ---------- any other replan carries the held changes ----------


def test_an_edit_during_a_hold_carries_the_held_change():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    _scene(node, [_rm("O60", (1.0, 0.2))], source="sget/chat")
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58", "o60"}
    assert reason.count("scene changed: ") == 1 and _questions(node)[-1] == ""


def test_a_burst_of_an_edit_and_a_detection_replans_both_at_once():
    node = _confirming(_node(_map()))
    _two(node)
    node._scene_changes_callback(_msg([_rm("O60", (1.0, 0.2))], "console/map_edit", 1))
    _scene(node, [_rm("O58")])  # within the same settle
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58", "o60"} and _pending(node) is None


def test_a_cut_route_during_a_hold_carries_the_held_change():
    node = _confirming(_node(_map("e1", 4)))
    _carrying(node)
    node._note_map_state()
    _scene(node, [_rm("O58")], version=4)
    node.dsg_last = _map("e1", 5, places={k: v for k, v in PLACES.items() if k != "t3"})
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    (reason,) = _replans(node)
    assert reason.startswith("map changed: ") and removed_symbols(reason) == {"o58"}
    assert _questions(node)[-1] == ""


def test_a_failed_pick_during_a_hold_carries_the_held_change():
    node = _confirming(_node(_map(), poses={"testbot_b": np.zeros(4)}))
    _two(node)
    _scene(node, [_rm("O58")])
    _done(node, "testbot_b", "PICK", plan_id="v", obj="o60", ok=False)
    (reason,) = _replans(node)
    assert reason.startswith("pick of o60 by testbot_b failed (attempt 1 of 2)")
    assert removed_symbols(reason) == {"o58"} and _pending(node) is None


# ---------- the robots asked about ----------


def test_a_new_goal_without_the_changed_object_answers_the_held_change():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    node._planning_goal = "(visited-place t3)"  # the console's new goal
    node.finalize_plans({"testbot_a": _visit("testbot_a", "n")}, node.f_poses)
    assert _questions(node)[-1] == "" and _replans(node) == []
    assert any("names none of its objects" in m for _, m in node.f_logs)


def test_a_given_up_pick_ends_the_plan_and_the_question():
    node = _confirming(_about_to_pick())
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (5.0, 0.2))])
    _done(node, "testbot_a", "PICK", obj="o58", ok=False, note="operator_gave_up")
    assert _replans(node) == [] and _pending(node) is None


def test_a_robot_asked_about_is_watched_as_without_a_hold():
    node = _confirming(_node(_map(), poses={"testbot_a": np.zeros(4)}))
    _carrying(node)
    _scene(node, [_rm("O58")])
    node.f_clock["t"] = 61.0
    node._progress_tick()  # it carries on: a real stall is still reported
    assert len(node.f_pubs["~/progress_alert"].sent) == 1


# ---------- with confirm_replans on and no change detection, as main ----------

MAIN_TESTS = sorted(n for n in dir(W) if n.startswith("test_"))


@pytest.mark.parametrize("name", MAIN_TESTS)
def test_the_repair_node_tests_pass_with_confirm_replans_on(name, monkeypatch):
    """Without a change detection report nothing is held: every repair node
    test passes as it is with confirm_replans on."""
    made = W._node
    monkeypatch.setattr(W, "_node", lambda *a, **k: _confirming(made(*a, **k)))
    test = getattr(W, name)
    kwargs = {}
    if "monkeypatch" in inspect.signature(test).parameters:
        kwargs["monkeypatch"] = monkeypatch
    test(**kwargs)


# ---------- review of 940b1db ----------


def test_a_retried_last_pick_carries_the_held_change():
    node = _confirming(_node(_map(), poses={"testbot_b": np.zeros(4)}))
    node._planning_goal = "(and (holding testbot_b o60) (visited-object o58))"
    node.finalize_plans({"testbot_b": _visit("testbot_b", "v")}, node.f_poses)
    _scene(node, [_rm("O58")])
    assert _pending(node)["robots"] == ["testbot_b"]
    _done(node, "testbot_b", "FOLLOW", plan_id="v")
    _done(node, "testbot_b", "PICK", plan_id="v", obj="o60", ok=False)  # its last
    (reason,) = _replans(node)
    assert reason.startswith("pick of o60 by testbot_b failed (attempt 1 of 2)")
    assert removed_symbols(reason) == {"o58"} and _questions(node)[-1] == ""


def test_an_edit_that_stops_a_held_husky_carries_the_held_change():
    node = _confirming(_node(_map(), poses={"testhusky": np.zeros(4)}))
    node._planning_goal = "(visited-place t2)"
    node.finalize_plans({"testhusky": _path((1, 0), (2, 0))}, node.f_poses)
    node._plan_deps = {"o58": ("visit", (2.0, 0.2)), "o59": ("visit", (3.0, 0.2))}
    _scene(node, [_rm("O58")])
    assert _pending(node)["robots"] == ["testhusky"]
    _scene(node, [_rm("O59", (3.0, 0.2))], source="console/map_edit", version=5)
    node._stop_seen("testhusky")  # its empty Path, seen back
    _map_arrives(node, 5)
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58", "o59"}


def test_a_confirmed_replan_waiting_for_its_map_is_sent_once():
    node = _confirming(_node(_map(), poses={"testbot_b": np.zeros(4)}))
    _two(node)
    _scene(node, [_rm("O58")], version=5)
    _decide(node, "confirm")  # waits for map version 5
    _done(node, "testbot_b", "PICK", plan_id="v", obj="o60", ok=False)
    (reason,) = _replans(node)  # the retry carries the confirmed change
    assert reason.startswith("pick of o60") and removed_symbols(reason) == {"o58"}
    _map_arrives(node, 5)
    assert len(_replans(node)) == 1


# ---------- v4: one question about the running goal ----------


def _scoped(node):
    """Robot-scoped goals: testbot_a carries o58, testbot_b visits o60, the
    Husky drives to t2."""
    _carrying(node)
    node._pending_busy = {"testbot_a", "testhusky"}
    node._planning_goal = "(holding testbot_b o60)"
    node.finalize_plans({"testbot_b": _visit("testbot_b", "v")}, node.f_poses)
    node._pending_busy = {"testbot_a", "testbot_b"}
    node._planning_goal = "(visited-place t2)"
    node.finalize_plans({"testhusky": _path((1, 0), (2, 0))}, node.f_poses)


def test_the_robots_shown_are_those_whose_plans_act_on_the_change():
    node = _confirming(_node(_map(), poses={"testhusky": np.zeros(4)}))
    _scoped(node)
    _scene(node, [_rm("O60", (1.0, 0.2))])
    assert _pending(node)["robots"] == ["testbot_b"]


def test_a_repeated_report_asks_nothing_new():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    q = _pending(node)
    for v in (1, 2, 3):
        _scene(node, [_rm("O58")], version=v)
    assert _pending(node) == q and len(_questions(node)) == 1
    _decide(node, "confirm")
    _map_arrives(node, 3)
    (reason,) = _replans(node)
    assert reason.count("o58 removed") == 1


def test_a_declined_change_is_not_asked_again_until_the_goal_changes():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    _decide(node, "decline")
    _scene(node, [_rm("O58")], version=2)  # reported again
    assert _questions(node)[-1] == "" and _replans(node) == []
    _carrying(node)  # the same goal planned again
    _scene(node, [_rm("O58")], version=3)
    assert _questions(node)[-1] == ""
    node._planning_goal = "(object-in-place o58 t3)"  # a new goal
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    _scene(node, [_rm("O58")], version=4)
    assert "o58 removed" in _pending(node)["reason"]


def test_a_new_plan_still_naming_a_held_object_asks_again_under_a_new_id():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    q = _pending(node)
    node._pending_busy = {"testbot_b", "testhusky"}  # use-robots testbot_a
    node._planning_goal = "(object-in-place o58 t3)"
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    again = _pending(node)
    assert again["id"] != q["id"] and again["reasons"] == q["reasons"]
    assert again["robots"] == q["robots"] and _replans(node) == []
    _decide(node, "confirm", q["id"])  # the old id
    assert _replans(node) == []
    _decide(node, "confirm")
    assert len(_replans(node)) == 1


def test_a_stale_question_does_not_ride_on_a_later_replan():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58"), _rm("O60", (1.0, 0.2))])
    node._planning_goal = "(visited-place t3)"  # the console's goal without them
    node.finalize_plans({"testbot_a": _visit("testbot_a", "n")}, node.f_poses)
    assert _questions(node)[-1] == ""
    node._planning_goal = "(and (object-in-place o60 t1) (visited-place t3))"
    node.finalize_plans({"testbot_b": _visit("testbot_b", "g3")}, node.f_poses)
    assert _questions(node)[-1] == ""
    _scene(node, [(MOVED, "O60", (1.0, 0.2), (1.0, 3.0))], source="sget/chat")
    (reason,) = _replans(node)
    assert "o60 moved" in reason and "removed" not in reason


def test_one_plan_ending_keeps_the_question_and_the_last_clears_it():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    q = _pending(node)
    node._stop_seen("testbot_b")
    assert _pending(node) == q
    node._stop_seen("testbot_a")
    assert _questions(node)[-1] == ""
    _decide(node, "confirm", q["id"])
    assert _replans(node) == []


def test_a_confirm_waits_for_the_newest_map_of_its_epoch_only():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")], version=5)
    _scene(node, [_rm("O60", (1.0, 0.2))], version=3)  # an older version
    assert node._hold_state()["map"] == (5, "e1")
    node.dsg_last = _map("e2", 1)  # a reload: a new epoch
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (5.0, 0.2))], version=1)
    assert node._hold_state()["map"] == (1, "e2")
    _decide(node, "confirm")
    (reason,) = _replans(node)  # the planner holds e2 already: no wait
    assert removed_symbols(reason) == {"o58", "o60"}


def test_a_confirm_leaves_an_edits_own_map_wait_alone():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O60", (1.0, 0.2))], source="console/map_edit", version=5)
    _scene(node, [_rm("O58")])  # asked while the edit waits for its map
    _decide(node, "confirm")
    (confirmed,) = _replans(node)  # the confirmed change's map is here
    assert removed_symbols(confirmed) == {"o58"}
    _map_arrives(node, 5)
    edit = _replans(node)[-1]
    assert len(_replans(node)) == 2 and removed_symbols(edit) == {"o60"}


# ---------- review of 9bdbaab ----------


def test_an_edit_moving_an_object_stops_every_robot_as_on_main():
    node = _confirming(_node(_map(), poses={"testhusky": np.zeros(4)}))
    _scoped(node)
    _scene(node, [(MOVED, "O60", (1.0, 0.2), (1.0, 3.0))], source="sget/chat")
    assert len(_stops(node)) == 3  # two pauses and the Husky's empty Path


def test_a_second_confirm_merges_with_the_first_still_waiting():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")], version=5)
    _decide(node, "confirm")
    _scene(node, [_rm("O60", (1.0, 0.2))], version=6)
    _decide(node, "confirm")
    node.dsg_last = _map("e1", 5)
    for t in [t for t in node.f_timers if t.period == 0.2]:
        t.f = True
        t.cb()  # one tick: version 5 is here
    assert _replans(node) == []  # the merged request needs version 6
    _map_arrives(node, 6)
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58", "o60"}


def test_a_confirmed_request_leaves_a_newer_question_open():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")], version=5)
    _decide(node, "confirm")
    _scene(node, [_rm("O60", (1.0, 0.2))], version=9)  # asked while it waits
    q = _pending(node)
    _map_arrives(node, 5)
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58"} and _pending(node) == q


def test_a_declined_object_reported_otherwise_is_asked_again():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    _decide(node, "decline")
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (2.1, 0.2))])  # asks nothing
    assert _questions(node)[-1] == ""
    _scene(node, [_rm("O58")], version=2)
    assert "o58 removed" in _pending(node)["reason"]


def test_robots_left_with_an_empty_plan_do_not_keep_the_question():
    node = _confirming(_node(_map()))
    node._planning_goal = CARRY_GOAL
    node.finalize_plans(
        {
            "testbot_a": _carry(),
            "testbot_b": repair.ActionSequence("e", "testbot_b", []),
        },
        node.f_poses,
    )
    _scene(node, [_rm("O58")])
    node._stop_seen("testbot_a")
    assert "testbot_b" in node._running_plans and _questions(node)[-1] == ""


# ---------- review of 22f0375 ----------


def test_a_question_judged_against_a_replaced_goal_is_judged_again():
    node = _confirming(_node(_map()))
    _carrying(node)
    old = node._active_goal
    node._planning_goal = "(object-in-place o58 t3)"  # a new plan went out
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    changes = [
        {"symbol": "O58", "kind": REMOVED, "old": (2.0, 0.2), "new": None,
         "source": "awcd/x", "held": True}
    ]  # fmt: skip
    held = [("o58", "o58 removed, and the plan still has to pick it")]
    assert node._ask_operator(["testbot_a"], held, changes, 1, "e1", old)
    assert _questions(node) == [] and node._pending_changes == changes
    node._judge_changes()  # against the new goal
    assert "o58 removed" in _pending(node)["reason"]


def test_an_edit_keeps_the_held_reasons_it_carries_tracked():
    node = _confirming(_node(_map(), poses={"testbot_b": np.zeros(4)}))
    _two(node)
    _scene(node, [_rm("O58")])
    _scene(node, [_rm("O60", (1.0, 0.2))], source="console/map_edit", version=5)
    _done(node, "testbot_b", "PICK", plan_id="v", obj="o61", ok=False)
    (pick,) = _replans(node)  # before the edit's map: it carries o58
    assert pick.startswith("pick of o61") and removed_symbols(pick) == {"o58"}
    _map_arrives(node, 5)
    edit = _replans(node)[-1]  # the edit's own request: o58 not again
    assert len(_replans(node)) == 2 and removed_symbols(edit) == {"o60"}


def test_declined_changes_reset_on_any_report_and_every_kind_in_it():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    _decide(node, "decline")
    assert node._hold_state()["declined"]
    node._scene_changes_callback(
        _msg([(ADDED, "O58", None, (2.0, 0.2))], "executor/testbot_a", 2)
    )
    assert node._hold_state()["declined"] == set()
    _scene(node, [_rm("O58")], version=3)
    _decide(node, "decline")
    node._scene_changes_callback(
        _msg([_rm("O58"), (ADDED, "O58", None, (2.0, 0.2))], "awcd/x", 4)
    )  # removed, then added: changed otherwise
    assert node._hold_state()["declined"] == set()


def test_declined_changes_go_when_the_mission_ends():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    _decide(node, "decline")
    node._stop_seen("testbot_a")  # no plan runs any more
    _carrying(node)  # the same goal, as a new mission
    _scene(node, [_rm("O58")], version=2)
    assert "o58 removed" in _pending(node)["reason"]


def test_a_new_plan_settles_the_question_under_the_plan_lock():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    seen = []
    replanned = node._goal_replanned
    node._goal_replanned = lambda: seen.append(node._plan_lock.locked()) or replanned()
    _carrying(node)
    assert seen == [True]


# ---------- review of ba9a4b8 ----------


def test_a_new_plan_with_nobody_listening_replans_the_held_change_at_once():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    node._pending_pub.subscribers = 0  # the console went away
    node._planning_goal = "(object-in-place o58 t3)"
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    assert _questions(node)[-1] == ""
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58"}


def test_a_repeated_report_while_the_confirm_waits_asks_nothing():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")], version=5)
    _decide(node, "confirm")  # waits for map version 5
    _scene(node, [_rm("O58")], version=5)  # change detection reports it again
    assert _questions(node)[-1] == ""
    _map_arrives(node, 5)
    (reason,) = _replans(node)
    assert reason.count("o58 removed") == 1


def test_a_new_question_needs_only_its_own_map_version():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")], version=9)
    _decide(node, "decline")
    _scene(node, [_rm("O60", (1.0, 0.2))], version=1)
    assert node._hold_state()["map"] == (1, "e1")
    _decide(node, "confirm")
    (reason,) = _replans(node)  # the map has version 1: no wait
    assert removed_symbols(reason) == {"o60"}


def test_with_nobody_listening_a_mixed_batch_is_judged_as_on_main():
    node = _confirming(_node(_map()))
    node._pending_pub.subscribers = 0
    _two(node)
    node._scene_changes_callback(_msg([_rm("O58")], "awcd/hamilton", 1))
    _scene(node, [_rm("O60", (1.0, 0.2))], source="sget/chat")
    (reason,) = _replans(node)  # one judgement, in the order reported
    assert reason.index("o58") < reason.index("o60") and reason.count(SCENE) == 1


def test_each_held_change_keeps_its_own_kind_and_reason():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58"), (ADDED, "O58", None, (2.0, 0.2))])
    q = _pending(node)
    assert q["reasons"] == [
        "scene changed: o58 removed, and the plan still has to pick it",
        "scene changed: o58 added, and the plan still has to pick it",
    ]
    assert node._hold_state()["keys"] == [("o58", REMOVED), ("o58", ADDED)]
