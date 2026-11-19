"""Held replans (confirm_replans): a replan for a change detection scene
change (source awcd/) waits for the operator's decision; a map edit (sget/,
console/), a cut route, a map reload and a failed pick replan at once, as
without it.

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


def _scene(node, changes, source="awcd/hamilton", version=1):
    """A scene change message, judged once it settled. The map already has
    version 1; a later version is waited for (see _map_arrives)."""
    msg = NS(
        source=source,
        map_version=version,
        changes=[
            NS(
                kind=kind,
                symbol=sym,
                old_position=_xy(old),
                new_position=_xy(new),
            )
            for kind, sym, old, new in changes
        ],
    )
    node._scene_changes_callback(msg)
    node._judge_changes()


def _rm(sym, xy=(2.0, 0.2)):
    return (REMOVED, sym, xy, None)


def _map_arrives(node, version):
    node.dsg_last = _map("e1", version)
    while True:  # the 0.2 s map wait, run to its end
        waits = [
            t for t in node.f_timers if t.period == 0.2 and not t.__dict__.get("f")
        ]
        if not waits:
            return
        waits[-1].f = True
        waits[-1].cb()


def _paused(node, robot):
    pub = node.f_pubs.get(f"/{robot}/spot_executor_node/pause")
    return [m.data for m in pub.sent] if pub else []


def _carrying(node):
    node._planning_goal = CARRY_GOAL
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)


def _two(node):
    node._planning_goal = GOAL2
    node.finalize_plans(
        {"testbot_a": _carry(), "testbot_b": _visit("testbot_b", "v")}, node.f_poses
    )


# ---------- which replans are held ----------


def test_a_change_detection_change_is_held_until_confirmed():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")], source="awcd/hamilton")
    assert _replans(node) == []
    assert _paused(node, "testbot_a") == [True]
    q = _pending(node)
    assert set(q) == QUESTION_KEYS
    assert q["robots"] == ["testbot_a"] and q["source"] == "awcd/hamilton"
    assert len(q["reasons"]) == 1 and q["reason"] == q["reasons"][0]
    assert "o58 removed" in q["reason"] and isinstance(q["t"], float)
    assert len(q["id"]) == 32
    _decide(node, "confirm")
    (reason,) = _replans(node)
    assert "o58 removed" in reason and _questions(node)[-1] == ""


@pytest.mark.parametrize("source", ["sget/chat", "console/map_edit", "other"])
def test_a_map_edit_replans_at_once(source):
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")], source=source)
    (reason,) = _replans(node)
    assert "o58 removed" in reason
    assert _paused(node, "testbot_a") == [True] and _questions(node) == []


def test_a_burst_of_an_edit_and_a_detection_replans_both_at_once():
    node = _confirming(_node(_map()))
    _two(node)
    node._scene_changes_callback(
        NS(
            source="console/map_edit",
            map_version=1,
            changes=[NS(kind=REMOVED, symbol="O60", old_position=_xy((1.0, 0.2)),
                        new_position=_xy(None))],
        )
    )  # fmt: skip
    _scene(node, [_rm("O58")])  # within the same settle
    (reason,) = _replans(node)  # the edit's replan carries the held change
    assert removed_symbols(reason) == {"o58", "o60"}
    assert reason.count("scene changed: ") == 1
    assert _pending(node) is None


def test_an_executor_placement_is_still_ignored():
    node = _confirming(_node(_map()))
    _carrying(node)
    node._scene_changes_callback(
        NS(source="executor/testbot_a", map_version=1, changes=[])
    )
    assert node._pending_changes == [] and _questions(node) == []


def test_without_confirm_replans_an_object_change_replans_at_once():
    node = _confirming(_node(_map()), on=False)
    _carrying(node)
    _scene(node, [_rm("O58")])
    (reason,) = _replans(node)
    assert "o58 removed" in reason and _questions(node) == []


def test_a_cut_route_replans_at_once():
    node = _confirming(_node(_map("e1", 4)))
    node.finalize_plans({"testbot_a": _carry()}, {"testbot_a": np.zeros(4)})
    node._note_map_state()
    node.dsg_last = _map("e1", 5, places={k: v for k, v in PLACES.items() if k != "t3"})
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    (reason,) = _replans(node)
    assert "t3" in reason and _questions(node) == []


def test_a_reload_replans_at_once():
    node = _confirming(_node(_map("e1", 4)))
    _carrying(node)
    node._note_map_state()
    node.dsg_last = _map("e2", 1, o58=(3.5, 0.2))  # o58 moved 1.5 m
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    (reason,) = _replans(node)
    assert reason.startswith("map reloaded: ") and "o58 moved" in reason
    assert _questions(node) == []


def test_a_failed_pick_is_replanned_at_once_during_a_hold():
    node = _confirming(_node(_map(), poses={"testbot_b": np.zeros(4)}))
    _two(node)
    _scene(node, [_rm("O58")])
    _done(node, "testbot_b", "PICK", plan_id="v", obj="o60", ok=False)
    pick, held = _replans(node)  # the held change right after, not lost
    assert "pick of o60 by testbot_b failed (attempt 1 of 2)" in pick
    assert removed_symbols(held) == {"o58"} and _pending(node) is None


def test_a_given_up_pick_is_not_replanned_and_leaves_the_question():
    node = _confirming(_about_to_pick())
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (5.0, 0.2))])
    qid = _pending(node)["id"]
    _done(node, "testbot_a", "PICK", obj="o58", ok=False, note="operator_gave_up")
    assert _replans(node) == [] and _pending(node) is None
    _decide(node, "confirm", qid)
    assert _replans(node) == []


# ---------- the question ----------


def _confirmed_two_removals():
    """A hold of two removed goal objects, in two questions, confirmed: the
    one replan request it sent."""
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


def test_a_declined_replan_is_not_sent_and_cannot_be_confirmed_after():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    qid = _pending(node)["id"]
    _decide(node, "decline")
    assert _replans(node) == [] and _pending(node) is None
    _decide(node, "confirm", qid)
    assert _replans(node) == []
    # the plan is still watched: a later change asks again, with a new id
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (5.0, 0.2))])
    assert _pending(node)["id"] != qid


def test_moves_within_move_threshold_are_not_changes():
    node = _confirming(_node(_map()))
    _carrying(node)
    for dx in (0.05, 0.08, 0.3):
        _scene(node, [(MOVED, "O58", (2.0, 0.2), (2.0 + dx, 0.2))])
    assert _paused(node, "testbot_a") == [] and _questions(node) == []
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (2.8, 0.2))])
    assert "o58 moved 0.8 m" in _pending(node)["reason"]


def test_an_added_object_asks_nothing():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [(ADDED, "O99", None, (1.0, 1.0))])
    assert _questions(node) == [] and _replans(node) == []


# ---------- the robots held ----------


def test_a_new_plan_for_some_held_robots_lets_only_those_go():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58"), _rm("O60", (1.0, 0.2))])
    q = _pending(node)
    assert q["robots"] == ["testbot_a", "testbot_b"]
    node._pending_busy = {"testbot_b", "testhusky"}  # use-robots testbot_a
    node._planning_goal = "(object-in-place o58 t3)"
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    left = _pending(node)
    assert left["robots"] == ["testbot_b"] and left["id"] == q["id"]
    assert left["reasons"] == q["reasons"]
    _decide(node, "confirm")
    assert len(_replans(node)) == 1 and _pending(node) is None


def test_a_new_plan_for_every_held_robot_clears_the_question():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    qid = _pending(node)["id"]
    _carrying(node)  # the operator's new goal
    assert _questions(node)[-1] == ""
    (reason,) = _replans(node)  # the held change is not dropped
    assert removed_symbols(reason) == {"o58"}
    _decide(node, "confirm", qid)
    assert len(_replans(node)) == 1


def _husky_echoing(node):
    """The node's own subscription to the Husky's plan topic: an empty Path
    comes back as a stop (queued, as ROS delivers it later)."""
    ad = node.robot_adaptors["testhusky"]
    orig, queue = ad.publish_plan, []

    def pub(msg):
        orig(msg)
        if not msg.poses:
            queue.append("testhusky")

    ad.publish_plan = pub
    return queue


def test_a_husky_stopped_for_the_hold_stays_held():
    node = _confirming(_node(_map(), poses={"testhusky": np.zeros(4)}))
    echoes = _husky_echoing(node)
    node._planning_goal = "(visited-object testhusky o58)"
    node.finalize_plans({"testhusky": _path((1, 0), (2, 0))}, node.f_poses)
    node._plan_deps = {"o58": ("visit", (2.0, 0.2))}
    _scene(node, [_rm("O58")], version=5)  # the map lags: a wait
    assert echoes == ["testhusky"] and _questions(node) == []
    for robot in echoes:
        node._stop_seen(robot)  # seen back before the map arrives
    assert node._plan_robots == ["testhusky"]  # still judged while held
    _map_arrives(node, 5)
    assert _pending(node)["robots"] == ["testhusky"]


def test_a_stop_of_a_held_robot_takes_it_out():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    qid = _pending(node)["id"]
    node._stop_seen("testbot_a")  # the operator stopped it
    assert _pending(node)["robots"] == ["testbot_b"]
    assert _pending(node)["id"] == qid
    node._stop_seen("testbot_b")
    assert _pending(node) is None
    _decide(node, "confirm", qid)
    assert _replans(node) == []


HUSKY_GOAL = "(and (visited-object testhusky o58) (visited-object testhusky o59))"


def _held_husky():
    node = _confirming(_node(_map(), poses={"testhusky": np.zeros(4)}))
    echoes = _husky_echoing(node)
    node._planning_goal = HUSKY_GOAL
    node.finalize_plans({"testhusky": _path((1, 0), (2, 0))}, node.f_poses)
    node._plan_deps = {"o58": ("visit", (2.0, 0.2)), "o59": ("visit", (3.0, 0.2))}
    _scene(node, [_rm("O58")])
    for robot in echoes:
        node._stop_seen(robot)  # its own empty Path, seen back
    echoes.clear()
    return node


def test_a_further_change_for_a_held_husky_joins_the_question():
    node = _held_husky()
    first = _pending(node)
    _scene(node, [_rm("O59", (3.0, 0.2))])
    q = _pending(node)
    assert q["id"] != first["id"] and q["robots"] == ["testhusky"]
    assert len(q["reasons"]) == 2
    _decide(node, "confirm")
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58", "o59"}


def test_any_empty_path_keeps_a_held_husky_held_and_a_decline_ends_its_plan():
    node = _held_husky()
    node._stop_seen("testhusky")  # late, or another node's: stopped either way
    assert _pending(node)["robots"] == ["testhusky"]
    assert node._plan_robots == ["testhusky"]
    _decide(node, "decline")
    assert _replans(node) == [] and node._plan_robots == []
    assert node._active_goal == ""


def test_a_robot_the_pause_did_not_reach_is_replanned_at_once():
    node = _confirming(_node(_map()))
    _two(node)
    node.f_pubs["/testbot_b/spot_executor_node/pause"].subscribers = 0
    _scene(node, [_rm("O58")])
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58"} and _questions(node) == []
    assert any("did not reach testbot_b" in m for m in W._warnings(node))


def test_a_change_asked_after_an_answer_pauses_its_robots_again():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")])
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (5.0, 0.2))], version=5)  # waits
    assert _paused(node, "testbot_a") == [True, True]
    _decide(node, "decline")  # the console resumes testbot_a
    _map_arrives(node, 5)
    assert _paused(node, "testbot_a") == [True, True, True]
    q = _pending(node)
    assert q["robots"] == ["testbot_a"] and len(q["reasons"]) == 1
    assert "o58 moved" in q["reason"]


def test_a_finished_robot_leaves_the_question():
    node = _confirming(_node(_map()))
    node._planning_goal = GOAL2
    seq = _visit("testbot_b", "v")
    node.finalize_plans({"testbot_a": _carry(), "testbot_b": seq}, node.f_poses)
    _scene(node, [_rm("O58")])
    _done(node, "testbot_b", "FOLLOW", plan_id="v")
    _done(node, "testbot_b", "PICK", plan_id="v", obj="o60")  # its last action
    assert _pending(node)["robots"] == ["testbot_a"]


def test_a_map_wait_does_not_ask_once_its_robots_have_a_new_plan():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")], version=5)
    _carrying(node)  # a new plan while the map is awaited
    waits = len(node.f_timers)
    node.f_timers[-1].cb()  # the map is still old: the wait stops anyway
    assert len(node.f_timers) == waits
    _map_arrives(node, 5)
    assert _questions(node) == []
    (reason,) = _replans(node)  # sent with the new plan, not asked
    assert removed_symbols(reason) == {"o58"}


def test_a_map_wait_does_not_bring_back_a_cleared_question():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    qid = _pending(node)["id"]
    _scene(node, [_rm("O60", (1.0, 0.2))], version=5)  # waits for the map
    _decide(node, "decline", qid)
    _two(node)  # the operator's new goal for both
    _map_arrives(node, 5)
    assert _questions(node)[-1] == ""
    (reason,) = _replans(node)  # the waiting change went with the new plan
    assert removed_symbols(reason) == {"o60"}


def test_a_map_wait_for_a_robot_whose_plan_ended_asks_nothing():
    node = _confirming(_node(_map()))
    _carrying(node)
    _scene(node, [_rm("O58")], version=5)
    node._stop_seen("testbot_a")
    _map_arrives(node, 5)
    assert _questions(node) == []


def test_no_robots_no_question():
    node = _confirming(_node(_map()))
    node._active_goal = CARRY_GOAL  # a goal, but no plan running
    _scene(node, [_rm("O58")])
    assert _questions(node) == [] and _replans(node) == []


# ---------- the progress watchdog ----------


def _statuses(node, robot):
    pub = node.f_pubs.get(f"/{robot}/omniplanner_node/path_status")
    return [json.loads(m.data) for m in pub.sent] if pub else []


def test_a_stall_flagged_before_the_hold_is_cleared_while_held():
    node = _confirming(_node(_map(), poses={"testbot_a": np.zeros(4)}))
    _carrying(node)
    node.f_clock["t"] = 61.0
    node._progress_tick()
    assert node.status_warnings() == ["testbot_a stalled on plan carry"]
    _scene(node, [_rm("O58")])
    node.f_clock["t"] = 62.0
    node._progress_tick()
    assert node.status_warnings() == []


def test_a_held_robot_is_not_flagged_and_is_reported_as_progress():
    node = _confirming(_node(_map(), poses={"testbot_a": np.zeros(4)}))
    _carrying(node)
    _scene(node, [_rm("O58")])
    for t in range(1, 301):
        node.f_clock["t"] = float(t)
        node._progress_tick()
    assert node.f_pubs["~/progress_alert"].sent == []
    pings = _statuses(node, "testbot_a")
    assert len(pings) >= 25 and all(
        s["status"] == "progress" and s["plan_id"] == "carry" for s in pings
    )
    _decide(node, "decline")  # resumed by the console; the clock starts now
    node.f_clock["t"] = 350.0
    node._progress_tick()
    assert node.f_pubs["~/progress_alert"].sent == []
    node.f_clock["t"] = 361.0
    node._progress_tick()
    assert len(node.f_pubs["~/progress_alert"].sent) == 1


def test_the_goal_manager_does_not_stall_a_held_spot(tmp_path):
    pytest.importorskip("robot_executor_msgs.msg")
    from goal_manager_harness import Harness
    from std_msgs.msg import Bool, String

    h = Harness(tmp_path)
    try:
        h.send("(holding euclid o58)")
        h.plan(euclid=("carry", 4, ["o58"]))
        node = _confirming(
            _node(
                _map(),
                poses={"euclid": np.zeros(4)},
                adaptors={"euclid": W._adaptor("euclid")},
            )
        )
        node._planning_goal = "(holding euclid o58)"
        node.finalize_plans({"euclid": _carry("euclid")}, node.f_poses)
        _scene(node, [_rm("O58")])
        h.node._pause_seen_cb("euclid", Bool(data=True))  # the hold's pause
        seen = 0
        for t in range(1, 201):
            node.f_clock["t"] = float(t)
            h.t += 1.0
            node._progress_tick()
            sent = node.f_pubs["/euclid/omniplanner_node/path_status"].sent
            for m in sent[seen:]:
                h.node._path_status_cb("euclid", String(data=m.data))
            seen = len(sent)
            h.node._check_progress()
        assert h.node._robots["euclid"].status == "running"
        assert not [t for t in h.texts if "no progress" in t]
    finally:
        h.close()


# ---------- with confirm_replans on and no scene change, as main ----------

MAIN_TESTS = sorted(n for n in dir(W) if n.startswith("test_"))


@pytest.mark.parametrize("name", MAIN_TESTS)
def test_the_repair_node_tests_pass_with_confirm_replans_on(name, monkeypatch):
    """Without a scene change message nothing is held: every repair node
    test passes as it is with confirm_replans on (its scene changes are
    judged straight from _pending_changes, as an immediate replan)."""
    made = W._node
    monkeypatch.setattr(W, "_node", lambda *a, **k: _confirming(made(*a, **k)))
    test = getattr(W, name)
    kwargs = {}
    if "monkeypatch" in inspect.signature(test).parameters:
        kwargs["monkeypatch"] = monkeypatch
    test(**kwargs)


# ---------- a held change is never lost; only held when someone listens ----------


def test_nothing_listening_on_replan_pending_replans_at_once():
    node = _confirming(_node(_map()))
    node._pending_pub.subscribers = 0
    _carrying(node)
    _scene(node, [_rm("O58")])
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58"} and _questions(node) == []
    assert any("nothing listens on ~/replan_pending" in m for m in W._warnings(node))


def test_an_edit_during_a_hold_carries_the_held_change():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    _scene(node, [_rm("O60", (1.0, 0.2))], source="sget/chat")
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58", "o60"}
    assert reason.count("scene changed: ") == 1 and _questions(node)[-1] == ""
    _two(node)  # the replan's plan: nothing left to send
    assert len(_replans(node)) == 1


def test_a_cut_route_during_a_hold_sends_the_held_change_right_after():
    node = _confirming(_node(_map("e1", 4)))
    _carrying(node)
    node._note_map_state()
    _scene(node, [(MOVED, "O58", (2.0, 0.2), (5.0, 0.2))], version=4)
    node.dsg_last = _map("e1", 5, places={k: v for k, v in PLACES.items() if k != "t3"})
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    cut, held = _replans(node)
    assert cut.startswith("map changed: ") and "t3" in cut
    assert held.startswith("scene changed: ") and "o58 moved" in held
    assert _questions(node)[-1] == ""


def test_a_confirm_carries_a_change_still_waiting_for_its_map():
    node = _confirming(_node(_map()))
    _two(node)
    _scene(node, [_rm("O58")])
    _scene(node, [_rm("O60", (1.0, 0.2))], version=5)  # the map lags
    _decide(node, "confirm")  # the question names o58 only
    (reason,) = _replans(node)
    assert removed_symbols(reason) == {"o58", "o60"}
    _two(node)
    _map_arrives(node, 5)
    assert _questions(node)[-1] == "" and len(_replans(node)) == 1


def test_a_held_husky_is_listed_as_stopped():
    node = _confirming(_node(_map(), poses={"testhusky": np.zeros(4)}))
    node._planning_goal = GOAL2
    node.finalize_plans(
        {"testbot_a": _carry(), "testhusky": _path((1, 0), (2, 0))}, node.f_poses
    )
    _scene(node, [_rm("O58")])
    q = _pending(node)
    assert q["robots"] == ["testbot_a", "testhusky"] and q["stopped"] == ["testhusky"]
    node._stop_seen("testbot_a")
    assert _pending(node)["stopped"] == ["testhusky"]
    node.finalize_plans({"testhusky": _path((1, 0), (3, 0))}, node.f_poses)
    assert _questions(node)[-1] == ""
