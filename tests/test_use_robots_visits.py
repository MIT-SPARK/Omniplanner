"""#67 with robot-scoped plan repair (use-robots): visit memory and re-visits
must leave the robots outside a scoped goal alone, and a scoped replan must
start from what its robots have done."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from test_willow_repair_node import _map, _node, _path  # noqa: E402

HUSKY, SPOT = "testhusky", "testbot_a"
ROW = {f"t{i}": (3.0 * i, 0.0) for i in range(1, 6)}


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _use(msg):
    return sorted(
        s for c in msg.constraints if c.predicate == "use-robots" for s in c.symbols
    )


def _done_path(h, robot):
    h.node._path_status_cb(
        robot, String(data='{"plan_id": "(path)", "status": "done"}')
    )


# ---------- goal manager ----------


def test_a_scoped_replan_does_not_send_back_a_robot_that_finished(h):
    h.send("(and (visited-object-by euclid o1) (visited-object-by hamilton o2))")
    h.plan(euclid=("pe", 3, ["o1"]), hamilton=("ph", 1, ["o2"]))
    h.done("hamilton", "ph")
    h.goals.clear(), h.sent.clear()
    h.node._replan_request_cb(String(data="a place on euclid's route was removed"))
    (sent,) = h.goals
    assert (
        _use(sent) == ["euclid"]
        and sent.goal.pddl_goal == "(visited-object-by euclid o1)"
    )
    assert [r for r, _ in h.sent] == ["euclid"]


def test_a_finished_husky_sent_back_is_scoped_to_it_and_pauses_nobody(h):
    """The 2026-11-18 case: gauss done, a Spot still on its scoped goal."""
    h.send("(visited-object-by euclid o1)")
    h.plan(euclid=("pe", 3, ["o1"]))
    h.send("(visited-place-by gauss t5)")
    h.plan(gauss=("", 0, ["t5"]))
    _done_path(h, "gauss")
    h.goals.clear(), h.sent.clear()
    h.send("(visited-place-by gauss t5)")
    (sent,) = h.goals
    assert (
        _use(sent) == ["gauss"] and sent.goal.pddl_goal == "(visited-place-by gauss t5)"
    )
    assert not any(r == "euclid" for r, _ in h.sent)


def test_a_replan_while_the_sent_back_husky_drives_keeps_its_revisit(h):
    h.send("(visited-object-by euclid o1)")
    h.plan(euclid=("pe", 3, ["o1"]))
    h.send("(visited-place-by gauss t5)")
    h.plan(gauss=("", 0, ["t5"]))
    _done_path(h, "gauss")
    h.send("(visited-place-by gauss t5)")
    h.plan(gauss=("", 0, ["t5"]))  # driving there again
    h.goals.clear()
    h.node._replan_request_cb(String(data="a place on gauss's route was removed"))
    assert any("(visited-place-by gauss t5)" in g.goal.pddl_goal for g in h.goals)


def test_a_scoped_goal_for_one_robot_does_not_reopen_anothers_visit(h):
    h.send("(visited-object-by euclid o1)")
    h.plan(euclid=("pe", 3, ["o1"]))
    h.send("(visited-object-by hamilton o2)")
    h.plan(hamilton=("ph", 1, ["o2"]))
    h.done("hamilton", "ph")  # while euclid still runs
    h.goals.clear()
    h.send("(and (visited-object-by hamilton o2) (visited-object-by euclid o9))")
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(visited-object-by euclid o9)"  # as on main


# ---------- repair node ----------


def _husky_partway(
    goal="(and (visited-place-by testhusky t2) (visited-place-by testhusky t5))",
):
    node = _node(
        _map(places=ROW, o58=None), poses={HUSKY: np.array([0.0, 0.0, 0.0, 0.0])}
    )
    node.world_state_for(
        goal, constraints=[NS(predicate="use-robots", symbols=[HUSKY])]
    )
    node.finalize_plans({HUSKY: _path(*ROW.values())}, node.f_poses)
    for x in np.arange(0, 6.01, 0.25):
        node.f_poses[HUSKY] = np.array([float(x), 0.0, 0.0, 0.0])
        node._progress_tick()
    assert {"t1", "t2"} <= node.world_tracker.visited_by[HUSKY]
    return node, goal


def test_a_scoped_replan_of_a_stopped_husky_starts_from_its_visits():
    node, goal = _husky_partway()
    node._stop_seen(HUSKY)  # stopped for the replan
    state = node.world_state_for(
        goal, constraints=[NS(predicate="use-robots", symbols=[HUSKY])]
    )
    assert {"t1", "t2"} <= state.visited_by[HUSKY]


def test_a_scoped_goal_for_another_robot_keeps_the_stopped_huskys_visits():
    node, goal = _husky_partway()
    node._stop_seen(HUSKY)
    node.world_state_for(
        "(visited-place-by testbot_a t1)",
        constraints=[NS(predicate="use-robots", symbols=[SPOT])],
    )
    state = node.world_state_for(
        goal, constraints=[NS(predicate="use-robots", symbols=[HUSKY])]
    )
    assert {"t1", "t2"} <= state.visited_by[HUSKY]


def test_a_new_scoped_goal_for_the_stopped_husky_starts_it_clean():
    node, _ = _husky_partway()
    node._stop_seen(HUSKY)
    state = node.world_state_for(
        "(visited-place-by testhusky t1)",
        constraints=[NS(predicate="use-robots", symbols=[HUSKY])],
    )
    assert HUSKY not in state.visited_by


def test_a_scoped_replan_leaves_the_running_robot_outside_it_and_its_visits():
    from robot_executor_interface.action_descriptions import ActionSequence, Follow

    node, goal = _husky_partway()
    node.world_state_for(
        "(visited-place t4)", constraints=[NS(predicate="use-robots", symbols=[SPOT])]
    )
    node.finalize_plans(
        {
            SPOT: ActionSequence(
                "s", SPOT, [Follow("map", np.array([[0.0, 0.0], [12.0, 0.0]]))]
            )
        },
        node.f_poses,
    )
    node.world_tracker.record("FOLLOW", True, SPOT, points=[(9, 0)], dsg=node.dsg_last)
    node._stop_seen(HUSKY)
    state = node.world_state_for(
        goal, constraints=[NS(predicate="use-robots", symbols=[HUSKY])]
    )
    assert "t3" in state.visited_by[SPOT] and {"t1", "t2"} <= state.visited_by[HUSKY]
