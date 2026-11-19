"""Findings of the independent (Codex) review of PR 66, each pinned as a test:
what main did, and what the PR must keep or do instead."""

import numpy as np
import pytest

pytest.importorskip("robot_executor_msgs.msg")

from goal_manager_harness import Harness  # noqa: E402
from robot_executor_interface.action_descriptions import (  # noqa: E402
    ActionSequence,
    Follow,
    Pick,
)
from std_msgs.msg import String  # noqa: E402
from test_willow_repair_node import _done, _map, _node, _path, repair  # noqa: E402


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


def _paused(h):
    return sorted(r for r, verb in h.sent if verb in ("pause", "stop"))


# ---------- goal manager ----------


def test_r1_1_a_done_goal_sent_again_while_another_robot_runs_pauses_nobody(h):
    """euclid did o1; hamilton is still running (putting something down). The
    same goal again: euclid may go again, but hamilton is not paused for it."""
    h.send("(visited-object o1)")
    h.plan(euclid=("pe", 1, ["o1"]), hamilton=("ph", 2, []))
    h.done("euclid", "pe")
    assert "hamilton" in h.node._robots
    h.goals.clear(), h.sent.clear()
    h.send("(visited-object o1)")
    assert _paused(h) == []


def test_r1_2_a_robot_that_failed_does_not_count_as_finished(h):
    """euclid reached o1, then failed before o2. Its goal is not accomplished:
    the goal sent again plans only what is left, as on main."""
    goal = "(and (visited-object-by euclid o1) (visited-object o2))"
    h.send(goal)
    h.plan(euclid=("pe", 3, ["o1", "o2"]))
    h.node._action_done_cb  # noqa: B018
    h.done("euclid", "pe", action="GAZE", obj="o1")
    h.done("euclid", "pe", success=False, action="PICK", obj="o9")
    assert "euclid" not in h.node._robots
    h.goals.clear()
    h.send(goal)
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(visited-object o2)"


def test_r1_3_reopening_a_named_visit_does_not_reopen_the_teams_visit_of_the_same_place(
    h,
):
    """euclid (finished) visited o1. A goal naming euclid's visit of o1, the
    team's visit of o1, and hamilton's running o2: as on main, the team's
    visit stays done and only hamilton's work and euclid's revisit are planned,
    without pausing hilbert, who has nothing to do with it."""
    h.send("(visited-object-by hamilton o2)")
    h.plan(hamilton=("ph", 3, ["o2"]))
    h.send("(visited-object-by hilbert o3)")
    h.plan(hilbert=("pi", 3, ["o3"]))
    h.send("(visited-object-by euclid o1)")
    h.plan(euclid=("pe", 1, ["o1"]))
    h.done("euclid", "pe")  # euclid finished while the others run
    h.goals.clear(), h.sent.clear()
    h.send(
        "(and (visited-object-by euclid o1) (visited-object o1) (visited-object-by hamilton o2))"
    )
    assert "hilbert" not in _paused(h)


# ---------- repair node ----------

HUSKY, SPOT = "testhusky", "testbot_a"
LEGS = {"t1": (0.0, 0.0), "t2": (10.0, 0.0), "t3": (10.0, 10.0)}


def _tick(node, robot, *xys):
    for x, y in xys:
        node.f_poses[robot] = np.array([float(x), float(y), 0.0, 0.0])
        node._progress_tick()


def test_r2_1a_a_husky_dispatched_at_the_end_of_its_path_finishes():
    node = _node(
        _map(places={"t1": (0.0, 0.0), "t2": (4.0, 0.0)}, o58=None),
        poses={HUSKY: np.array([4.0, 0.0, 0.0, 0.0])},
    )
    node.finalize_plans({HUSKY: _path((0, 0), (4, 0))}, node.f_poses)
    _tick(node, HUSKY, (4, 0))
    assert HUSKY not in node._dispatched


def test_r2_1b_a_husky_cutting_a_corner_by_a_metre_finishes():
    """Waypoints 10 m apart; the robot rounds the corner 1 m inside it."""
    node = _node(
        _map(places=LEGS, o58=None), poses={HUSKY: np.array([0.0, 0.0, 0.0, 0.0])}
    )
    node.finalize_plans({HUSKY: _path((0, 0), (10, 0), (10, 10))}, node.f_poses)
    pts = (
        [(x, 0.0) for x in np.arange(0, 9.3, 0.25)]
        + [(9.3, 0.7)]
        + [(10.0, y) for y in np.arange(1.0, 10.01, 0.25)]
    )
    _tick(node, HUSKY, *pts)
    assert HUSKY not in node._dispatched


def _team_two_robots():
    G = "(and (visited-place t1) (visited-place t3))"
    node = _node(
        _map(places=LEGS, o58=None), poses={HUSKY: np.array([0.0, 0.0, 0.0, 0.0])}
    )
    node._planning_goal = G
    node.finalize_plans(
        {
            HUSKY: _path((0, 0)),
            SPOT: ActionSequence(
                "p", SPOT, [Follow("map", np.array([[0.0, 0.0], [10.0, 10.0]]))]
            ),
        },
        node.f_poses,
    )
    _tick(node, HUSKY, (0, 0))  # the Husky's share (t1) is done
    assert HUSKY not in node._dispatched and SPOT in node._dispatched
    return node, G


def test_r2_2_a_team_robot_stopped_for_a_replan_keeps_a_finished_robots_share():
    node, G = _team_two_robots()
    node._stop_seen(SPOT)  # stopped for the replan
    state = node.world_state_for(G)
    assert "t1" in state.visited


def test_r2_3_a_given_up_pick_as_the_last_action_keeps_the_visits():
    node = _node(_map(), poses={SPOT: np.array([0.0, 0.0, 0.0, 0.0])})
    G = "(and (visited-place-by testbot_a t2) (holding testbot_a o58))"
    node._planning_goal = G
    seq = ActionSequence(
        "carry",
        SPOT,
        [
            Follow("map", np.array([[0.0, 0.0], [2.0, 0.0]])),
            Pick("map", "box", np.zeros(3), np.array([2.0, 0.2, 0.0]), "o58"),
        ],
    )
    node.finalize_plans({SPOT: seq}, node.f_poses)
    node.world_tracker.record(
        "FOLLOW", True, SPOT, points=[(1, 0), (2, 0)], dsg=node.dsg_last
    )
    _done(node, SPOT, "FOLLOW")
    _done(node, SPOT, "PICK", obj="o58", ok=False, note=repair.OPERATOR_GAVE_UP)
    state = node.world_state_for(G)
    assert "t2" in state.visited_by.get(SPOT, set())


def test_r2_4_an_empty_replan_for_a_finished_robot_keeps_it_finished():
    from types import SimpleNamespace as NS

    from omniplanner.world_state import NEW_MISSION

    node, G = _team_two_robots()
    node.finalize_plans(
        {
            HUSKY: _path(),
            SPOT: ActionSequence(
                "p2", SPOT, [Follow("map", np.array([[0.0, 0.0], [10.0, 10.0]]))]
            ),
        },
        node.f_poses,
    )
    node._stop_seen(HUSKY)  # its empty Path
    _done(node, SPOT, "FOLLOW", plan_id="p2")
    state = node.world_state_for(G, constraints=[NS(predicate=NEW_MISSION, symbols=[])])
    assert state.visited == set()


def test_r2_5_robot_names_in_any_case():
    node = _node(
        _map(places=LEGS, o58=None),
        poses={"TestHusky": np.array([0.0, 0.0, 0.0, 0.0])},
        adaptors={
            "TestHusky": repair_adaptor("TestHusky", True),
            "TestBot": repair_adaptor("TestBot", False),
        },
    )
    G = "(and (visited-place t1) (visited-place t3))"
    node._planning_goal = G
    node.finalize_plans(
        {
            "TestHusky": _path((0, 0)),
            "TestBot": ActionSequence(
                "p", "TestBot", [Follow("map", np.array([[0.0, 0.0], [10.0, 10.0]]))]
            ),
        },
        node.f_poses,
    )
    _tick(node, "TestHusky", (0, 0))
    assert "t1" in node.world_state_for(G).visited


def repair_adaptor(name, husky):
    from test_willow_repair_node import _adaptor

    return _adaptor(name, husky=husky)


def test_r2_6_goal_parts_ignore_spaces_inside_parentheses():
    from omniplanner.world_state import goal_parts

    assert goal_parts(
        "(and ( visited-object-by r p1 ) ( visited-object-by r p2 ))"
    ) == goal_parts("(and (visited-object-by r p1) (visited-object-by r p2))")


# ---------- second review round ----------


def test_s1_1_the_mission_sent_again_after_its_last_robot_failed_keeps_the_finished_part(
    h,
):
    goal = "(and (visited-object-by euclid o1) (visited-object o2))"
    h.send(goal)
    h.plan(euclid=("pe", 1, ["o1"]), hamilton=("ph", 2, ["o2"]))
    h.done("euclid", "pe")
    h.done("hamilton", "ph", success=False, action="PICK", obj="o9")
    h.goals.clear()
    h.send(goal)
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(visited-object o2)"


def test_s1_2_a_later_scoped_goal_sent_again_keeps_its_finished_robots_part(h):
    h.send("(visited-object-by hilbert o3)")
    h.plan(hilbert=("pi", 3, ["o3"]))
    goal = "(and (visited-object-by euclid o1) (visited-object-by hamilton o2))"
    h.send(goal)
    h.plan(euclid=("pe", 1, ["o1"]), hamilton=("ph", 3, ["o2"]))
    h.done("euclid", "pe")
    h.goals.clear()
    h.send(goal)
    (sent,) = h.goals
    assert sent.goal.pddl_goal == "(visited-object-by hamilton o2)"


def test_s1_3_a_named_revisit_next_to_new_work_does_not_reopen_the_teams_visit(h):
    h.send("(and (visited-object o1) (visited-object o2))")
    h.plan(euclid=("pe", 1, ["o1"]), hamilton=("ph", 3, ["o2"]))
    h.done("euclid", "pe")
    h.goals.clear(), h.sent.clear()
    h.send("(and (visited-object-by euclid o1) (visited-object o7))")
    (sent,) = h.goals
    assert _paused(h) == []  # an additional task: hamilton goes on with o2
    assert "o2" not in sent.goal.pddl_goal


def test_s1_4_a_newer_plan_stopped_short_is_not_finished(h):
    h.send("(visited-object o9)")
    h.plan(euclid=("pe", 5, ["o9"]))
    h.send("(visited-object-by gauss o1)")
    h.plan(gauss=("", 0, ["o1"]))
    h.node._path_status_cb(
        "gauss", String(data='{"plan_id": "(path)", "status": "done"}')
    )
    h.send("(visited-object-by gauss o3)")
    h.plan(gauss=("", 0, ["o3"]))
    h.node._drop("gauss", "stopped")  # its newer plan was stopped short
    h.goals.clear()
    h.send("(visited-object-by gauss o1)")
    assert h.goals == []  # as on main: its last plan did not finish


def _track_node(places, path, start):
    node = _node(
        _map(places=places, o58=None), poses={HUSKY: np.array([*start, 0.0, 0.0])}
    )
    node.finalize_plans({HUSKY: _path(*path)}, node.f_poses)
    return node


@pytest.mark.xfail(
    strict=True,
    reason="known limitation: a robot that starts 2 m past its first waypoint and cuts "
    "every corner of a loop by more than the visit radius is credited with the first "
    "waypoint at the end of the loop, and the track since then no longer shows the "
    "corners; main finished it. Keeping the whole track instead breaks paths that pass "
    "the same place twice (back and forth, twice round a loop).",
)
def test_s2_1_a_loop_driven_cutting_every_corner_finishes():
    sq = {"t1": (0.0, 0.0), "t2": (4.0, 0.0), "t3": (4.0, 4.0), "t4": (0.0, 4.0)}
    node = _track_node(sq, [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)], (2.0, 0.0))
    _tick(node, HUSKY, (4, 2), (2, 4), (0, 2), (0, 0), (0, 0), (0, 0))
    assert HUSKY not in node._dispatched


def test_s2_2_an_empty_spot_replan_does_not_count_as_running():
    """A replan gives the finished robot an empty action sequence (no stop
    message follows for it) and the other a real plan; when that one is done,
    a new mission starts clean."""
    from types import SimpleNamespace as NS

    from omniplanner.world_state import NEW_MISSION

    node, G = _team_two_robots()
    follow = Follow("map", np.array([[0.0, 0.0], [10.0, 10.0]]))
    node.finalize_plans(
        {
            HUSKY: ActionSequence("e", HUSKY, []),
            SPOT: ActionSequence("p2", SPOT, [follow]),
        },
        node.f_poses,
    )
    _done(node, SPOT, "FOLLOW", plan_id="p2")
    state = node.world_state_for(G, constraints=[NS(predicate=NEW_MISSION, symbols=[])])
    assert state.visited == set()


def test_s2_3_an_additional_task_keeps_a_stopped_robots_visits():
    from types import SimpleNamespace as NS

    from omniplanner.world_state import BUSY_ROBOTS

    node = _track_node(LEGS, [(0, 0), (10, 0), (10, 10)], (0.0, 0.0))
    node._planning_goal = (
        "(and (visited-place-by testhusky t2) (visited-place-by testhusky t3))"
    )
    node.finalize_plans({HUSKY: _path((0, 0), (10, 0), (10, 10))}, node.f_poses)
    _tick(node, HUSKY, *[(x, 0.0) for x in np.arange(0, 10.01, 0.5)])
    node._stop_seen(HUSKY)
    node.world_state_for(
        "(visited-place t1)", constraints=[NS(predicate=BUSY_ROBOTS, symbols=[SPOT])]
    )
    state = node.world_state_for(
        node._planning_goal
        or "(and (visited-place-by testhusky t2) (visited-place-by testhusky t3))"
    )
    assert "t2" in state.visited_by.get(HUSKY, set())


def test_s2_4_holding_names_a_robot_so_it_is_not_a_team_part():
    from omniplanner.world_state import is_team_part

    assert not is_team_part("(holding spot1 box)", {"spot1", "spot2"})
    assert is_team_part("(visited-object o1)", {"spot1"})


# ---------- third review round ----------


def test_u1_an_empty_path_is_a_husky_latest_plan(h):
    from nav_msgs.msg import Path

    h.send("(visited-object o9)")
    h.plan(euclid=("pe", 5, ["o9"]))
    h.send("(visited-object-by gauss o1)")
    h.plan(gauss=("", 0, ["o1"]))
    h.node._path_status_cb(
        "gauss", String(data='{"plan_id": "(path)", "status": "done"}')
    )
    h.node._path_cb("gauss", Path())  # its next plan: an empty Path
    h.goals.clear()
    h.send("(visited-object-by gauss o1)")
    assert h.goals == []  # as on main


def test_u2_1_a_corner_between_two_pose_updates_still_finishes():
    pts = {"t1": (0.0, 0.0), "t2": (4.0, 0.0), "t3": (4.0, 4.0)}
    node = _track_node(pts, [(0, 0), (4, 0), (4, 4)], (0.0, 0.0))
    _tick(node, HUSKY, (0, 0), (4, 4), (4, 4))  # no pose update on the way
    assert HUSKY not in node._dispatched


def test_u2_2_a_predicate_is_not_a_robot_name():
    from types import SimpleNamespace as NS

    node = _node(
        _map(places=LEGS, o58=None),
        poses={"by": np.array([0.0, 0.0, 0.0, 0.0])},
        adaptors={"by": repair_adaptor("by", True), "r2": repair_adaptor("r2", True)},
    )
    node._planning_goal = "(and (visited-place-by by t2) (visited-place-by by t3))"
    node.finalize_plans({"by": _path((0, 0), (10, 0), (10, 10))}, node.f_poses)
    _tick(node, "by", *[(x, 0.0) for x in np.arange(0, 10.01, 0.5)])
    node._stop_seen("by")
    node.world_state_for(
        "(visited-place-by r2 t1)",
        constraints=[NS(predicate="use-robots", symbols=["r2"])],
    )
    assert "t2" in node.world_state_for(node._planning_goal).visited_by.get("by", set())


# ---------- fourth review round ----------


def test_w1_a_late_done_for_an_earlier_path_does_not_finish_the_latest(h):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    h.send("(visited-object o9)")
    h.plan(euclid=("pe", 5, ["o9"]))
    h.send("(visited-place-by gauss t5)")
    b = Path()
    b.poses = [PoseStamped() for _ in range(7)]
    h.node._path_cb("gauss", b)  # its latest Path: 7 poses
    h.plan(gauss=("", 0, ["t5"]))
    h.node._path_status_cb(
        "gauss",
        String(
            data='{"plan_id": "(path)", "status": "done", "reached": 3, "steps": 3}'
        ),
    )
    assert "gauss" not in h.node._finished


def test_w2_a_pose_outage_mid_path_still_finishes_at_the_end():
    pts = {"t1": (0.0, 0.0), "t2": (6.0, 0.0), "t3": (6.0, 4.0), "t4": (2.0, 4.0)}
    node = _track_node(pts, [(0, 0), (6, 0), (6, 4), (2, 4)], (0.0, 0.0))
    _tick(node, HUSKY, (0, 0), (2, 4), (2, 4))  # no pose on the way
    assert HUSKY not in node._dispatched
