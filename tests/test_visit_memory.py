"""When a visit is remembered, and when it is forgotten.

A visit is remembered for one reason: so that a replan of the mission it was
made in does not make it again. Everywhere else, a goal that names a place
means go there. These tests pin both halves, for a Husky (whose plan is a Path
and who is stopped, not paused, before every replan) and for a Spot.
"""

from types import SimpleNamespace as NS

import numpy as np
from test_willow_repair_node import _carry, _done, _map, _node, _path, repair

HUSKY, SPOT = "testhusky", "testbot_a"
# Places 3 m apart: t1 (3, 0) ... t5 (15, 0).
ROW = {f"t{i}": (3.0 * i, 0.0) for i in range(1, 6)}


def _drive(node, robot, to_x, frm_x, step=0.25):
    for x in np.arange(frm_x, to_x + 1e-9, step):
        node.f_poses[robot] = np.array([float(x), 0.0, 0.0, 0.0])
        node._progress_tick()


def _husky_to_t2_of_a_path_to_t5():
    node = _node(
        _map(places=ROW, o58=None), poses={HUSKY: np.array([0.0, 0.0, 0.0, 0.0])}
    )
    node._planning_goal = (
        "(and (visited-place-by testhusky t2) (visited-place-by testhusky t5))"
    )
    node.finalize_plans({HUSKY: _path(*ROW.values())}, node.f_poses)
    _drive(node, HUSKY, 6.0, 0.0)
    assert {"t1", "t2"} <= node.world_tracker.visited_by[HUSKY]
    assert HUSKY in node._dispatched
    return node


# ---------- remembered: a replan of the running mission ----------


def test_a_husky_stopped_for_a_replan_keeps_the_visits_of_its_mission():
    """A Husky cannot be paused: before a replan it is stopped with an empty
    Path, which ends its plan. The replan that follows is the same mission and
    must not send it back to t2."""
    node = _husky_to_t2_of_a_path_to_t5()
    node._pause_robots([HUSKY], "a place on the route was removed")
    node._stop_seen(HUSKY)  # the empty Path, seen on its plan topic
    state = node.world_state_for(node._planning_goal)
    assert "t2" in state.visited_by.get(HUSKY, set())


def test_a_paused_spot_keeps_the_visits_of_its_mission():
    node = _node(_map(), poses={SPOT: np.array([0.0, 0.0, 0.0, 0.0])})
    node.finalize_plans({SPOT: _carry()}, node.f_poses)
    node.world_tracker.record(
        "FOLLOW", True, SPOT, points=[(1, 0), (2, 0)], dsg=node.dsg_last
    )
    node._pause_robots([SPOT], "scene changed")
    state = node.world_state_for("(visited-place-by testbot_a t2)")
    assert "t2" in state.visited_by[SPOT]


def test_a_replan_for_one_robot_keeps_what_a_stopped_other_robot_visited():
    """hamilton's pick fails and the mission is replanned; gauss was stopped
    for it mid-path. gauss's part of the replan starts from what it has done."""
    node = _husky_to_t2_of_a_path_to_t5()
    goal = node._planning_goal
    node._stop_seen(HUSKY)
    spot_only = [NS(predicate="use-robots", symbols=[SPOT])]
    node.world_state_for("(holding testbot_a o58)", constraints=spot_only)
    state = node.world_state_for(goal)
    assert "t2" in state.visited_by.get(HUSKY, set())


# ---------- forgotten: the plan finished ----------


def test_a_husky_that_finished_its_path_starts_the_next_goal_with_no_visits():
    node = _husky_to_t2_of_a_path_to_t5()
    _drive(node, HUSKY, 15.0, 6.0)
    assert HUSKY not in node._dispatched
    state = node.world_state_for("(visited-place-by testhusky t2)")
    assert HUSKY not in state.visited_by and "t2" not in state.visited


def test_a_spot_that_finished_its_actions_starts_the_next_goal_with_no_visits():
    node = _node(_map(), poses={SPOT: np.array([0.0, 0.0, 0.0, 0.0])})
    node.finalize_plans({SPOT: _carry()}, node.f_poses)
    node.world_tracker.record(
        "FOLLOW", True, SPOT, points=[(1, 0), (2, 0)], dsg=node.dsg_last
    )
    for action, obj in (
        ("FOLLOW", ""),
        ("PICK", "o58"),
        ("FOLLOW", ""),
        ("PLACE", "o58"),
    ):
        _done(node, SPOT, action, obj=obj)
    assert SPOT not in node._dispatched
    state = node.world_state_for("(visited-place-by testbot_a t2)")
    assert SPOT not in state.visited_by and "t2" not in state.visited


def test_a_finished_robot_goes_again_while_another_still_runs():
    """The field case: gauss was done, the Spots were not, and gauss was sent
    to the same place once more."""
    node = _husky_to_t2_of_a_path_to_t5()
    node.finalize_plans({SPOT: _carry()}, node.f_poses)  # the Spot starts its own
    _drive(node, HUSKY, 15.0, 6.0)
    assert SPOT in node._dispatched and HUSKY not in node._dispatched
    state = node.world_state_for("(visited-place-by testhusky t5)")
    assert "t5" not in state.visited


def test_what_a_running_robot_visited_survives_another_finishing():
    node = _husky_to_t2_of_a_path_to_t5()
    node.finalize_plans({SPOT: _carry()}, node.f_poses)
    node.world_tracker.record("FOLLOW", True, SPOT, points=[(6, 0)], dsg=node.dsg_last)
    _drive(node, HUSKY, 15.0, 6.0)
    state = node.world_state_for("(visited-place t2)")
    assert "t2" in state.visited_by[SPOT] and "t2" in state.visited
    assert "t5" not in state.visited  # only the finished Husky had been there


# ---------- a goal is the team's when it names no robot ----------

TEAM = "(explored-region r1)"


def _both_exploring():
    node = _node(
        _map(places=ROW, o58=None), poses={HUSKY: np.array([0.0, 0.0, 0.0, 0.0])}
    )
    node._planning_goal = TEAM
    node.finalize_plans(
        {HUSKY: _path(*list(ROW.values())[:2]), SPOT: _carry()}, node.f_poses
    )
    _drive(node, HUSKY, 6.0, 0.0)  # the Husky's share is done
    assert HUSKY not in node._dispatched and SPOT in node._dispatched
    return node


def test_a_robot_done_with_its_share_of_a_team_goal_keeps_its_visits_for_the_team():
    """The team has not explored the region yet. A replan for the Spot must
    not plan the Husky's half again."""
    node = _both_exploring()
    state = node.world_state_for(TEAM)
    assert {"t1", "t2"} <= state.visited


def test_the_teams_visits_go_when_the_team_is_done():
    node = _both_exploring()
    for action, obj in (
        ("FOLLOW", ""),
        ("PICK", "o58"),
        ("FOLLOW", ""),
        ("PLACE", "o58"),
    ):
        _done(node, SPOT, action, obj=obj)
    state = node.world_state_for(TEAM)
    assert state.visited == set()


def test_a_robot_done_with_its_own_goal_forgets_while_the_team_goes_on():
    node = _both_exploring()
    node._plan_goals[SPOT] = "(visited-object-by testbot_a o58)"  # the Spot's own
    node._visit_goals = {}
    with node._plan_lock:
        node._visit_goals[HUSKY] = ("(visited-place-by testhusky t2)", True)
    state = node.world_state_for("(visited-place-by testhusky t2)")
    assert "t2" not in state.visited


# ---------- stopped short of the goal ----------


def test_every_robot_stopped_and_the_goal_sent_again_goes_on_from_where_it_was():
    """Stop all, then the same goal: a new mission by the old rule, which
    forgot everything and started the robots over."""
    from omniplanner.world_state import NEW_MISSION

    node = _husky_to_t2_of_a_path_to_t5()
    goal = node._planning_goal
    node._stop_seen(HUSKY)
    state = node.world_state_for(
        goal, constraints=[NS(predicate=NEW_MISSION, symbols=[])]
    )
    assert "t2" in state.visited_by.get(HUSKY, set())


def test_a_stopped_robot_given_another_goal_starts_it_with_no_visits():
    node = _husky_to_t2_of_a_path_to_t5()
    node._stop_seen(HUSKY)
    state = node.world_state_for("(visited-place-by testhusky t1)")
    assert HUSKY not in state.visited_by and "t1" not in state.visited


def test_a_stopped_robot_keeps_its_visits_while_others_are_given_goals():
    node = _husky_to_t2_of_a_path_to_t5()
    goal = node._planning_goal
    node._stop_seen(HUSKY)
    node.world_state_for(
        "(visited-object-by testbot_a o58)",
        constraints=[NS(predicate="use-robots", symbols=[SPOT])],
    )
    state = node.world_state_for(goal)
    assert "t2" in state.visited_by.get(HUSKY, set())


def test_a_pick_the_operator_gave_up_keeps_the_robots_visits():
    """Its plan is over, its goal is not accomplished."""
    node = _node(_map(), poses={SPOT: np.array([0.0, 0.0, 0.0, 0.0])})
    node._planning_goal = (
        "(and (visited-place-by testbot_a t2) (holding testbot_a o58))"
    )
    node.finalize_plans({SPOT: _carry()}, node.f_poses)
    node.world_tracker.record(
        "FOLLOW", True, SPOT, points=[(1, 0), (2, 0)], dsg=node.dsg_last
    )
    _done(node, SPOT, "FOLLOW")
    _done(node, SPOT, "PICK", obj="o58", ok=False, note=repair.OPERATOR_GAVE_UP)
    assert SPOT not in node._dispatched
    state = node.world_state_for(node._planning_goal)
    assert "t2" in state.visited_by.get(SPOT, set())
