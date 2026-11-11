"""GRSTAPS-X: the use-robots constraint and (holding <robot> <object>) goals.

use-robots keeps every task to the robots it names; the rest of the fleet is
left out of the solve and gets an empty plan. A holding goal is a pick with no
place, pinned to its robot by the identity trait; the holder's species then
carries only that identity, so it can take no other task.

Grounding is checked on the ITAGS input it writes. The plans come from
fake_itags.py, a greedy allocator that, like ITAGS, can only give a task to
robots whose traits cover it -- so these tests check the trait encoding, not
the real solver's search.
"""

import os
import stat
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import spark_dsg
from test_grstaps_regions import G as _G  # noqa: F401  (fixture)
from test_grstaps_regions import _itags_input

from omniplanner.grstaps_planner import (
    MANIP,
    SENSOR,
    TRAIT_DIM,
    GrstapsDomain,
    GrstapsGoal,
    _check_holders,
    _identity_dims,
    enumerate_tasks,
    ground_problem,
    make_plan,
    parse_goal_targets,
)
from omniplanner.world_state import WorldState

FAKE = os.path.join(os.path.dirname(__file__), "fake_itags.py")
DURATIONS = {
    "visit-location": 5.0,
    "inspect-object": 10.0,
    "pick-object": 10.0,
    "place-object": 10.0,
}


@pytest.fixture
def G(_G):  # noqa: F811
    """The regions grid plus o1 at (3, 3) and o2 at (9, 1)."""
    for i, xy in ((1, (3.0, 3.0)), (2, (9.0, 1.0))):
        o = spark_dsg.KhronosObjectAttributes()
        o.position = np.array([xy[0], xy[1], 0.0])
        _G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", i).value, o)
    return _G


def _use(*robots):
    return NS(predicate="use-robots", symbols=list(robots))


POSES = {
    "euclid": np.array([0.0, 0.0, 0.0]),
    "hamilton": np.array([18.0, 6.0, 0.0]),
    "gauss": np.array([10.0, 2.0, 0.0]),
}
SPOTS = {r: "spot" for r in POSES}


def _domain(tmp_path, **kw):
    fake = tmp_path / "itags"
    fake.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    args = dict(
        run_mode="native",
        entry="itags",
        work_dir=str(tmp_path),
        scenario_name="regions",
        itags_binary=str(fake),
        robot_type_species={"spot": "spot_arm"},
        inspect_requirement={"ground": 1, "sensor": 1},
    )
    args.update(kw)
    return GrstapsDomain(**args)


def _goal(pddl, constraints=(), robot_types=None, world_state=None):
    found = parse_goal_targets(pddl)
    return GrstapsGoal(
        goal_points=found["visit"],
        inspect_points=found["inspect"],
        manipulate_points=found["manipulate"],
        manipulate_destinations=found["destinations"],
        robot_bound=found["bound"],
        holding=found["holding"],
        robot_id="willow",
        constraints=list(constraints),
        robot_types=dict(SPOTS if robot_types is None else robot_types),
        world_state=world_state,
    )


def _solve(G, tmp_path, pddl, poses=POSES, **kw):
    """(wrapper names, {robot -> [task names]}, plan) through the fake solver."""
    wrapper = ground_problem(_domain(tmp_path), G, poses, _goal(pddl, **kw))
    plan = make_plan(wrapper.value, G)
    name = {t["id"]: t["name"] for t in plan.tasks}
    got = {r: [] for r in wrapper.names}
    for a in plan.agents:
        got[a["name"]] = [name[i] for i in a["individual_plan"]]
    return list(wrapper.names), got, plan


def _compiled(plan, robot):
    ros = pytest.importorskip("omniplanner_ros.grstaps_planner_ros")
    return ros.compile_grstaps_plan(plan, "p", robot, "map").actions


# -- use-robots ---------------------------------------------------------------


def test_use_robots_one_robot_takes_every_task(G, tmp_path):
    names, got, plan = _solve(
        G,
        tmp_path,
        "(and (visited-object o0)(visited-object o1)(visited-object o2))",
        poses={r: POSES[r] for r in ("euclid", "hamilton")},
        constraints=[_use("euclid")],
    )
    assert sorted(names) == ["euclid", "hamilton"]
    assert [r["name"] for r in _itags_input(tmp_path, "willow")["robots"]] == ["euclid"]
    assert sorted(got["euclid"]) == [
        "visit-location o0 :: visit",
        "visit-location o1 :: visit",
        "visit-location o2 :: visit",
    ]
    assert got["hamilton"] == []
    assert _compiled(plan, "hamilton") == []  # an empty plan: it stops


def test_use_robots_two_of_three_leaves_the_third_idle(G, tmp_path):
    names, got, plan = _solve(
        G,
        tmp_path,
        "(and (visited-object o0)(visited-object o1)(visited-object o2))",
        constraints=[_use("euclid", "Hamilton")],
    )
    assert sorted(names) == ["euclid", "gauss", "hamilton"]
    assert sorted(r["name"] for r in _itags_input(tmp_path, "willow")["robots"]) == [
        "euclid",
        "hamilton",
    ]
    assert got["gauss"] == [] and _compiled(plan, "gauss") == []
    assert sum(len(got[r]) for r in ("euclid", "hamilton")) == 3


def test_use_robots_with_a_by_task_inside_the_set(G, tmp_path):
    _, got, _ = _solve(
        G,
        tmp_path,
        "(and (visited-object-by hamilton o1)(visited-object o2))",
        constraints=[_use("euclid", "hamilton")],
    )
    assert "visit-location o1 :: visit by hamilton" in got["hamilton"]
    assert got["gauss"] == []


def test_a_by_robot_outside_use_robots_is_an_error(G, tmp_path):
    goal = _goal("(visited-object-by gauss o1)", constraints=[_use("euclid")])
    with pytest.raises(
        ValueError,
        match="assigns work to gauss, but use-robots limits the plan to euclid",
    ):
        ground_problem(_domain(tmp_path), G, POSES, goal)


def test_an_unknown_robot_in_use_robots_is_an_error(G, tmp_path):
    goal = _goal("(visited-object o1)", constraints=[_use("euclid", "kepler")])
    with pytest.raises(
        ValueError, match=r"unknown robot 'kepler' \(known: euclid, gauss, hamilton\)"
    ):
        ground_problem(_domain(tmp_path), G, POSES, goal)


def test_use_robots_with_no_usable_robot_is_an_error(G, tmp_path):
    poses = dict(POSES, hamilton=None)
    goal = _goal("(visited-object o1)", constraints=[_use("hamilton")])
    with pytest.raises(ValueError, match="use-robots names.*hamilton has no pose"):
        ground_problem(_domain(tmp_path), G, poses, goal)
    # A husky cannot run a GRSTAPS-X plan, so naming only it plans nothing.
    goal = _goal(
        "(visited-object o1)",
        constraints=[_use("gauss")],
        robot_types=dict(SPOTS, gauss="husky"),
    )
    with pytest.raises(ValueError, match="use-robots names.*gauss: a husky cannot"):
        ground_problem(_domain(tmp_path), G, POSES, goal)


def test_without_use_robots_every_robot_still_plans(G, tmp_path):
    ground_problem(_domain(tmp_path), G, POSES, _goal("(visited-object o1)"))
    assert sorted(r["name"] for r in _itags_input(tmp_path, "willow")["robots"]) == [
        "euclid",
        "gauss",
        "hamilton",
    ]


# -- holding ------------------------------------------------------------------


def test_holding_parses_with_its_robot():
    found = parse_goal_targets("(and (holding Hamilton O1)(visited-object o2))")
    assert found["holding"] == {"hamilton": "o1"}
    assert found["manipulate"] == [] and found["visit"] == ["o2"]


def test_two_holding_goals_for_one_robot_are_an_error():
    with pytest.raises(ValueError, match="Two holding goals for hamilton"):
        parse_goal_targets("(and (holding hamilton o1)(holding hamilton o2))")
    with pytest.raises(ValueError, match="Two robots asked to hold o1"):
        parse_goal_targets("(and (holding hamilton o1)(holding euclid o1))")
    with pytest.raises(ValueError, match="negative"):
        parse_goal_targets("(not (holding hamilton o1))")
    with pytest.raises(ValueError, match="holding <robot> <object>"):
        parse_goal_targets("(holding o1)")


def test_final_poses_and_disjunctions_stay_refused():
    for goal in (
        "(at-place hamilton p3)",
        "(at-object hamilton o1)",
        "(or (visited-object o1)(visited-object o2))",
    ):
        with pytest.raises(ValueError, match="Unsupported GRSTAPS-X goal predicate"):
            parse_goal_targets(goal)


def test_the_hold_task_demands_only_its_robots_identity():
    identity = _identity_dims(["euclid", "hamilton"])
    vertex = {s: {"id": i, "x": float(i), "y": 0.0} for i, s in enumerate(["o1"])}
    goal = GrstapsGoal(
        goal_points=["o1"],
        inspect_points=["o1"],
        robot_id="willow",
        holding={"hamilton": "o1"},
    )
    tasks, of_symbol, precedence = enumerate_tasks(
        goal, DURATIONS, vertex, inspect_traits=[1, 0, 1], identity=identity
    )
    names = [t["name"] for t in tasks]
    # The visit of o1 is covered by the pick; the inspection goes first.
    assert names == [
        "inspect-object o1 :: joint_observation",
        "pick-object o1 :: hold by hamilton",
    ]
    want = [0] * (TRAIT_DIM + 2)
    want[identity["hamilton"]] = 1
    assert tasks[1]["desired_traits"] == want
    assert precedence == [[0, 1]]


def test_holding_robot_does_only_its_pick(G, tmp_path):
    names, got, plan = _solve(
        G,
        tmp_path,
        "(and (holding hamilton o1)(not (suspicious o1))(visited-object o0)"
        "(visited-object o2))",
    )
    assert got["hamilton"] == ["pick-object o1 :: hold by hamilton"]
    others = sorted(got["euclid"] + got["gauss"])
    assert others == [
        "inspect-object o1 :: joint_observation",
        "visit-location o0 :: visit",
        "visit-location o2 :: visit",
    ]
    # The inspection gates the pick.
    ids = {t["name"]: t["id"] for t in plan.tasks}
    assert [
        ids["inspect-object o1 :: joint_observation"],
        ids["pick-object o1 :: hold by hamilton"],
    ] in plan.precedence
    # The holder's species supplies nothing but its identity.
    data = _itags_input(tmp_path, "willow")
    robot = {r["name"]: r["species"] for r in data["robots"]}
    traits = {s["name"]: s["traits"] for s in data["species"]}
    held = traits[robot["hamilton"]]
    assert sum(held) == 1 and held[TRAIT_DIM:].count(1) == 1
    assert traits[robot["euclid"]][MANIP] == 1 and traits[robot["euclid"]][SENSOR]
    # Compiled: drive there and pick, with no place and nothing after it.
    actions = _compiled(plan, "hamilton")
    assert [type(a).__name__ for a in actions] == ["Follow", "Pick"]
    assert actions[-1].object_id == "o1"
    assert actions[-1].after_task_ids == [
        str(ids["inspect-object o1 :: joint_observation"])
    ]


def test_holding_with_use_robots(G, tmp_path):
    _, got, _ = _solve(
        G,
        tmp_path,
        "(and (holding hamilton o1)(visited-object o2))",
        constraints=[_use("euclid", "hamilton")],
    )
    assert got == {
        "euclid": ["visit-location o2 :: visit"],
        "hamilton": ["pick-object o1 :: hold by hamilton"],
        "gauss": [],
    }
    goal = _goal("(holding hamilton o1)", constraints=[_use("euclid")])
    with pytest.raises(ValueError, match="assigns work to hamilton, but use-robots"):
        ground_problem(_domain(tmp_path), G, POSES, goal)


def test_the_holder_alone_cannot_do_the_rest(G, tmp_path):
    goal = _goal(
        "(and (holding hamilton o1)(visited-object o2))",
        constraints=[_use("hamilton")],
    )
    with pytest.raises(ValueError, match="hamilton must end the plan holding o1"):
        ground_problem(_domain(tmp_path), G, POSES, goal)


def test_holding_for_a_husky_is_an_error(G, tmp_path):
    goal = _goal("(holding gauss o1)", robot_types=dict(SPOTS, gauss="husky"))
    with pytest.raises(ValueError, match="gauss is a husky with no arm"):
        ground_problem(_domain(tmp_path), G, POSES, goal)
    # ...even when robot_species puts it back in the fleet.
    with pytest.raises(ValueError, match="gauss is a husky with no arm"):
        ground_problem(
            _domain(tmp_path, robot_species={"gauss": "spot_arm"}), G, POSES, goal
        )


def test_holding_needs_a_manipulator(G, tmp_path):
    goal = _goal("(holding hamilton o1)")
    with pytest.raises(ValueError, match="hamilton .*has no manipulator"):
        ground_problem(_domain(tmp_path, robot_type_species={}), G, POSES, goal)


def test_a_holding_robot_named_in_a_by_task_is_an_error(G, tmp_path):
    goal = _goal("(and (holding hamilton o1)(visited-object-by hamilton o2))")
    with pytest.raises(ValueError, match="hamilton must end the plan holding o1.*-by"):
        ground_problem(_domain(tmp_path), G, POSES, goal)


def test_holding_and_placing_the_same_object_is_an_error(G, tmp_path):
    goal = _goal("(and (holding hamilton o1)(object-in-place o1 o2))")
    with pytest.raises(ValueError, match="cannot both stay in hamilton's hand"):
        ground_problem(_domain(tmp_path), G, POSES, goal)


def test_an_object_already_held_is_kept_and_the_holder_idles(G, tmp_path):
    names, got, plan = _solve(
        G,
        tmp_path,
        "(and (holding hamilton o1)(visited-object o2))",
        world_state=WorldState(holding={"hamilton": ["o1"]}),
    )
    assert "hamilton" in names and got["hamilton"] == []
    assert sorted(r["name"] for r in _itags_input(tmp_path, "willow")["robots"]) == [
        "euclid",
        "gauss",
    ]
    assert _compiled(plan, "hamilton") == []  # no put-down either


def test_an_object_in_another_robots_hand_is_an_error(G, tmp_path):
    goal = _goal(
        "(holding hamilton o1)", world_state=WorldState(holding={"euclid": ["o1"]})
    )
    with pytest.raises(ValueError, match="o1 is in euclid's hand"):
        ground_problem(_domain(tmp_path), G, POSES, goal)


def test_a_solution_giving_the_holder_more_is_refused():
    by_id = {
        0: {"name": "pick-object o1 :: hold by hamilton"},
        1: {"name": "visit-location o2 :: visit"},
    }
    _check_holders(
        {"hamilton": "o1"}, [{"name": "hamilton", "individual_plan": [0]}], by_id
    )
    with pytest.raises(Exception, match="hamilton, which must end holding o1"):
        _check_holders(
            {"hamilton": "o1"},
            [{"name": "hamilton", "individual_plan": [0, 1]}],
            by_id,
        )
