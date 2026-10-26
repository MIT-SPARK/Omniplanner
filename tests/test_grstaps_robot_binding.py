"""GRSTAPS-X keeps the robot a goal names for a task.

The -by predicates pin a task to a robot; grounding gives each robot an
identity trait and makes its pinned tasks demand it. The parsing and task
tests need nothing; the solver tests run the native solver on the
plan_repair_graph prior map and skip without it.
"""

import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import spark_dsg

from omniplanner.grstaps_planner import (
    SENSOR,
    TRAIT_DIM,
    GrstapsGoal,
    _identity_dims,
    enumerate_tasks,
    parse_goal_targets,
    residual_goal,
)
from omniplanner.omniplanner import full_planning_pipeline
from omniplanner.world_state import WorldState
from omniplanner_msgs.msg import ConstrainedPddlGoalMsg
from omniplanner_ros.grstaps_planner_ros import GrstapsConfig, GrstapsRos

DSG = os.path.expanduser(
    os.environ.get(
        "OMNIPLANNER_TEST_DSG",
        "~/colcon_ws/assets/adt4_output/plan_repair_graph/hydra/backend/dsg.json",
    )
)
SOLVER = os.path.expanduser(os.environ.get("ADT4_GRSTAPS_ROOT", "~/grstapsx"))
needs_solver = pytest.mark.skipif(
    not (os.path.exists(DSG) and os.path.isdir(SOLVER)),
    reason="needs the prior map and the GRSTAPS-X solver",
)
DURATIONS = {
    "visit-location": 5.0,
    "inspect-object": 10.0,
    "pick-object": 10.0,
    "place-object": 10.0,
}
VERTEX = {s: {"id": i, "x": float(i), "y": 0.0} for i, s in enumerate(["o2", "o6"])}


def test_by_predicates_pin_their_robot():
    found = parse_goal_targets(
        "(and (visited-object-by hilbert o6)(visited-object-by euclid o6)"
        "(visited-place-by euclid t5)(inspected-by euclid o2)(visited-object o7))"
    )
    assert found["visit"] == ["o7"]
    assert found["bound"] == {
        "visit": {"o6": ["hilbert", "euclid"], "t5": ["euclid"]},
        "inspect": {"o2": ["euclid"]},
    }


def test_negated_by_predicate_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        parse_goal_targets("(not (visited-object-by hilbert o6))")


def test_pinned_tasks_demand_their_robots_identity():
    identity = _identity_dims(["euclid", "hilbert"])
    goal = GrstapsGoal(
        goal_points=["o6"],
        robot_id="hilbert",
        robot_bound={
            "visit": {"o6": ["hilbert", "euclid"]},
            "inspect": {"o2": ["euclid"]},
        },
    )
    tasks, of_symbol, _ = enumerate_tasks(
        goal, DURATIONS, VERTEX, inspect_traits=[1, 0, 2], identity=identity
    )
    by_name = {t["name"]: t["desired_traits"] for t in tasks}
    # The unpinned visit of o6 is covered by the pinned ones, not added again.
    assert sorted(by_name) == [
        "inspect-object o2 :: joint_observation by euclid",
        "visit-location o6 :: visit by euclid",
        "visit-location o6 :: visit by hilbert",
    ]
    assert all(len(v) == TRAIT_DIM + 2 for v in by_name.values())
    hilbert = by_name["visit-location o6 :: visit by hilbert"]
    assert hilbert[SENSOR] == 1 and hilbert[identity["hilbert"]] == 1
    assert hilbert[identity["euclid"]] == 0
    inspect = by_name["inspect-object o2 :: joint_observation by euclid"]
    assert inspect[:3] == [1, 0, 2] and inspect[identity["euclid"]] == 1
    assert len(of_symbol["o6"]) == 2


def test_unpinned_goal_keeps_the_plain_trait_vectors():
    goal = GrstapsGoal(goal_points=["o6"], robot_id="hilbert")
    tasks, _, _ = enumerate_tasks(goal, DURATIONS, VERTEX)
    assert [len(t["desired_traits"]) for t in tasks] == [TRAIT_DIM]


def test_pinned_task_is_done_only_when_that_robot_did_it():
    goal = GrstapsGoal(
        goal_points=[],
        robot_id="hilbert",
        robot_bound={"visit": {"o6": ["hilbert", "euclid"]}},
        world_state=WorldState(visited={"o6"}, visited_by={"euclid": {"o6"}}),
    )
    left, _ = residual_goal(goal, None)
    assert left.robot_bound == {"visit": {"o6": ["hilbert"]}}


# -- with the solver ---------------------------------------------------------


def _node(G, sym):
    for layer in (spark_dsg.DsgLayers.OBJECTS, spark_dsg.DsgLayers.MESH_PLACES):
        for node in G.get_layer(layer).nodes:
            if node.id.str(True).lower() == sym:
                return node
    raise KeyError(sym)


def _assignment(G, goal, poses):
    """{robot -> [task names]} the solver chose for a two-robot fleet."""
    plugin = GrstapsRos(
        GrstapsConfig(
            entry="itags",
            run_mode="native",
            scenario_name="test_robot_binding",
            robot_species={"hilbert": "spot_arm", "euclid": "spot"},
            inspect_requirement={"ground": 2, "sensor": 2},
        )
    )
    plugin._node = NS(
        robot_adaptors={r: NS(robot_type="simulated-spot") for r in poses}
    )
    msg = ConstrainedPddlGoalMsg()
    msg.goal.robot_id, msg.goal.pddl_goal = "hilbert", goal
    request = plugin.constrained_pddl_callback(msg, poses)
    plan = full_planning_pipeline(request, G)
    while hasattr(plan, "value"):
        plan = plan.value
    name = {t["id"]: t["name"] for t in plan.tasks}
    return {
        a["name"]: [name[i] for i in a.get("individual_plan") or []]
        for a in plan.agents
    }


def _beside(G, sym):
    return np.array(_node(G, sym).attributes.position[:2]) + np.array([0.5, 0.0])


@needs_solver
def test_pinning_overrides_the_nearer_robot():
    G = spark_dsg.DynamicSceneGraph.load(DSG)
    # Each robot stands at the object the OTHER one is told to visit.
    poses = {"hilbert": _beside(G, "o16"), "euclid": _beside(G, "o7")}
    got = _assignment(
        G, "(and (visited-object-by hilbert o7)(visited-object-by euclid o16))", poses
    )
    assert got["hilbert"] == ["visit-location o7 :: visit by hilbert"]
    assert got["euclid"] == ["visit-location o16 :: visit by euclid"]


@needs_solver
def test_both_robots_visit_the_same_object():
    G = spark_dsg.DynamicSceneGraph.load(DSG)
    poses = {"hilbert": _beside(G, "o16"), "euclid": _beside(G, "o7")}
    got = _assignment(
        G, "(and (visited-object-by hilbert o6)(visited-object-by euclid o6))", poses
    )
    assert got["hilbert"] == ["visit-location o6 :: visit by hilbert"]
    assert got["euclid"] == ["visit-location o6 :: visit by euclid"]


@needs_solver
def test_mixed_goal_leaves_the_unnamed_task_to_the_allocator():
    G = spark_dsg.DynamicSceneGraph.load(DSG)
    poses = {"hilbert": _beside(G, "o16"), "euclid": _beside(G, "o7")}
    got = _assignment(
        G, "(and (visited-object-by euclid o16)(visited-object o7))", poses
    )
    assert "visit-location o16 :: visit by euclid" in got["euclid"]
    assert (
        sum(t == "visit-location o7 :: visit" for ts in got.values() for t in ts) == 1
    )


@needs_solver
def test_pinning_a_robot_without_a_pose_fails_and_says_why():
    G = spark_dsg.DynamicSceneGraph.load(DSG)
    poses = {"hilbert": _beside(G, "o16"), "euclid": None}
    with pytest.raises(
        ValueError, match="assigns work to euclid, but euclid has no pose"
    ):
        _assignment(G, "(visited-object-by euclid o7)", poses)
    with pytest.raises(ValueError, match="hamilton is not a known robot"):
        _assignment(G, "(visited-object-by hamilton o7)", poses)
