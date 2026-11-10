"""GRSTAPS-X replans only what is left of the mission.

Runs the native solver on the plan_repair_graph prior map, so it needs both
(OMNIPLANNER_TEST_DSG, ADT4_GRSTAPS_ROOT) and skips without them.
"""

import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import spark_dsg
from robot_executor_interface.action_descriptions import Follow, Pick, Place

from omniplanner.compile_plan import collect_plans, compile_plan
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
pytestmark = pytest.mark.skipif(
    not (os.path.exists(DSG) and os.path.isdir(SOLVER)),
    reason="needs the prior map and the GRSTAPS-X solver",
)
ROBOT = "hilbert"


def _node(G, sym):
    for layer in (spark_dsg.DsgLayers.OBJECTS, spark_dsg.DsgLayers.MESH_PLACES):
        for node in G.get_layer(layer).nodes:
            if node.id.str(True).lower() == sym:
                return node
    raise KeyError(sym)


def _plan(G, goal, start_xy, state):
    plugin = GrstapsRos(
        GrstapsConfig(
            entry="itags",
            run_mode="native",
            scenario_name="test_world_state",
            robot_species={ROBOT: "spot_arm"},
            inspect_requirement={"ground": 2, "sensor": 2},
        )
    )
    plugin._node = NS(
        robot_adaptors={ROBOT: NS(robot_type="simulated-spot")},
        world_state_for=lambda _goal: state,
    )
    msg = ConstrainedPddlGoalMsg()
    msg.goal.robot_id, msg.goal.pddl_goal = ROBOT, goal
    request = plugin.constrained_pddl_callback(msg, {ROBOT: np.array(start_xy)})
    plans = full_planning_pipeline(request, G)
    actions = collect_plans(compile_plan({ROBOT: NS(name=ROBOT)}, "map", plans))[
        ROBOT
    ].actions
    raw = plans
    while hasattr(raw, "value"):
        raw = raw.value
    return actions, raw


def _kinds(actions):
    return [(type(a).__name__, getattr(a, "object_id", "")) for a in actions]


def test_held_object_is_carried_and_put_down_not_picked_again():
    G = spark_dsg.DynamicSceneGraph.load(DSG)
    o4 = _node(G, "o4")
    robot_xy = np.array(o4.attributes.position[:2]) + np.array([1.0, 0.0])
    o4.attributes.position = np.array([*robot_xy, 0.0])  # follows its robot
    actions, plan = _plan(
        G,
        "(and (object-in-place o4 t598) (visited-object o5))",
        robot_xy,
        WorldState(holding={ROBOT: ["o4"]}),
    )
    kinds = _kinds(actions)
    assert ("Pick", "o4") not in kinds
    assert kinds[0] == ("Follow", "") and kinds[1] == ("Place", "o4")
    np.testing.assert_allclose(
        actions[1].object_point[:2], _node(G, "t598").attributes.position[:2]
    )
    assert [t["name"] for t in plan.tasks] == ["visit-location o5 :: visit"]


def test_finished_relocation_and_visits_are_not_planned_again():
    G = spark_dsg.DynamicSceneGraph.load(DSG)
    _node(G, "o4").attributes.position = np.array(
        _node(G, "t598").attributes.position
    )  # already put down at its destination
    start = _node(G, "t3").attributes.position[:2]
    actions, plan = _plan(
        G,
        "(and (object-in-place o4 t598) (visited-object o5) (visited-object o17))",
        start,
        WorldState(visited={"o5"}),
    )
    assert not any(isinstance(a, (Pick, Place)) for a in actions)
    assert [t["name"] for t in plan.tasks] == ["visit-location o17 :: visit"]


def test_achieved_goal_plans_nothing():
    G = spark_dsg.DynamicSceneGraph.load(DSG)
    start = _node(G, "t3").attributes.position[:2]
    actions, plan = _plan(G, "(visited-object o5)", start, WorldState(visited={"o5"}))
    assert actions == [] and plan.tasks == []


def test_only_a_held_carry_left_needs_no_solver():
    G = spark_dsg.DynamicSceneGraph.load(DSG)
    start = _node(G, "t3").attributes.position[:2]
    actions, plan = _plan(
        G, "(object-in-place o4 t598)", start, WorldState(holding={ROBOT: ["o4"]})
    )
    assert plan.tasks == []
    assert [type(a) for a in actions] == [Follow, Place]
