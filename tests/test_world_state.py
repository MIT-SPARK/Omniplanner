"""A replan plans only what is left: world state in, fast-downward plan out.

Needs the plan_repair_graph prior map (OMNIPLANNER_TEST_DSG) and fast-downward;
skips without them.
"""

import os
from importlib.resources import files

import dsg_pddl.domains
import dsg_pddl.dsg_pddl_grounding_multirobot  # noqa: F401  registers the grounding
import dsg_pddl.dsg_pddl_planning  # noqa: F401  registers make_plan
import numpy as np
import pytest
import spark_dsg
from dsg_pddl.pddl_grounding import MultiRobotPddlDomain, PddlGoal

from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner.world_state import (
    WorldState,
    WorldStateTracker,
    holding_from_dsg,
    kept_objects,
    split_held,
)

DSG = os.path.expanduser(
    os.environ.get(
        "OMNIPLANNER_TEST_DSG",
        "~/colcon_ws/assets/adt4_output/plan_repair_graph/hydra/backend/dsg.json",
    )
)
needs_map = pytest.mark.skipif(not os.path.exists(DSG), reason=f"no map at {DSG}")
ROBOT = "hilbert"


def _map():
    return spark_dsg.DynamicSceneGraph.load(DSG)


def _pos(G, sym):
    for layer in (spark_dsg.DsgLayers.OBJECTS, spark_dsg.DsgLayers.MESH_PLACES):
        for node in G.get_layer(layer).nodes:
            if node.id.str(True).lower() == sym:
                return np.array(node.attributes.position)
    raise KeyError(sym)


def _fd(G, goal, start_xy, world_state=None, constraints=()):
    text = (
        files(dsg_pddl.domains)
        .joinpath("RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl")
        .read_text()
    )
    req = PlanRequest(
        domain=MultiRobotPddlDomain(text),
        goal=PddlGoal(
            pddl_goal=goal,
            robot_id=ROBOT,
            world_state=world_state,
            constraints=list(constraints),
        ),
        robot_states={ROBOT: np.array(start_xy)},
    )
    plan = full_planning_pipeline(req, G)
    while hasattr(plan, "value"):
        plan = plan.value
    return [a[0] for a in plan.symbolic_actions], plan.symbolic_actions


# ---------- pure helpers ----------


def test_kept_objects():
    assert kept_objects("(and (object-in-place o4 t598) (visited-object o5))") == {"o4"}
    assert kept_objects("(and (holding hilbert o4)(at-place hilbert t598))") == {"o4"}
    assert kept_objects("(visited-object o5)") == set()


def test_split_held_releases_what_the_goal_dropped():
    state = WorldState(holding={"hilbert": ["o4", "o8"]})
    planning, release = split_held(state, {"o4"})
    assert planning.holding == {"hilbert": ["o4"]}
    assert release == {"hilbert": ["o8"]}


@needs_map
def test_tracker_from_action_reports():
    G = _map()
    t = WorldStateTracker(visited_radius_m=0.5)
    path = [_pos(G, "t3")[:2], _pos(G, "o2")[:2]]
    assert {"t3", "o2"} <= t.record("FOLLOW", True, ROBOT, points=path, dsg=G)
    assert t.record("FOLLOW", False, ROBOT, points=[_pos(G, "o5")[:2]], dsg=G) == set()
    # o6 is 2.2 m from its nearest place: reaching that place still visits it.
    places = [n for n in G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes]
    o6 = _pos(G, "o6")[:2]
    stop = min(
        places, key=lambda n: np.linalg.norm(np.array(n.attributes.position[:2]) - o6)
    )
    assert "o6" in t.record(
        "FOLLOW", True, ROBOT, points=[stop.attributes.position[:2]], dsg=G
    )
    t.record("GAZE", True, ROBOT, object_id="O4")
    t.record("PICK", True, ROBOT, object_id="O4")
    s = t.snapshot(G)  # the JSON map tracks no holding: reports stand in
    assert "o4" in s.inspected
    assert s.holding == {ROBOT: ["o4"]}
    t.record("PLACE", True, ROBOT, object_id="O4")
    t.reset()
    s = t.snapshot(G)
    assert s.holding == {} and not s.visited and not s.inspected


@needs_map
def test_map_holding_overrides_reports():
    G = _map()
    G.metadata.add({"heracles": {"map_version": 3, "holding": {"hilbert": ["O8"]}}})
    assert holding_from_dsg(G) == {"hilbert": ["o8"]}
    t = WorldStateTracker()
    t.record("PICK", True, ROBOT, object_id="O4")
    assert t.snapshot(G).holding == {"hilbert": ["o8"]}


# ---------- fast-downward plans from the world state ----------


@needs_map
def test_held_object_is_carried_not_picked_again():
    G = _map()
    robot_xy = _pos(G, "o4")[:2] + np.array([1.0, 0.0])
    # The holding rule moves a held object onto its robot.
    G.get_node(spark_dsg.NodeSymbol("O", 4)).attributes.position = np.array(
        [*robot_xy, 0.0]
    )
    goal = "(object-in-place o4 t598)"
    fresh, _ = _fd(G, goal, robot_xy)
    assert "pick-object" in fresh and "inspect" in fresh  # the bug, without state
    held, actions = _fd(G, goal, robot_xy, WorldState(holding={ROBOT: ["o4"]}))
    assert "pick-object" not in held and "inspect" not in held
    assert actions[-1][0] == "place-object" and actions[-1][-1] == "t598"


@needs_map
def test_plan_starting_with_the_object_in_hand_compiles():
    # Live, this raised: Place took its class from a pick earlier in the plan.
    from robot_executor_interface.action_descriptions import Pick, Place

    from omniplanner.omniplanner import SymbolicContext
    from omniplanner_ros.pddl_planner_ros import compile_pddl_plan

    G = _map()
    robot_xy = _pos(G, "o4")[:2] + np.array([1.0, 0.0])
    text = (
        files(dsg_pddl.domains)
        .joinpath("RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl")
        .read_text()
    )
    req = PlanRequest(
        domain=MultiRobotPddlDomain(text),
        goal=PddlGoal(
            pddl_goal="(object-in-place o4 t598)",
            robot_id=ROBOT,
            world_state=WorldState(holding={ROBOT: ["o4"]}),
        ),
        robot_states={ROBOT: robot_xy},
    )
    plan = full_planning_pipeline(req, G)
    while hasattr(plan, "value"):
        plan = plan.value
    # project the robot argument away, as the multi-robot compile path does
    plan.symbolic_actions = [
        (a[0],) + tuple(a[2:]) if a[0] != "inspect" else a
        for a in plan.symbolic_actions
    ]
    seq = compile_pddl_plan(SymbolicContext({}, plan), "p", ROBOT, "map")
    assert not any(isinstance(a, Pick) for a in seq.actions)
    assert isinstance(seq.actions[-1], Place) and seq.actions[-1].object_id == "o4"


@needs_map
def test_inspected_object_is_not_inspected_again():
    G = _map()
    start = _pos(G, "t3")[:2]
    acts, _ = _fd(G, "(object-in-place o4 t598)", start, WorldState(inspected={"o4"}))
    assert "inspect" not in acts and "pick-object" in acts


@needs_map
def test_visited_targets_are_dropped():
    G = _map()
    start = _pos(G, "t3")[:2]
    goal = "(and (visited-object o2) (visited-object o16))"
    _, actions = _fd(G, goal, start, WorldState(visited={"o2"}))
    targets = {a[-1] for a in actions}
    assert "o16" in targets and "o2" not in targets


@needs_map
def test_goal_already_met_plans_nothing():
    G = _map()
    start = _pos(G, "t3")[:2]
    acts, _ = _fd(G, "(visited-object o2)", start, WorldState(visited={"o2"}))
    assert acts == []


# ---------- the repair node puts down what the new goal dropped ----------


def _repair_node_stub(holding, dsg=None):
    import threading
    from types import SimpleNamespace as NS

    tracker = WorldStateTracker()
    for robot, objects in holding.items():
        for obj in objects:
            tracker.record("PICK", True, robot, object_id=obj)
    return NS(
        _current_dsg=lambda: dsg,
        _world_lock=threading.Lock(),
        world_tracker=tracker,
        _pending_release={},
        robot_adaptors={"hilbert": None},
        dsg_frame="map",
        get_logger=lambda: NS(info=lambda *_: None, error=lambda *_: None),
    )


def test_dropped_object_is_put_down_first():
    from robot_executor_interface.action_descriptions import (
        ActionSequence,
        Follow,
        Place,
    )

    from omniplanner_ros.omniplanner_repair_node import OmniPlannerRepairRos as N

    node = _repair_node_stub({"hilbert": ["o8"]})
    state = N.world_state_for(node, "(visited-object o5)")
    assert state.holding == {}  # planned with an empty hand
    plans = {
        "hilbert": ActionSequence("p", "hilbert", [Follow("map", np.zeros((2, 2)))])
    }
    out = N.finalize_plans(node, plans, {"hilbert": np.array([1.0, 2.0, 0.1, 0.0])})
    first = out["hilbert"].actions[0]
    assert isinstance(first, Place) and first.object_id == "o8"
    np.testing.assert_allclose(first.object_point, [1.0, 2.0, 0.1])
    assert isinstance(out["hilbert"].actions[1], Follow)
    # consumed: the next plan does not put it down again
    assert N.finalize_plans(node, plans, {"hilbert": np.zeros(4)}) is plans


def test_object_the_goal_still_wants_stays_held():
    from omniplanner_ros.omniplanner_repair_node import OmniPlannerRepairRos as N

    node = _repair_node_stub({"hilbert": ["o4"]})
    state = N.world_state_for(
        node, "(and (object-in-place o4 t598) (visited-object o5))"
    )
    assert state.holding == {"hilbert": ["o4"]}
    assert node._pending_release == {}


@needs_map
def test_compiled_pick_and_place_carry_the_object_class():
    # The real grasp finds the object by class; an empty one fails on hardware
    # (the simulated grasp ignores it, so sim never showed this).
    from types import SimpleNamespace as NS

    import omniplanner_ros.pddl_planner_ros  # noqa: F401  registers compile_plan
    from omniplanner.compile_plan import collect_plans, compile_plan

    G = _map()
    text = (
        files(dsg_pddl.domains)
        .joinpath("RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl")
        .read_text()
    )
    req = PlanRequest(
        domain=MultiRobotPddlDomain(text),
        goal=PddlGoal(pddl_goal="(object-in-place o4 t598)", robot_id=ROBOT),
        robot_states={ROBOT: _pos(G, "t3")[:2], "euclid": _pos(G, "o5")[:2]},
    )
    plans = full_planning_pipeline(req, G)
    adaptors = {r: NS(name=r) for r in (ROBOT, "euclid")}
    acts = [
        a
        for seq in collect_plans(compile_plan(adaptors, "map", plans)).values()
        for a in seq.actions
        if type(a).__name__ in ("Pick", "Place")
    ]
    assert len(acts) == 2 and all(a.object_class == "trash" for a in acts)
