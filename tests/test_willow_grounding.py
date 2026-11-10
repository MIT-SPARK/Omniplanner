"""Multi-robot PDDL grounding against the goals that failed on willow (Nov 10):
a misspelled robot ('hamiltonm'), a region the map does not have ('r3'), a map
whose regions broke grounding for a goal that named none ('R(0)'), the new
robot-bound placing goal, and goals that add work while robots are busy.

The planning cases run fast-downward on a small synthetic map; they skip
without it.
"""

import shutil
from importlib.resources import files
from types import SimpleNamespace as NS

import dsg_pddl.domains
import dsg_pddl.dsg_pddl_grounding_multirobot  # noqa: F401  registers the grounding
import dsg_pddl.dsg_pddl_planning  # noqa: F401  registers make_plan
import numpy as np
import pytest
import spark_dsg
from dsg_pddl.dsg_pddl_grounding_multirobot import (
    check_goal_symbols,
    predicate_argument_types,
)
from dsg_pddl.pddl_grounding import ConstraintFact, MultiRobotPddlDomain, PddlGoal

from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner.plan_validity import goal_objects
from omniplanner.world_state import (
    BUSY_ROBOTS,
    WorldState,
    kept_objects,
    unavailable_robots,
)

needs_fd = pytest.mark.skipif(
    shutil.which("fast-downward") is None, reason="needs fast-downward"
)
DOMAIN = MultiRobotPddlDomain(
    files(dsg_pddl.domains)
    .joinpath("RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl")
    .read_text()
)


def _map(rooms=(("r", 0),)):
    """t1..t6 in a row 1 m apart, o58 at t2; rooms parent every place."""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    places = []
    for i in range(1, 7):
        a = spark_dsg.PlaceNodeAttributes()
        a.position = np.array([float(i), 0.0, 0.0])
        nid = spark_dsg.NodeSymbol("t", i).value
        G.add_node(spark_dsg.DsgLayers.MESH_PLACES, nid, a)
        if places:
            G.insert_edge(places[-1], nid)
        places.append(nid)
    for prefix, index in rooms:
        a = spark_dsg.RoomNodeAttributes()
        a.position = np.array([3.0, 0.0, 0.0])
        rid = spark_dsg.NodeSymbol(prefix, index).value
        G.add_node(spark_dsg.DsgLayers.ROOMS, rid, a)
    if rooms:
        rid = spark_dsg.NodeSymbol(*rooms[0]).value
        for nid in places:
            G.insert_edge(rid, nid)
    o = spark_dsg.KhronosObjectAttributes()
    o.position = np.array([2.0, 0.1, 0.0])
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 58).value, o)
    return G


# testbot_a starts beside o58, testbot_b at the far end; willow is the base
# station, which names itself as the goal's robot_id but is no robot.
POSES = {"testbot_a": np.array([1.0, 0.0]), "testbot_b": np.array([6.0, 0.0])}


def _plan(G, goal, robots=POSES, constraints=(), world_state=None):
    req = PlanRequest(
        domain=DOMAIN,
        goal=PddlGoal(
            pddl_goal=goal,
            robot_id="willow",
            constraints=list(constraints),
            world_state=world_state,
        ),
        robot_states=dict(robots),
    )
    plan = full_planning_pipeline(req, G)
    names = list(getattr(getattr(plan, "value", plan), "names", []))
    while hasattr(plan, "value"):
        plan = plan.value
    return plan.symbolic_actions, names


# ---------- P9: robot names ----------


def test_a_misspelled_robot_is_named_not_a_bare_key_error():
    with pytest.raises(ValueError) as err:
        _plan(_map(), "(visited-object-by testbot_bm o58)")
    assert "unknown robot 'testbot_bm' (known: testbot_a, testbot_b)" in str(err.value)


def test_a_misspelled_use_robots_is_named_too():
    use = [ConstraintFact(predicate="use-robots", symbols=["testbot_bm"])]
    with pytest.raises(ValueError, match="unknown robot 'testbot_bm'"):
        _plan(_map(), "(visited-place t3)", constraints=use)


def test_robot_arguments_are_found_by_the_domain_types():
    types = predicate_argument_types(DOMAIN)
    assert types["visited-object-by"] == ["robot", "dsg_object"]
    assert types["object-placed-by"] == ["robot", "dsg_object", "place"]


# ---------- P12: regions ----------


def test_a_region_absent_from_the_map_says_which_regions_exist():
    syms = [NS(symbol="r0", layer="region"), NS(symbol="t1", layer="place")]
    with pytest.raises(ValueError) as err:
        check_goal_symbols(
            "(visited-region r3)", predicate_argument_types(DOMAIN), POSES, syms
        )
    text = str(err.value)
    assert "r3" in text and "regions in it: r0" in text
    assert "could not be read" in text


def test_quantified_variables_are_not_symbols():
    check_goal_symbols(
        "(exists (?p - place) (visited-place ?p))",
        predicate_argument_types(DOMAIN),
        POSES,
        [],
    )


@needs_fd
def test_a_goal_without_regions_plans_on_a_map_of_lowercase_regions():
    # 10:53, multi-robot PDDL: "missing node 'R(0)'" for a goal with no region.
    actions, _ = _plan(_map(rooms=(("r", 0), ("r", 1))), "(visited-place t4)")
    assert actions and actions[-1][-1] == "t4"


@needs_fd
def test_a_room_and_a_region_both_named_r0_do_not_break_grounding():
    actions, _ = _plan(_map(rooms=(("R", 0), ("r", 0))), "(visited-place t4)")
    assert actions


@needs_fd
def test_visiting_a_region_that_exists():
    actions, _ = _plan(_map(), "(visited-region r0)")
    assert actions


# ---------- P15: object-placed-by ----------


def test_the_goal_helpers_read_object_placed_by():
    goal = "(object-placed-by testbot_b o58 t5)"
    assert kept_objects(goal) == {"o58"}
    assert goal_objects(goal)["relocate"] == {"o58"}


@needs_fd
def test_the_named_robot_places_the_object_not_the_nearest():
    actions, _ = _plan(_map(), "(object-placed-by testbot_b o58 t5)")
    placing = [a for a in actions if a[0] in ("pick-object", "place-object")]
    assert [a[1] for a in placing] == ["testbot_b", "testbot_b"]
    assert placing[-1][-1] == "t5"


@needs_fd
def test_an_object_the_robot_already_placed_is_not_moved_again():
    G = _map()
    G.get_node(spark_dsg.NodeSymbol("O", 58).value).attributes.position = np.array(
        [5.0, 0.1, 0.0]
    )
    done = WorldState(placed_by={"testbot_b": {"o58"}})
    actions, _ = _plan(G, "(object-placed-by testbot_b o58 t5)", world_state=done)
    assert actions == []


# ---------- P14: an additional task leaves busy robots alone ----------


def _busy(*robots):
    return [ConstraintFact(predicate=BUSY_ROBOTS, symbols=list(robots))]


def test_without_the_constraint_nobody_is_busy():
    state = WorldState(holding={"testbot_a": ["o58"]})
    assert unavailable_robots([], state, set()) == set()
    assert unavailable_robots(_busy(), state, set()) == {"testbot_a"}
    assert unavailable_robots(_busy("testbot_b"), state, {"testbot_b"}) == {"testbot_a"}


@needs_fd
def test_an_additional_task_is_planned_on_the_free_robots_only():
    actions, names = _plan(_map(), "(visited-place t2)", constraints=_busy("testbot_a"))
    assert names == ["testbot_b"]  # testbot_a gets no plan, not even an empty one
    assert {a[1] for a in actions} == {"testbot_b"}


@needs_fd
def test_naming_a_busy_robot_lets_it_be_interrupted():
    actions, names = _plan(
        _map(), "(visited-place-by testbot_a t2)", constraints=_busy("testbot_a")
    )
    assert "testbot_a" in names


def test_an_additional_task_with_every_robot_busy_says_who():
    with pytest.raises(ValueError) as err:
        _plan(_map(), "(visited-place t2)", constraints=_busy("testbot_a", "testbot_b"))
    assert "busy (testbot_a, testbot_b)" in str(err.value)


# ---------- a Husky has no arm ----------


@needs_fd
def test_a_fleet_pick_and_place_never_uses_a_husky():
    """Found in the local e2e: the planner gave a Husky inspect, pick and place,
    which its Path plan silently drops (it just drives there). testbot_a,
    beside o58, is a Husky here: testbot_b has to do the carrying."""
    husky = WorldState(cannot_manipulate={"testbot_a"})
    actions, _ = _plan(_map(), "(object-in-place o58 t5)", world_state=husky)
    hands = {
        a[1] for a in actions if a[0] in ("inspect", "pick-object", "place-object")
    }
    assert hands == {"testbot_b"}


def test_a_goal_naming_a_husky_to_place_says_it_cannot():
    husky = WorldState(cannot_manipulate={"testbot_a"})
    with pytest.raises(ValueError) as err:
        _plan(_map(), "(object-placed-by testbot_a o58 t5)", world_state=husky)
    assert "testbot_a cannot inspect, pick or place" in str(err.value)
