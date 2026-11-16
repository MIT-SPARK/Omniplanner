"""Region goals on a map named the way Hydra names it: mesh places T(i) under rooms R(i).

The goal-relevant grounding picks each named region's places by looking them up
in the places graph. Looking ``t12`` up as ``t(12)`` instead of ``T(12)`` found
none of them, so every region was empty: visits and in-region goals came back
"unreachable ... the place graph is disconnected", and explored-region goals were
"already satisfied" with nothing driven.

Needs fast-downward on PATH (as the other planning tests do).
"""

from importlib.resources import as_file, files

import dsg_pddl.domains
import numpy as np
import pytest
import spark_dsg
from dsg_pddl.dsg_pddl_grounding_multirobot import ground_problem
from dsg_pddl.dsg_pddl_planning import make_plan
from dsg_pddl.grounding_errors import EmptyRegionError
from dsg_pddl.pddl_grounding import ConstraintFact, MultiRobotPddlDomain, PddlGoal

DOMAIN = "RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl"
NX, NY = 10, 4  # grid of places, 1 m apart
REGION_X = range(6, NX)  # R(3) = the places with x >= 6, away from both starts
ROBOTS = {"euclid": np.array([0.0, 0.0]), "gauss": np.array([0.0, 3.0])}


def _t(x, y):
    return spark_dsg.NodeSymbol("T", y * NX + x).value


def hydra_dsg():
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    for y in range(NY):
        for x in range(NX):
            a = spark_dsg.PlaceNodeAttributes()
            a.position = np.array([x, y, 0.0])
            G.add_node(spark_dsg.DsgLayers.MESH_PLACES, _t(x, y), a)
    for y in range(NY):
        for x in range(NX):
            if x + 1 < NX:
                G.insert_edge(_t(x, y), _t(x + 1, y))
            if y + 1 < NY:
                G.insert_edge(_t(x, y), _t(x, y + 1))
    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([7.5, 1.5, 0.0])
    region = spark_dsg.NodeSymbol("R", 3).value
    G.add_node(spark_dsg.DsgLayers.ROOMS, region, room)
    for y in range(NY):
        for x in REGION_X:
            G.insert_edge(region, _t(x, y))
    return G


MEMBERS = {f"t{y * NX + x}" for y in range(NY) for x in REGION_X}


def _plan(goal, scope="goal_relevant"):
    with as_file(files(dsg_pddl.domains).joinpath(DOMAIN)) as p:
        domain = MultiRobotPddlDomain(open(p).read())
    domain.scene_scope = scope
    G = hydra_dsg()
    g = PddlGoal(pddl_goal=goal, robot_id="euclid", constraints=[])
    return make_plan(ground_problem(domain, G, ROBOTS, g).value, G)


def _visits(plan, robot=None):
    return {
        a[3]
        for a in plan.symbolic_actions
        if a[0] == "goto-poi" and robot in (None, a[1])
    }


def _last_place(plan, robot):
    return [a for a in plan.symbolic_actions if a[0] == "goto-poi" and a[1] == robot][
        -1
    ][3]


def test_the_region_gets_its_places():
    with as_file(files(dsg_pddl.domains).joinpath(DOMAIN)) as p:
        domain = MultiRobotPddlDomain(open(p).read())
    domain.scene_scope = "goal_relevant"
    g = PddlGoal(pddl_goal="(visited-region r3)", robot_id="euclid", constraints=[])
    problem = ground_problem(domain, hydra_dsg(), ROBOTS, g).value.problem_str
    assert "(place-in-region" in problem


@pytest.mark.parametrize("scope", ["goal_relevant", "full"])
def test_visit_a_region_by_a_named_robot(scope):
    plan = _plan("(visited-region-by euclid r3)", scope)
    assert _visits(plan, "euclid") & MEMBERS


@pytest.mark.parametrize("scope", ["goal_relevant", "full"])
def test_end_in_a_region(scope):
    plan = _plan("(and (in-region euclid r3) (in-region gauss r3))", scope)
    assert _last_place(plan, "euclid") in MEMBERS
    assert _last_place(plan, "gauss") in MEMBERS


@pytest.mark.parametrize("scope", ["goal_relevant", "full"])
def test_explore_a_region_visits_every_place(scope):
    plan = _plan("(explored-region r3)", scope)
    assert MEMBERS <= _visits(plan)


# ---------------------------------------------------------------------------
# a region the robots cannot use is reported as such, before Fast Downward
# ---------------------------------------------------------------------------


def _ground(G, goal, constraints=()):
    with as_file(files(dsg_pddl.domains).joinpath(DOMAIN)) as p:
        domain = MultiRobotPddlDomain(open(p).read())
    domain.scene_scope = "goal_relevant"
    g = PddlGoal(pddl_goal=goal, robot_id="euclid", constraints=list(constraints))
    return ground_problem(domain, G, ROBOTS, g).value


def _with_empty_room(G):
    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([3.0, 1.0, 0.0])
    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("R", 4).value, room)
    return G


def _with_island_region(G):
    """R(5): two places joined to each other but not to the grid."""
    region = spark_dsg.NodeSymbol("R", 5).value
    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([20.5, 0.0, 0.0])
    G.add_node(spark_dsg.DsgLayers.ROOMS, region, room)
    for i, x in enumerate((20.0, 21.0)):
        a = spark_dsg.PlaceNodeAttributes()
        a.position = np.array([x, 0.0, 0.0])
        sym = spark_dsg.NodeSymbol("T", 900 + i).value
        G.add_node(spark_dsg.DsgLayers.MESH_PLACES, sym, a)
        G.insert_edge(region, sym)
    G.insert_edge(
        spark_dsg.NodeSymbol("T", 900).value, spark_dsg.NodeSymbol("T", 901).value
    )
    return G


@pytest.mark.parametrize(
    "goal",
    ["(visited-region-by euclid r4)", "(in-region gauss r4)", "(explored-region r4)"],
)
def test_a_region_without_places_says_so(goal):
    # explored-region would otherwise be vacuously true: an empty "already satisfied" plan
    with pytest.raises(EmptyRegionError, match="r4 has no places"):
        _ground(_with_empty_room(hydra_dsg()), goal)


@pytest.mark.parametrize("goal", ["(visited-region r5)", "(explored-region r5)"])
def test_a_region_off_the_robots_graph_says_so(goal):
    with pytest.raises(EmptyRegionError, match="none of its 2 places .* disconnected"):
        _ground(_with_island_region(hydra_dsg()), goal)


def test_a_region_behind_the_constraints_says_so(monkeypatch):
    # forbidding the region's centre with a radius that covers all of it
    monkeypatch.setenv("ADT4_FORBIDDEN_RADIUS_M", "5")
    forbid = ConstraintFact(predicate="forbidden-poi", symbols=["t17"])
    with pytest.raises(EmptyRegionError, match="once the constraints are applied"):
        _ground(hydra_dsg(), "(visited-region r3)", constraints=[forbid])


def test_only_the_unusable_region_is_named():
    G = _with_empty_room(hydra_dsg())
    with pytest.raises(EmptyRegionError) as e:
        _ground(G, "(and (visited-region r3) (visited-region r4))")
    assert set(e.value.regions) == {"r4"}
