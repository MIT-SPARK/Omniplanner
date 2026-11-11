"""Island mesh-places must not disconnect goal-relevant PDDL problems.

Motion-tier benchmark v1 (floor3 seed 71, NL arm): the discovered object's
position snapped to a mesh-place in a disconnected component of the saved DSG
(saved graphs contain island places -- see the placement-connectivity work),
so every POI pair touching the object came back inf from the distance matrix
and was silently dropped, producing a two-component problem Fast Downward
correctly called unsolvable (exit 5) for a physically reachable object.

Nearest-place snapping must be restricted to the component reachable from the
robot start.

Runnable both under pytest and directly:
    PYTHONPATH=omniplanner/src python omniplanner/tests/test_pddl_island_places.py
"""
import os
import sys

import numpy as np
import spark_dsg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal  # noqa: E402
from dsg_pddl.dsg_pddl_grounding import ground_problem  # noqa: E402
from importlib.resources import as_file, files  # noqa: E402
import dsg_pddl.domains  # noqa: E402


def build_island_dsg():
    """Two connected mesh-places near the start + one ISLAND mesh-place with
    the object right next to it."""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, "O", spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(20, "P", spark_dsg.DsgLayers.MESH_PLACES)

    def add_place(idx, x):
        a = spark_dsg.PlaceNodeAttributes()
        a.position = np.array([x, 0.0, 0.0])
        a.semantic_label = 4  # ground
        G.add_node(spark_dsg.DsgLayers.MESH_PLACES,
                   spark_dsg.NodeSymbol("P", idx).value, a)

    add_place(0, -1.0)   # main component (robot starts at the origin)
    add_place(1, 1.0)    # main component
    add_place(2, 6.0)    # ISLAND: no edges at all

    obj = spark_dsg.ObjectNodeAttributes()
    obj.position = np.array([6.2, 0.0, 0.0])  # nearest place is the island
    obj.semantic_label = 34
    G.add_node(spark_dsg.DsgLayers.OBJECTS,
               spark_dsg.NodeSymbol("O", 0).value, obj)

    G.insert_edge(spark_dsg.NodeSymbol("P", 0).value,
                  spark_dsg.NodeSymbol("P", 1).value)
    G.insert_edge(spark_dsg.NodeSymbol("P", 2).value,
                  spark_dsg.NodeSymbol("O", 0).value)
    return G


def _ground(goal_str):
    with as_file(files(dsg_pddl.domains)
                 .joinpath("RegionObjectRearrangementDomain.pddl")) as p:
        domain = PddlDomain(open(p).read())
    domain.scene_scope = "goal_relevant"
    G = build_island_dsg()
    goal = PddlGoal(robot_id="euclid", pddl_goal=goal_str)
    grounded = ground_problem(domain, G, {"euclid": np.array([0.0, 0.0])}, goal)
    return grounded.value


def test_object_current_place_is_in_start_component():
    gp = _ground("(and (visited-object o0))")
    assert "(object-in-place o0 p1)" in gp.problem_str, gp.problem_str
    assert "p2" not in gp.symbols, (
        "the island place must not be selected as the object's current place")


def test_object_is_connected_to_start():
    gp = _ground("(and (visited-object o0))")
    assert ("(connected pstart o0)" in gp.problem_str
            or "(connected o0 pstart)" in gp.problem_str), (
        "dropping the object's connectivity produces an unsolvable "
        "two-component problem:\n" + gp.problem_str)


if __name__ == "__main__":
    test_object_current_place_is_in_start_component()
    test_object_is_connected_to_start()
    print("ok")


def test_make_plan_expands_waypoints_despite_island():
    # motion-tier v1 nl_s71 re-run (2026-08-05): grounding produced a solvable
    # problem (the component-restricted fix), gate 1 verified, then plan
    # COMPILATION crashed with NetworkXNoPath -- make_plan's own LayerPlanner
    # was unrestricted, so get_external_path snapped a goto-poi endpoint to an
    # island node. Same bug, one layer down.
    from dsg_pddl.dsg_pddl_planning import make_plan

    with as_file(files(dsg_pddl.domains)
                 .joinpath("RegionObjectRearrangementDomain.pddl")) as p:
        domain = PddlDomain(open(p).read())
    domain.scene_scope = "goal_relevant"
    G = build_island_dsg()
    goal = PddlGoal(robot_id="euclid", pddl_goal="(and (visited-object o0))")
    grounded = ground_problem(domain, G, {"euclid": np.array([0.0, 0.0])}, goal)
    plan = make_plan(grounded.value, G)   # must not raise NetworkXNoPath
    gotos = [a for a in plan.symbolic_actions if a[0] == "goto-poi"]
    assert gotos, "expected at least one goto-poi action"


# ---------------------------------------------------------------------------
# regions whose member places include an island
# ---------------------------------------------------------------------------


def build_island_region_dsg():
    """A room containing three places, one of which is an ISLAND mesh-place.

    ``generate_place_containment`` reports every member of the room, but
    ``place_sym_to_pos`` is built only from the component reachable from the
    robot start, so ranking the members by distance to the region centroid
    used to raise ``KeyError: 'p2'`` and kill grounding outright.
    """
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, "O", spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(3, "p", spark_dsg.DsgLayers.PLACES)
    G.add_layer(4, "R", spark_dsg.DsgLayers.ROOMS)
    G.add_layer(20, "P", spark_dsg.DsgLayers.MESH_PLACES)

    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([0.0, 0.0, 0.0])
    room.semantic_label = 0
    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("R", 0).value, room)

    def add_place(idx, x):
        # 3D place (what the room parents in this fixture shape) ...
        a3 = spark_dsg.PlaceNodeAttributes()
        a3.position = np.array([x, 0.0, 0.0])
        G.add_node(
            spark_dsg.DsgLayers.PLACES, spark_dsg.NodeSymbol("p", idx).value, a3
        )
        # ... and the mesh-place of the same index, which is what every other
        # place-symbol source in the grounder is keyed off.
        a2 = spark_dsg.PlaceNodeAttributes()
        a2.position = np.array([x, 0.0, 0.0])
        a2.semantic_label = 4  # ground
        G.add_node(
            spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("P", idx).value, a2
        )
        G.insert_edge(
            spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("p", idx).value
        )

    add_place(0, -1.0)  # main component (robot starts at the origin)
    add_place(1, 1.0)  # main component
    add_place(2, 6.0)  # ISLAND: no mesh-place edges at all

    G.insert_edge(
        spark_dsg.NodeSymbol("P", 0).value, spark_dsg.NodeSymbol("P", 1).value
    )
    return G


def _ground_region_goal(goal_str):
    with as_file(
        files(dsg_pddl.domains).joinpath("RegionObjectRearrangementDomain.pddl")
    ) as p:
        domain = PddlDomain(open(p).read())
    domain.scene_scope = "goal_relevant"
    G = build_island_region_dsg()
    goal = PddlGoal(robot_id="euclid", pddl_goal=goal_str)
    return ground_problem(domain, G, {"euclid": np.array([0.0, 0.0])}, goal).value


def test_region_with_island_member_grounds_without_keyerror():
    gp = _ground_region_goal("(and (visited-region r0))")
    assert "r0" in gp.symbols


def test_region_members_are_restricted_to_reachable_places():
    gp = _ground_region_goal("(and (visited-region r0))")
    assert "p2" not in gp.symbols, (
        "the island place must not be selected as a representative place "
        "for the region:\n" + gp.problem_str
    )
    assert "(place-in-region p2 r0)" not in gp.problem_str, gp.problem_str
    assert (
        "(place-in-region p0 r0)" in gp.problem_str
        or "(place-in-region p1 r0)" in gp.problem_str
    ), gp.problem_str
