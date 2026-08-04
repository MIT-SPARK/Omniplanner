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
