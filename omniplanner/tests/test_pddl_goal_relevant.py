"""Regression tests for goal-relevant PDDL grounding (scene_scope).

These verify that:
  * dynamic, goal-based pruning produces a much smaller PDDL problem than the
    legacy "full" scope,
  * the pruned problem still yields a valid plan that achieves the goal, and
  * the legacy "full" scope still works.

Runnable both under pytest and directly:
    PYTHONPATH=omniplanner/src python omniplanner/tests/test_pddl_goal_relevant.py
"""

import os
import sys

import numpy as np
import spark_dsg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from utils import build_test_dsg  # noqa: E402

from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal  # noqa: E402
from dsg_pddl.dsg_pddl_grounding import (  # noqa: E402
    generate_place_containment,
    ground_problem,
)
from dsg_pddl.dsg_pddl_planning import make_plan  # noqa: E402
from importlib.resources import as_file, files  # noqa: E402
import dsg_pddl.domains  # noqa: E402


def _load_domain(name, scene_scope):
    with as_file(files(dsg_pddl.domains).joinpath(name)) as p:
        domain = PddlDomain(open(p).read())
    domain.scene_scope = scene_scope
    return domain


def _ground(domain, goal_str):
    G = build_test_dsg()
    goal = PddlGoal(robot_id="euclid", pddl_goal=goal_str)
    grounded = ground_problem(domain, G, {"euclid": np.array([0.0, 0.0])}, goal)
    return G, grounded.value


def _plan_action_names(G, gp):
    plan = make_plan(gp, G)
    return [a[0] for a in plan.symbolic_actions]


def test_goal_relevant_is_smaller_than_full():
    goal = "(and (visited-object o0))"
    _, gp_rel = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"), goal
    )
    _, gp_full = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "full"), goal
    )
    # The pruned problem references only a couple of symbols; full encodes them all.
    assert len(gp_rel.symbols) < len(gp_full.symbols)
    assert "o0" in gp_rel.symbols
    assert "pstart" in gp_rel.symbols


def test_goal_relevant_goto_plan_visits_object():
    G, gp = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"),
        "(and (visited-object o0))",
    )
    plan = make_plan(gp, G)
    gotos = [a for a in plan.symbolic_actions if a[0] == "goto-poi"]
    assert gotos, "expected at least one goto-poi action"
    # the object should be the final POI we navigate to
    assert gotos[-1][-1] == "o0"


def test_goal_relevant_pick_place_plan():
    G, gp = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"),
        "(and (object-in-place o0 p1))",
    )
    actions = _plan_action_names(G, gp)
    assert "pick-object" in actions
    assert "place-object" in actions


def test_offline_targeted_robot_raises_clear_error():
    # A None pose is what a lazy/offline robot lookup yields; planning *for* such
    # a robot should fail with a clear message rather than a cryptic TypeError.
    domain = _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant")
    G = build_test_dsg()
    goal = PddlGoal(robot_id="euclid", pddl_goal="(and (visited-object o0))")
    try:
        ground_problem(domain, G, {"euclid": None}, goal)
        assert False, "expected RuntimeError for offline targeted robot"
    except RuntimeError as e:
        assert "no transform" in str(e)


def test_full_scope_still_plans():
    G, gp = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "full"),
        "(and (visited-object o0))",
    )
    actions = _plan_action_names(G, gp)
    assert "goto-poi" in actions


def test_quantified_goal_keeps_unnamed_objects():
    goal = "(forall (?o - dsg_object) (visited-object ?o))"
    G, relevant = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"), goal
    )
    _, full = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "full"), goal
    )
    assert relevant.symbols.keys() == full.symbols.keys()
    plan = make_plan(relevant, G)
    visited = {a[-1] for a in plan.symbolic_actions if a[0] == "goto-poi"}
    assert {"o0", "o1"} <= visited


def test_goal_relevant_object_in_region_plan():
    # o0's current place is a member of r0, so target the OTHER region r1:
    # the plan must pick o0 and place it at a place-in-region of r1.
    G, gp = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"),
        "(and (object-in-region o0 r1))",
    )
    actions = _plan_action_names(G, gp)
    assert "pick-object" in actions
    assert "place-object" in actions


def test_object_in_region_already_satisfied_yields_empty_plan():
    # Degenerate case: if the object's current place is
    # already in the target region, the goal is initially true -> empty plan.
    G, gp = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"),
        "(and (object-in-region o0 r0))",
    )
    plan = make_plan(gp, G)
    assert plan.symbolic_actions == []


def _build_real_graph_shape_dsg():
    """A DSG mirroring the real hydra/camp-graph shape rather than the toy
    ``build_test_dsg`` fixture: ROOMS parent MESH_PLACES-layer places
    directly (region_injector's actual membership edge), with MESH_PLACES
    registered at a layer id BELOW ROOMS (like a real graph) so insert_edge
    actually records parent/child (see generate_place_containment's docstring
    and the task-D1 report for why layer id 20 -- as used by build_test_dsg --
    can never do this).

    Mesh-place indices (7, 8, 9, 10) are deliberately disjoint from any 3D
    PLACES-layer indices -- indeed no 3D PLACES nodes are added at all -- so
    this graph cannot be satisfied by accidentally reusing PLACES-layer
    containment (the pre-fix, PLACES-only code path) or by index collision
    (p7/p8 do not exist anywhere).
    """
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, "O", spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(3, "P", spark_dsg.DsgLayers.MESH_PLACES)
    G.add_layer(4, "R", spark_dsg.DsgLayers.ROOMS)

    room0 = spark_dsg.RoomNodeAttributes()
    room0.position = np.array([0, 0, 0])
    room0.semantic_label = 0
    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("R", 0).value, room0)

    room1 = spark_dsg.RoomNodeAttributes()
    room1.position = np.array([10, 0, 0])
    room1.semantic_label = 1
    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("R", 1).value, room1)

    def add_mesh_place(idx, pos):
        p = spark_dsg.PlaceNodeAttributes()
        p.position = np.array(pos)
        p.semantic_label = 4  # ground
        G.add_node(
            spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("P", idx).value, p
        )

    add_mesh_place(7, [0, 0, 0])
    add_mesh_place(8, [1, 0, 0])
    add_mesh_place(9, [9, 0, 0])
    add_mesh_place(10, [10, 0, 0])

    obj0 = spark_dsg.ObjectNodeAttributes()
    obj0.position = np.array([0.1, 0, 0])
    obj0.semantic_label = 34  # box
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 0).value, obj0)

    # Rooms parent MESH_PLACES places directly (no 3D PLACES layer involved).
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("P", 7).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("P", 8).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 1).value, spark_dsg.NodeSymbol("P", 9).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 1).value, spark_dsg.NodeSymbol("P", 10).value
    )

    # Places graph so LayerPlanner/goto-poi routing has a path to walk.
    G.insert_edge(
        spark_dsg.NodeSymbol("P", 7).value, spark_dsg.NodeSymbol("P", 8).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("P", 8).value, spark_dsg.NodeSymbol("P", 9).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("P", 9).value, spark_dsg.NodeSymbol("P", 10).value
    )

    G.insert_edge(
        spark_dsg.NodeSymbol("P", 7).value, spark_dsg.NodeSymbol("O", 0).value
    )

    return G


def test_generate_place_containment_real_graph_shape():
    # Direct unit test of generate_place_containment against the real-graph
    # shape (rooms parent MESH_PLACES directly, no 3D PLACES layer at all).
    # This is exactly what the PLACES-only (post-652b1f3, pre-fix) code
    # returned [] for -- G.get_layer(PLACES) is present-but-empty on this
    # graph, so a PLACES-only walk finds no place nodes and thus no facts.
    G = _build_real_graph_shape_dsg()
    facts = generate_place_containment(G)
    fact_set = set(facts)
    assert ("place-in-region", "p7", "r0") in fact_set
    assert ("place-in-region", "p8", "r0") in fact_set
    assert ("place-in-region", "p9", "r1") in fact_set
    assert ("place-in-region", "p10", "r1") in fact_set


def _build_real_graph_shape_dsg_with_places_collision():
    """Real-graph shape (as in ``_build_real_graph_shape_dsg``) PLUS a 3D
    PLACES layer that rooms also parent, with an index collision against
    MESH_PLACES: 3D place "p7" (room R1's child) collides post-
    normalize_symbol with mesh place "P7" (room R0's child, from path (a)).

    This is the shape path (b) was wrongly emitting facts for even when path
    (a) already had the authoritative answer: if (b) ran unconditionally, it
    would add ("place-in-region", "p7", "r1") -- aliasing R0's real mesh
    place P7 into R1 -- alongside (a)'s correct ("place-in-region", "p7",
    "r0"). generate_place_containment must return ONLY the path (a) facts.
    """
    G = _build_real_graph_shape_dsg()

    G.add_layer(5, "p", spark_dsg.DsgLayers.PLACES)

    def add_3d_place(idx, pos):
        p = spark_dsg.PlaceNodeAttributes()
        p.position = np.array(pos)
        G.add_node(spark_dsg.DsgLayers.PLACES, spark_dsg.NodeSymbol("p", idx).value, p)

    # Colliding index: 3D place p7 vs. mesh place P7 (both normalize to "p7").
    add_3d_place(7, [0, 5, 0])
    add_3d_place(9, [9, 5, 0])

    # Rooms also parent the 3D PLACES nodes (the toy-fixture-style edge),
    # deliberately crossing rooms so the alias would be detectable: p7 is
    # parented under R1, not R0 (where mesh P7 actually lives).
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 1).value, spark_dsg.NodeSymbol("p", 7).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("p", 9).value
    )

    return G


def test_generate_place_containment_path_b_is_fallback_only():
    # When path (a) (MESH_PLACES-parents) yields facts, path (b) (ROOMS
    # children resolving to place-type nodes) must NOT also run -- it is a
    # fallback for graphs where (a) is empty, not a union. Otherwise
    # normalize_symbol's p<i>/P<i> case-collapse aliases unrelated symbols.
    G = _build_real_graph_shape_dsg_with_places_collision()
    facts = generate_place_containment(G)
    fact_set = set(facts)

    # Only the path (a) (MESH_PLACES-derived) facts should be present.
    assert fact_set == {
        ("place-in-region", "p7", "r0"),
        ("place-in-region", "p8", "r0"),
        ("place-in-region", "p9", "r1"),
        ("place-in-region", "p10", "r1"),
    }

    # In particular, no aliased/duplicate fact from the colliding 3D place
    # nodes (which would incorrectly claim p7 is in r1, or p9 is in r0).
    assert ("place-in-region", "p7", "r1") not in fact_set
    assert ("place-in-region", "p9", "r0") not in fact_set


def test_goal_relevant_object_in_region_plan_real_graph_shape():
    # End-to-end regression: object-in-region must ground and plan on a DSG
    # shaped like a real hydra/camp graph (rooms parenting MESH_PLACES
    # directly), which the toy build_test_dsg fixture cannot exercise/fake.
    G = _build_real_graph_shape_dsg()
    domain = _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant")
    goal = PddlGoal(robot_id="euclid", pddl_goal="(and (object-in-region o0 r1))")
    grounded = ground_problem(domain, G, {"euclid": np.array([0.0, 0.0])}, goal)
    gp = grounded.value
    actions = _plan_action_names(G, gp)
    assert "pick-object" in actions
    assert "place-object" in actions


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}: {e!r}")
    sys.exit(1 if failures else 0)
