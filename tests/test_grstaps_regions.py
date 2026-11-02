"""GRSTAPS-X region goals: visited-region / explored-region (and -by), plus
operator-drawn polygon areas.

An ITAGS task stands at one vertex, so grounding expands a region into place
visits (expand_regions). None of these tests needs the solver: they check the
expansion and the ITAGS input grounding writes.
"""

import json
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import spark_dsg
from omniplanner.grstaps_planner import (
    GrstapsDomain,
    GrstapsGoal,
    _mesh_place_graph,
    coverage_waypoints,
    expand_regions,
    ground_problem,
    parse_goal_targets,
    region_places,
)
from omniplanner.world_state import WorldState

# A 10 x 4 grid of mesh places 2 m apart (x 0..18, y 0..6). Room r0 holds the
# left half (x <= 8), r1 the right half; o0 is an object near (17, 3).
NX, NY, STEP = 10, 4, 2.0


def _pid(i, j):
    return spark_dsg.NodeSymbol("P", i * NY + j)


@pytest.fixture
def G():
    H = spark_dsg.DynamicSceneGraph()
    H.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    for i in range(NX):
        for j in range(NY):
            a = spark_dsg.PlaceNodeAttributes()
            a.position = np.array([i * STEP, j * STEP, 0.0])
            H.add_node(spark_dsg.DsgLayers.MESH_PLACES, _pid(i, j).value, a)
    for i in range(NX):
        for j in range(NY):
            if i + 1 < NX:
                H.insert_edge(_pid(i, j).value, _pid(i + 1, j).value)
            if j + 1 < NY:
                H.insert_edge(_pid(i, j).value, _pid(i, j + 1).value)
    for r, (lo, hi) in enumerate([(0, 4), (5, 9)]):
        a = spark_dsg.RoomNodeAttributes()
        a.position = np.array([(lo + hi) / 2 * STEP, 3.0, 0.0])
        rid = spark_dsg.NodeSymbol("R", r)
        H.add_node(spark_dsg.DsgLayers.ROOMS, rid.value, a)
        for i in range(lo, hi + 1):
            for j in range(NY):
                H.insert_edge(rid.value, _pid(i, j).value)
    o = spark_dsg.KhronosObjectAttributes()
    o.position = np.array([17.0, 3.0, 0.0])
    H.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 0).value, o)
    return H


def _graph(G):
    graph, vertices, _, symbol_of = _mesh_place_graph(G)
    return graph, vertices, symbol_of


def _start(vertices, x, y):
    return min(vertices, key=lambda v: (v["x"] - x) ** 2 + (v["y"] - y) ** 2)


def _xy(G, sym):
    for n in G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes:
        if n.id.str(True).lower() == sym:
            return tuple(n.attributes.position[:2])
    raise KeyError(sym)


# -- parsing -------------------------------------------------------------------


def test_region_predicates_parse():
    found = parse_goal_targets(
        "(and (visited-region r0)(explored-region r1)(visited-region-by spot1 r1)"
        "(explored-region-by spot2 area1)(visited-object o0))"
    )
    assert found["regions"] == {"visit": ["r0"], "explore": ["r1"]}
    assert found["region_bound"] == {
        "visit": {"r1": ["spot1"]},
        "explore": {"area1": ["spot2"]},
    }
    assert found["visit"] == ["o0"]


def test_negated_region_goal_is_refused():
    with pytest.raises(ValueError, match="negative"):
        parse_goal_targets("(not (visited-region r0))")


def test_unsupported_predicate_error_names_regions():
    with pytest.raises(ValueError, match="visited-region"):
        parse_goal_targets("(at-place spot1 p3)")


# -- members and waypoints -----------------------------------------------------


def test_room_and_polygon_members(G):
    graph, _, symbol_of = _graph(G)
    xy = {
        s.lower(): (v["x"], v["y"])
        for v, s in zip(graph["vertices"], symbol_of.values())
    }
    members, centre = region_places(G, "r0", xy)
    assert len(members) == 5 * NY and all(xy[m][0] <= 8 for m in members)
    assert tuple(centre) == (4.0, 3.0)
    members, _ = region_places(
        G, "area1", xy, {"area1": [[11, -1], [15, -1], [15, 3], [11, 3]]}
    )
    assert sorted(xy[m] for m in members) == [
        (x, y) for x in (12.0, 14.0) for y in (0.0, 2.0)
    ]
    with pytest.raises(ValueError, match="not in the scene graph"):
        region_places(G, "r7", xy)


def test_waypoints_keep_their_spacing_and_cap():
    xy = {f"p{i}": (float(i), 0.0) for i in range(40)}
    pts, spacing = coverage_waypoints(list(xy), xy, (0.0, 0.0), 8.0)
    assert spacing == 8.0 and [xy[p][0] for p in pts] == [0, 8, 16, 24, 32]
    pts, spacing = coverage_waypoints(list(xy), xy, (0.0, 0.0), 4.0, cap=4)
    assert len(pts) <= 4 and spacing > 4.0
    gaps = np.diff(sorted(xy[p][0] for p in pts))
    assert gaps.min() >= spacing - 1e-9


# -- expansion -----------------------------------------------------------------


def test_region_visit_is_one_core_place_near_the_robot(G):
    graph, vertices, symbol_of = _graph(G)
    goal = GrstapsGoal(goal_points=[], robot_id="spot1", region_visits=["r1"])
    out, region_of, places_of = expand_regions(
        goal, G, graph, symbol_of, {"spot1": _start(vertices, 18, 6)}
    )
    (p,) = out.goal_points
    assert places_of == {"r1": [p]}
    assert region_of[p]["region"] == "r1" and region_of[p]["kind"] == "visit"
    x, y = _xy(G, p)
    # In the core (inner half by distance from (14, 3)), on the robot's side.
    assert abs(x - 14) <= 2 and x >= 14 and y >= 2
    assert out.region_visits == []


def test_pinned_region_visit_goes_to_that_robot(G):
    graph, vertices, symbol_of = _graph(G)
    goal = GrstapsGoal(
        goal_points=[], robot_id="spot1", region_bound={"visit": {"r0": ["spot2"]}}
    )
    out, _, _ = expand_regions(
        goal,
        G,
        graph,
        symbol_of,
        {"spot1": _start(vertices, 18, 0), "spot2": _start(vertices, 0, 0)},
    )
    assert out.goal_points == []
    ((p, robots),) = out.robot_bound["visit"].items()
    assert robots == ["spot2"] and _xy(G, p)[0] <= 8


def test_explore_spreads_waypoints_over_the_region(G, monkeypatch):
    monkeypatch.setenv("OMNIPLANNER_GRSTAPS_EXPLORE_SPACING_M", "4")
    graph, vertices, symbol_of = _graph(G)
    goal = GrstapsGoal(goal_points=[], robot_id="spot1", region_explores=["r0"])
    out, region_of, places_of = expand_regions(
        goal, G, graph, symbol_of, {"spot1": _start(vertices, 18, 0)}
    )
    pts = places_of["r0"]
    assert len(pts) >= 4 and set(out.goal_points) == set(pts)
    coords = np.array([_xy(G, p) for p in pts])
    assert coords[:, 0].max() <= 8
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    assert d[np.triu_indices(len(pts), 1)].min() >= 4
    assert {region_of[p]["total"] for p in pts} == {len(pts)}


def test_done_waypoints_and_visits_are_dropped(G, monkeypatch):
    monkeypatch.setenv("OMNIPLANNER_GRSTAPS_EXPLORE_SPACING_M", "4")
    graph, vertices, symbol_of = _graph(G)
    starts = {"spot1": _start(vertices, 18, 0)}
    full, _, places = expand_regions(
        GrstapsGoal(goal_points=[], robot_id="spot1", region_explores=["r0"]),
        G, graph, symbol_of, starts,
    )  # fmt: skip
    seen = places["r0"][0]
    visit, _, vp = expand_regions(
        GrstapsGoal(goal_points=[], robot_id="spot1", region_visits=["r1"]),
        G, graph, symbol_of, starts,
    )  # fmt: skip
    state = WorldState(visited={seen, vp["r1"][0]})
    left, _, lp = expand_regions(
        GrstapsGoal(
            goal_points=[],
            robot_id="spot1",
            region_explores=["r0"],
            region_visits=["r1"],
            world_state=state,
        ),
        G, graph, symbol_of, starts,
    )  # fmt: skip
    assert seen not in left.goal_points
    assert len(lp["r0"]) == len(places["r0"]) - 1
    assert lp["r1"] == []


def test_unreachable_region_is_explained(G):
    graph, vertices, symbol_of = _graph(G)
    # Cut every edge between the halves: r1 is then cut off from a robot in r0.
    graph = dict(graph)
    xs = {v["id"]: v["x"] for v in graph["vertices"]}
    graph["edges"] = [
        e
        for e in graph["edges"]
        if (xs[e["vertex_a"]] <= 8) == (xs[e["vertex_b"]] <= 8)
    ]
    with pytest.raises(ValueError, match="No robot can reach region r1"):
        expand_regions(
            GrstapsGoal(goal_points=[], robot_id="spot1", region_visits=["r1"]),
            G, graph, symbol_of, {"spot1": _start(vertices, 0, 0)},
        )  # fmt: skip


# -- grounding -----------------------------------------------------------------


def _domain(tmp_path, entry="itags"):
    return GrstapsDomain(
        run_mode="native",
        entry=entry,
        work_dir=str(tmp_path),
        scenario_name="regions",
        robot_type_species={"spot": "spot_arm"},
        inspect_requirement={"ground": 1, "sensor": 1},
    )


def _itags_input(tmp_path, robot="spot1"):
    with open(tmp_path / f"regions_{robot}" / "itags_input.json") as f:
        return json.load(f)


def test_grounding_orders_a_whole_region_before_a_target(G, tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIPLANNER_GRSTAPS_EXPLORE_SPACING_M", "4")
    goal = GrstapsGoal(
        goal_points=["o0"],
        robot_id="spot1",
        region_explores=["r0"],
        region_bound={"visit": {"r1": ["spot2"]}},
        robot_types={"spot1": "spot", "spot2": "spot"},
        constraints=[NS(predicate="before", symbols=["r0", "o0"])],
    )
    poses = {"spot1": np.array([0.0, 0.0, 0.0]), "spot2": np.array([18.0, 6.0, 0.0])}
    ground_problem(_domain(tmp_path), G, poses, goal)
    data = _itags_input(tmp_path)
    names = [t["name"] for t in data["tasks"]]
    o0 = names.index("visit-location o0 :: visit")
    explore = [
        i
        for i, n in enumerate(names)
        if n.startswith("visit-location p") and "by" not in n
    ]
    pinned = [n for n in names if n.endswith("by spot2")]
    assert len(explore) >= 4 and len(pinned) == 1
    assert sorted(a for a, b in data["precedence_constraints"] if b == o0) == explore


def test_pddl_entry_refuses_region_goals(G, tmp_path):
    goal = GrstapsGoal(goal_points=[], robot_id="spot1", region_visits=["r0"])
    with pytest.raises(ValueError, match="region goals require entry='itags'"):
        ground_problem(_domain(tmp_path, entry="pddl"), G, {"spot1": np.zeros(3)}, goal)


def test_achieved_region_goal_plans_nothing(G, tmp_path):
    graph, vertices, symbol_of = _graph(G)
    _, _, vp = expand_regions(
        GrstapsGoal(goal_points=[], robot_id="spot1", region_visits=["r0"]),
        G, graph, symbol_of, {"spot1": _start(vertices, 0, 0)},
    )  # fmt: skip
    goal = GrstapsGoal(
        goal_points=[],
        robot_id="spot1",
        region_visits=["r0"],
        world_state=WorldState(visited=set(vp["r0"])),
    )
    ground_problem(_domain(tmp_path), G, {"spot1": np.zeros(3)}, goal)
    assert not os.path.exists(tmp_path / "regions_spot1" / "itags_input.json")


def test_goal_manager_treats_region_goals_as_unknown():
    gm = pytest.importorskip("omniplanner_ros.goal_manager_node")
    for goal in (
        "(visited-region r0)",
        "(explored-region-by spot1 r0)",
        "(and (visited-place p1) (explored-region r1))",
    ):
        assert gm.goal_satisfied_by(goal, {"p1", "p2"}) is None
    # ...while a goal it can judge stays judged.
    assert (
        gm.goal_satisfied_by("(and (visited-place p1) (visited-region r0))", set())
        is False
    )
