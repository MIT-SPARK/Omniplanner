"""Goal-relevant grounding for the multi-robot PDDL domain (scene_scope).

The full encoding gives Fast Downward every place, object and region of the
map; on a fused multi-session map its translator never finishes. The
goal-relevant one keeps only the goal's points of interest, joined by shortest
paths in metres, and must plan the same missions:

* visits, robot assignment (v4 ``-by``), inspect -> pick -> place, ``or`` goals;
* runtime constraints: routes -- and the paths a plan is expanded into -- go
  around a forbidden place;
* robots in different components; an object a robot already holds;
* which robot is closer is decided in metres, not hops.

Needs fast-downward on PATH (as the other planning tests do).
"""

import ast
import math
import pathlib
from importlib.resources import as_file, files

import dsg_pddl.domains
import numpy as np
import pytest
import spark_dsg
from dsg_pddl.dsg_pddl_grounding_multirobot import ground_problem
from dsg_pddl.dsg_pddl_planning import make_plan
from dsg_pddl.grounding_errors import MissingSymbolError
from dsg_pddl.pddl_grounding import ConstraintFact, MultiRobotPddlDomain, PddlGoal

from omniplanner.world_state import WorldState

DOMAIN = "RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl"
NX, NY = 10, 6  # grid of places, 1 m apart


def place(x, y):
    return f"t{y * NX + x}"


def _new_dsg():
    """An empty graph laid out like the maps: mesh places are layer 3, partition 1, under rooms."""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    return G


def _add_place(G, idx, xy):
    a = spark_dsg.PlaceNodeAttributes()
    a.position = np.array([xy[0], xy[1], 0.0])
    G.add_node(spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("t", idx).value, a)


def _add_object(G, idx, xy):
    a = spark_dsg.ObjectNodeAttributes()
    a.position = np.array([xy[0], xy[1], 0.0])
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", idx).value, a)


def _edge(G, a, b):
    G.insert_edge(
        spark_dsg.NodeSymbol("t", a).value, spark_dsg.NodeSymbol("t", b).value
    )


def grid_dsg():
    """10 x 6 places, 4-connected, four objects, and region r0 = the places with x >= 7.

    Regions are r(i), as hydra_multi's open-vocabulary regions are.
    """
    G = _new_dsg()
    for y in range(NY):
        for x in range(NX):
            _add_place(G, y * NX + x, (x, y))
    for y in range(NY):
        for x in range(NX):
            if x + 1 < NX:
                _edge(G, y * NX + x, y * NX + x + 1)
            if y + 1 < NY:
                _edge(G, y * NX + x, (y + 1) * NX + x)
    for i, xy in enumerate([(1.2, 0.3), (8.1, 4.2), (5.3, 2.1), (0.2, 5.1)]):
        _add_object(G, i, xy)
    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([8.0, 2.5, 0.0])
    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("r", 0).value, room)
    for y in range(NY):
        for x in range(7, NX):
            G.insert_edge(
                spark_dsg.NodeSymbol("r", 0).value,
                spark_dsg.NodeSymbol("t", y * NX + x).value,
            )
    return G


ROBOTS = {"hamilton": np.array([0.0, 0.0]), "euclid": np.array([9.0, 5.0])}


def _domain(scene_scope):
    with as_file(files(dsg_pddl.domains).joinpath(DOMAIN)) as p:
        domain = MultiRobotPddlDomain(open(p).read())
    domain.scene_scope = scene_scope
    return domain


def _grounded(
    G, goal, robots=ROBOTS, scope="goal_relevant", constraints=(), world_state=None
):
    g = PddlGoal(
        pddl_goal=goal,
        robot_id="hamilton",
        constraints=list(constraints),
        world_state=world_state,
    )
    return ground_problem(_domain(scope), G, robots, g).value


def _plan(G, goal, **kw):
    return make_plan(_grounded(G, goal, **kw), G)


def _travel(plan):
    """Metres driven: length of every expanded goto path."""
    total = 0.0
    for action, path in zip(plan.symbolic_actions, plan.parameterized_actions):
        if action[0] == "goto-poi":
            pts = [np.asarray(p)[:2] for p in path]
            total += sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:]))
    return total


def _visits(plan, robot=None):
    return {
        a[3]
        for a in plan.symbolic_actions
        if a[0] == "goto-poi" and robot in (None, a[1])
    }


def _index(plan, action):
    return plan.symbolic_actions.index(action)


# ---------------------------------------------------------------------------
# the problem itself
# ---------------------------------------------------------------------------


def test_problem_keeps_only_goal_points():
    G = grid_dsg()
    rel = _grounded(G, "(and (visited-object o0) (visited-object o1))")
    full = _grounded(G, "(and (visited-object o0) (visited-object o1))", scope="full")
    # robot starts (2) + o0, o1 + their places (2)
    assert len([s for s in rel.symbols.values() if s.layer in ("place", "object")]) == 6
    assert len(full.symbols) > 60
    assert rel.layer_planner is not None and full.layer_planner is None


def test_distances_are_shortest_paths_in_decimetres():
    G = grid_dsg()
    problem = _grounded(G, "(visited-object o1)").problem_str
    # hamilton starts at t0 (0, 0); o1 sits 0.22 m from place (8, 4): 12 m along the grid
    o1 = math.hypot(0.1, 0.2)
    assert f"(= (distance t0 o1) {round((12 + o1) * 10)})" in problem


def test_missing_symbol_is_reported():
    with pytest.raises(MissingSymbolError):
        _grounded(grid_dsg(), "(visited-object o99)")


# ---------------------------------------------------------------------------
# same missions as the full encoding
# ---------------------------------------------------------------------------

GOALS = {
    "visits": "(and (visited-object o0) (visited-object o1))",
    "assigned": "(and (visited-object-by euclid o0) (visited-object-by hamilton o1))",
    "inspect_pick_place": f"(and (inspected-by hamilton o2) (object-in-place o2 {place(9, 0)}))",
    "or": "(or (visited-object o3) (visited-object o1))",
    "end_at": f"(and (at-place hamilton {place(9, 5)}) (safe o2))",
    "region": "(visited-region r0)",
}


def _check(name, plan):
    acts = plan.symbolic_actions
    if name == "visits":
        assert {"o0", "o1"} <= _visits(plan)
    elif name == "assigned":
        assert "o0" in _visits(plan, "euclid") and "o1" in _visits(plan, "hamilton")
    elif name == "inspect_pick_place":
        picks = [a for a in acts if a[0] == "pick-object" and a[2] == "o2"]
        places = [a for a in acts if a[0] == "place-object" and a[2] == "o2"]
        assert ("inspect", "hamilton", "o2") in acts and picks and places
        assert places[-1][3] == place(9, 0)
        assert (
            _index(plan, ("inspect", "hamilton", "o2"))
            < _index(plan, picks[0])
            < _index(plan, places[-1])
        )
    elif name == "or":
        assert _visits(plan) & {"o3", "o1"}
    elif name == "end_at":
        last = [a for a in acts if a[0] == "goto-poi" and a[1] == "hamilton"][-1]
        assert last[3] == place(9, 5) and any(
            a[0] == "inspect" and a[2] == "o2" for a in acts
        )
    elif name == "region":
        assert any(int(t[1:]) % NX >= 7 for t in _visits(plan) if t.startswith("t"))


@pytest.mark.parametrize("name", sorted(GOALS))
def test_same_missions_as_full_encoding(name):
    G = grid_dsg()
    rel, full = _plan(G, GOALS[name]), _plan(G, GOALS[name], scope="full")
    _check(name, rel)
    _check(name, full)
    # never a longer drive than the full encoding's plan (small slack: both searches are greedy)
    assert _travel(rel) <= _travel(full) * 1.05 + 1e-6


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


def corridors_dsg():
    """Two rows (y = 0 and y = 4) joined by a short connector at x = 0 and a long one at x = 9."""
    G = _new_dsg()
    idx = {}

    def add(x, y):
        idx[(x, y)] = len(idx)
        _add_place(G, idx[(x, y)], (x, y))

    for x in range(NX):
        add(x, 0)
        add(x, 4)
    for x in (0, 9):
        for y in (1, 2, 3):
            add(x, y)
    for (x, y), i in idx.items():
        for dx, dy in ((1, 0), (0, 1)):
            j = idx.get((x + dx, y + dy))
            if j is not None:
                _edge(G, i, j)
    _add_object(G, 0, (0.2, 4.1))
    return G, idx


def test_forbidden_place_is_routed_around_in_the_plan_and_its_path():
    G, idx = corridors_dsg()
    robots = {"hamilton": np.array([0.0, 0.0])}
    blocked = f"t{idx[(0, 2)]}"
    free = _plan(G, "(visited-object o0)", robots=robots)
    assert _travel(free) < 6
    plan = _plan(
        G,
        "(visited-object o0)",
        robots=robots,
        constraints=[ConstraintFact(predicate="forbidden-poi", symbols=[blocked])],
    )
    # the long way: along y = 0, up x = 9, back along y = 4
    assert _travel(plan) > 20
    for action, path in zip(plan.symbolic_actions, plan.parameterized_actions):
        if action[0] == "goto-poi":
            assert (
                min(np.linalg.norm(np.asarray(p)[:2] - [0.0, 2.0]) for p in path) > 0.9
            )
    # the places passed on the way are reported, not just the goal
    assert f"t{idx[(9, 2)]}" in plan.traversed["hamilton"]


def test_forbidden_goal_is_unreachable_with_a_hint():
    G, idx = corridors_dsg()
    grounded = _grounded(
        G,
        "(visited-object o0)",
        robots={"hamilton": np.array([0.0, 0.0])},
        constraints=[ConstraintFact(predicate="forbidden-poi", symbols=["o0"])],
    )
    assert "o0 cannot be reached" in grounded.failure_hint


def test_forbidden_edge_is_not_driven():
    G, idx = corridors_dsg()
    a, b = f"t{idx[(0, 1)]}", f"t{idx[(0, 2)]}"
    plan = _plan(
        G,
        "(visited-object o0)",
        robots={"hamilton": np.array([0.0, 0.0])},
        constraints=[ConstraintFact(predicate="forbidden-edge", symbols=[a, b])],
    )
    # the short connector is cut between (0, 1) and (0, 2): the long way round
    assert _travel(plan) > 20
    cut = {(0.0, 1.0), (0.0, 2.0)}
    for action, path in zip(plan.symbolic_actions, plan.parameterized_actions):
        if action[0] == "goto-poi":
            steps = [tuple(np.round(np.asarray(p)[:2], 3)) for p in path[1:-1]]
            assert not any({u, v} == cut for u, v in zip(steps, steps[1:]))


def test_robot_starting_on_a_forbidden_place_can_leave():
    G, idx = corridors_dsg()
    robots = {"hamilton": np.array([0.0, 0.0])}
    plan = _plan(
        G,
        "(visited-object o0)",
        robots=robots,
        constraints=[
            ConstraintFact(predicate="forbidden-poi", symbols=[f"t{idx[(0, 0)]}"])
        ],
    )
    assert "o0" in _visits(plan, "hamilton")
    # it drives there along the graph (places 1 m apart), no jump out of the area
    assert _travel(plan) >= 4.0
    for action, path in zip(plan.symbolic_actions, plan.parameterized_actions):
        if action[0] == "goto-poi":
            nodes = [np.asarray(p)[:2] for p in path[1:-1]]
            assert all(
                np.linalg.norm(b - a) <= 1.0 + 1e-9 for a, b in zip(nodes, nodes[1:])
            )
            # ending at the graph node next to the target, not one 4 m away
            assert np.linalg.norm(np.asarray(path[-1])[:2] - nodes[-1]) < 1.0


def test_object_beside_a_forbidden_place_is_off_limits():
    G, idx = corridors_dsg()
    # o0 sits 0.22 m from (0, 4); forbid that place, not the object
    grounded = _grounded(
        G,
        "(visited-object o0)",
        robots={"hamilton": np.array([0.0, 0.0])},
        constraints=[
            ConstraintFact(predicate="forbidden-poi", symbols=[f"t{idx[(0, 4)]}"])
        ],
    )
    assert "o0 cannot be reached" in grounded.failure_hint


# ---------------------------------------------------------------------------
# robots
# ---------------------------------------------------------------------------


def test_closer_robot_is_chosen_in_metres_not_hops():
    """hamilton: 2 hops of 5 m to the object; euclid: 6 hops of 1 m."""
    G = _new_dsg()
    for i, xy in enumerate([(0, 0), (5, 0), (10, 0)]):  # t0 - t1 - t2 (object at t2)
        _add_place(G, i, xy)
    for i in range(6):  # t3..t8 from (10, 6) down to (10, 1)
        _add_place(G, 3 + i, (10, 6 - i))
    for a, b in [(0, 1), (1, 2)] + [(3 + i, 4 + i) for i in range(5)] + [(8, 2)]:
        _edge(G, a, b)
    _add_object(G, 0, (10.0, 0.0))
    robots = {"hamilton": np.array([0.0, 0.0]), "euclid": np.array([10.0, 6.0])}
    plan = _plan(G, "(visited-object o0)", robots=robots)
    assert "o0" in _visits(plan, "euclid") and not _visits(plan, "hamilton")


def test_robots_in_separate_components():
    G = grid_dsg()
    # a second, unconnected 2-place island with an object
    _add_place(G, 100, (30.0, 0.0))
    _add_place(G, 101, (31.0, 0.0))
    _edge(G, 100, 101)
    _add_object(G, 9, (31.1, 0.0))
    robots = {"hamilton": np.array([0.0, 0.0]), "euclid": np.array([30.0, 0.0])}
    plan = _plan(
        G,
        "(and (visited-object-by euclid o9) (visited-object-by hamilton o2))",
        robots=robots,
    )
    assert "o9" in _visits(plan, "euclid") and "o2" in _visits(plan, "hamilton")
    with pytest.raises(Exception):
        _plan(G, "(visited-object-by hamilton o9)", robots=robots)


def test_a_held_object_is_put_down_before_the_next_pick():
    G = grid_dsg()
    held = WorldState(holding={"hamilton": ["o3"]})
    plan = _plan(
        G, "(and (inspected-by hamilton o2) (holding hamilton o2))", world_state=held
    )
    acts = plan.symbolic_actions
    put_down = [
        a for a in acts if a[0] == "place-object" and a[1:3] == ("hamilton", "o3")
    ]
    pick = [a for a in acts if a[0] == "pick-object" and a[1:3] == ("hamilton", "o2")]
    assert put_down and pick and _index(plan, put_down[0]) < _index(plan, pick[0])


def test_each_robot_acts_from_where_it_is():
    """Interleaved actions: an inspect starts where *that* robot's last goto ended."""
    G = grid_dsg()
    plan = _plan(G, "(and (inspected-by hamilton o0) (inspected-by euclid o1))")
    for action, path in zip(plan.symbolic_actions, plan.parameterized_actions):
        if action[0] == "inspect":
            target = plan.symbols[action[2]].position[:2]
            assert np.linalg.norm(np.asarray(path[0])[:2] - target) < 1.0


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------


def test_region_visit_uses_a_few_representative_places():
    """Three central places, plus the member nearest each robot (so a visit stays short)."""
    rel = _grounded(grid_dsg(), "(visited-region r0)")
    members = [f for f in rel.problem_str.splitlines() if "(place-in-region" in f]
    assert 0 < len(members) <= 3 + len(ROBOTS)
    # euclid starts on the region's corner (9, 5): its nearest other member is next to it
    assert {
        f"(place-in-region {place(8, 5)} r0)",
        f"(place-in-region {place(9, 4)} r0)",
    } & set(members)


def test_explored_region_includes_every_place():
    G = grid_dsg()
    rel = _grounded(G, "(explored-region r0)")
    assert rel.problem_str.count("(place-in-region") == 3 * NY
    plan = make_plan(rel, G)
    region = {place(x, y) for x in range(7, NX) for y in range(NY)}
    assert region <= _visits(plan)


def _extract_visited_pois():
    """multirobot_ros._extract_visited_pois_from_pipeline from this checkout's source.

    The ROS module pulls in rclpy / omniplanner_msgs (see
    test_pddl_planner_ros_source), so the two plain functions are compiled out
    of its source with the non-ROS names they use.
    """
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "omniplanner_ros/src/omniplanner_ros/multirobot_ros.py"
    )
    if not path.is_file():
        pytest.skip(f"{path} not present in this checkout")
    tree = ast.parse(path.read_text())
    wanted = {"_peel_symbolic", "_extract_visited_pois_from_pipeline"}
    module = ast.Module(
        body=[n for n in tree.body if getattr(n, "name", None) in wanted],
        type_ignores=[],
    )
    from dsg_pddl.dsg_pddl_planning import PddlPlan
    from omniplanner.omniplanner import MultiRobotWrapper, SymbolicContext

    namespace = {
        "PddlPlan": PddlPlan,
        "MultiRobotWrapper": MultiRobotWrapper,
        "SymbolicContext": SymbolicContext,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_extract_visited_pois_from_pipeline"]


def test_traversed_places_reach_the_goal_manager():
    extract = _extract_visited_pois()
    from omniplanner.omniplanner import MultiRobotWrapper

    G, idx = corridors_dsg()
    plan = _plan(G, "(visited-object o0)", robots={"hamilton": np.array([0.0, 0.0])})
    wrapper = MultiRobotWrapper(["hamilton"], plan)
    wrapper.set_name_remap("hamilton", "hamilton")
    visited = extract(wrapper)["hamilton"]
    assert "o0" in visited and f"t{idx[(0, 2)]}" in visited
