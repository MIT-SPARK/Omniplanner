import logging
import os
import re
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import spark_dsg
from plum import dispatch

from dsg_pddl.dsg_pddl_grounding import (
    DEFAULT_SCENE_SCOPE,
    GOAL_RELEVANT_MAX_REGION_PLACES,
    add_symbol_positions,
    collect_goal_symbol_names,
    could_name_a_scene_symbol,
    explicit_edges_from_layer,
    extract_all_symbols,
    generate_object_containment,
    generate_objects,
    generate_place_containment,
    implicit_edges_from_layers,
    normalize_symbols,
    normalized,
    simplify,
    symbol_connectivity_to_pddl,
)
from dsg_pddl.grounding_errors import (
    MissingSymbol,
    MissingSymbolError,
    kind_hint_for_symbol,
)
from dsg_pddl.pddl_grounding import (
    GroundedPddlProblem,
    MultiRobotPddlDomain,
    PddlGoal,
    PddlProblem,
    PddlSymbol,
)
from dsg_pddl.pddl_utils import (
    extract_facts,
    lisp_string_to_ast,
    pddl_symbol_to_node_symbol,
)
from omniplanner.omniplanner import MultiRobotWrapper
from omniplanner.tsp import LayerPlanner
from omniplanner.world_state import goal_tokens, unavailable_robots

logger = logging.getLogger(__name__)


def assigned_robots(constraints):
    """Robots a ``use-robots`` constraint limits the plan to, or None for any.

    Visit goals name no robot, so "robot hilbert should go to t5" needs this to
    stop the planner giving the job to whichever robot is nearer.
    """
    names = [
        s.lower()
        for c in constraints or []
        if c.predicate == "use-robots"
        for s in c.symbols
    ]
    return set(names) or None


def planning_robot_states(robot_states, constraints, busy=()):
    """The robots that may act on this goal: all with a pose, or those named,
    less the busy ones an additional task must leave alone."""
    assigned = assigned_robots(constraints)
    unknown = sorted((assigned or set()) - set(robot_states))
    if unknown:
        raise ValueError(_unknown_robots(unknown, robot_states))
    if busy:
        robot_states = {r: p for r, p in robot_states.items() if r not in busy}
        if not any(p is not None for p in robot_states.values()):
            raise ValueError(
                f"every robot that could do this is busy ({', '.join(sorted(busy))}); "
                "name one in the goal to interrupt it"
            )
    if assigned is None:
        return robot_states
    usable = {
        r: pose
        for r, pose in robot_states.items()
        if r in assigned and pose is not None
    }
    if not usable:
        known = sorted(r for r, pose in robot_states.items() if pose is not None)
        raise ValueError(
            f"use-robots names {sorted(assigned)}, but none of them has a pose; "
            f"robots with a pose: {known}"
        )
    return usable


def _unknown_robots(names, robot_states) -> str:
    return "; ".join(
        f"unknown robot '{n}' (known: {', '.join(sorted(robot_states))})" for n in names
    )


def predicate_argument_types(domain) -> Dict[str, List[str]]:
    """{predicate -> the type of each argument}, from the domain's :predicates."""
    out = {}
    for pred in getattr(domain, "predicates", ()) or ():
        if not pred or not isinstance(pred[0], str):
            continue
        types, pending, i = [], 0, 1
        while i < len(pred):
            tok = pred[i]
            if tok == "-" and i + 1 < len(pred):
                types += [pred[i + 1]] * pending
                pending, i = 0, i + 2
                continue
            pending += 1
            i += 1
        types += ["object"] * pending
        out[pred[0]] = types
    return out


def check_goal_symbols(goal_string, arg_types, robots, symbols):
    """Fail early, and say why, when a goal names a robot or symbol that does
    not exist: fast-downward would only report it could not read the problem.

    Robot arguments are checked against the whole fleet (robots), the rest
    against the scene graph's places, objects and regions.
    """
    try:
        ast = lisp_string_to_ast((goal_string or "").replace(")(", ") ("))
    except Exception:
        return  # fast-downward says what is wrong with it
    known = {s.symbol for s in symbols}
    bad_robots, missing = [], []

    def walk(node):
        if isinstance(node, str) or not node:
            return
        head = node[0]
        if not isinstance(head, str) or head not in arg_types:
            for sub in node:
                walk(sub)
            return
        for arg, typ in zip(node[1:], arg_types[head]):
            if not isinstance(arg, str) or arg.startswith("?"):
                continue
            arg = arg.lower()
            if typ == "robot":
                if arg not in robots and arg not in bad_robots:
                    bad_robots.append(arg)
            elif arg not in known and arg not in missing:
                missing.append(arg)

    walk(ast)
    if bad_robots:
        raise ValueError(_unknown_robots(bad_robots, robots))
    if missing:
        regions = sorted(s.symbol for s in symbols if s.layer == "region")
        hint = ""
        if any(m.startswith("r") for m in missing):
            hint = f" (regions in it: {', '.join(regions) or 'none'})"
        raise ValueError(
            f"The goal could not be read: it names {', '.join(missing)}, which "
            f"the scene graph does not have{hint}"
        )


def _extract_forbidden_sets(constraints):
    """Pull forbidden POIs and forbidden edges out of a list of ConstraintFact-like objects.

    Each entry is expected to have `predicate` and `symbols` attributes
    (matches both the Python dataclass and the ROS `ConstraintFact` message).
    """
    forbidden_pois = set()
    forbidden_edges = set()
    for c in constraints or []:
        if c.predicate == "forbidden-poi" and len(c.symbols) >= 1:
            forbidden_pois.add(c.symbols[0])
        elif c.predicate == "forbidden-edge" and len(c.symbols) >= 2:
            # Store unordered so we match either direction
            forbidden_edges.add(tuple(sorted([c.symbols[0], c.symbols[1]])))
    return forbidden_pois, forbidden_edges


# A forbidden-poi names one node, but the robot should stay out of its
# neighbourhood, not just off the node itself. Expand each forbidden POI to every
# symbol within this many metres of it, measured as navigable path distance.
#
# Node spacing decides what a given radius means, and it varies enough between
# maps that one number cannot be right for both:
#
#   b45                 median nearest-neighbour 2.1 m; 5 m pulls in ~9 symbols,
#                       about the 8.3 average graph degree plus a margin
#   plan_repair_graph   nearest neighbour 0.61 m; 5 m pulls in 38 symbols
#
# 5 m was tuned on b45 and is far too wide on the denser graph, where it
# swallowed a goal object lying 2.4 m from the forbidden POI, isolated it, and
# left the problem unsolvable. 1 m keeps the exclusion to the node and its
# immediate neighbour there and the same goal plans fine.
#
# The tension is real: 1 m is below b45's node spacing, so on a sparse map it
# expands to nothing and only the named node is excluded. Retune per map --
# ADT4_FORBIDDEN_RADIUS_M overrides this without a rebuild -- or switch to graph
# adjacency, which would be spacing-independent and is the better long-term fix.
FORBIDDEN_RADIUS_M = 1.0

# Env override so the radius can be retuned per map without a rebuild. Read at
# call time, not import time, so a field test only has to restart the node.
FORBIDDEN_RADIUS_ENV = "ADT4_FORBIDDEN_RADIUS_M"


def _forbidden_radius_m():
    """Exclusion radius in metres, from the environment or the default above.

    A malformed value falls back rather than raising: this runs mid-mission on
    every grounding, and a typo in the tmux env should not take the planner down.
    """
    raw = os.getenv(FORBIDDEN_RADIUS_ENV)
    if raw is None:
        return FORBIDDEN_RADIUS_M
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using the %.1f m default",
            FORBIDDEN_RADIUS_ENV,
            raw,
            FORBIDDEN_RADIUS_M,
        )
        return FORBIDDEN_RADIUS_M


def _expand_forbidden_pois(forbidden_pois, symbols, symbol_lookup, layer_planner):
    """Grow each forbidden POI into the set of symbols within the exclusion radius.

    Distance is path distance through the places layer, not Euclidean, so a
    symbol that is metrically close but only reachable the long way around is
    left alone. Disconnected symbols come back as inf and are likewise excluded.
    """
    radius = _forbidden_radius_m()
    if not forbidden_pois or radius <= 0:
        return forbidden_pois

    expanded = set(forbidden_pois)
    for name in forbidden_pois:
        origin = symbol_lookup.get(name)
        if origin is None:
            continue
        for s in symbols:
            if s.symbol in expanded:
                continue
            d = layer_planner.get_external_distance(origin.position[:2], s.position[:2])
            if d < radius:
                expanded.add(s.symbol)

    if len(expanded) > len(forbidden_pois):
        logger.info(
            "Expanded %d forbidden POI(s) to %d symbols within %.1f m",
            len(forbidden_pois),
            len(expanded),
            radius,
        )
    return expanded


def _edge_allowed(s, t, forbidden_pois, forbidden_edges):
    if s.symbol in forbidden_pois or t.symbol in forbidden_pois:
        return False
    if tuple(sorted([s.symbol, t.symbol])) in forbidden_edges:
        return False
    return True


def generate_dense_region_symbol_connectivity_multirobot(
    G, symbols, robot_states, constraints=None
):
    symbol_lookup = {s.symbol: s for s in symbols}
    forbidden_pois, forbidden_edges = _extract_forbidden_sets(constraints)
    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)
    edges = []

    # Place <-> Place Edges
    edges += explicit_edges_from_layer(symbol_lookup, G, places_layer)

    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)
    # Object <-> Object Edges
    edges += implicit_edges_from_layers(
        symbol_lookup,
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        True,
        3,
        layer_planner,
    )

    # Object <-> Place Edges
    edges += implicit_edges_from_layers(
        symbol_lookup,
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        places_layer,
        False,
        10,
        layer_planner,
    )
    # No synthetic start edges: robots begin at a real place (see
    # robot_start_places), so they inherit that node's own connectivity.

    forbidden_pois = _expand_forbidden_pois(
        forbidden_pois, symbols, symbol_lookup, layer_planner
    )

    if forbidden_pois or forbidden_edges:
        before = len(edges)
        edges = [
            e
            for e in edges
            if _edge_allowed(e[0], e[1], forbidden_pois, forbidden_edges)
        ]
        logger.info(
            "Dropped %d connectivity edges due to runtime constraints (%d forbidden POIs, %d forbidden edges)",
            before - len(edges),
            len(forbidden_pois),
            len(forbidden_edges),
        )

    return edges


def nearest_place_for_position(place_symbols: List[PddlSymbol], pos: np.ndarray):
    """(place symbol, metres away) for the place closest to `pos`.

    Compared in 2D: symbol positions carry a z that the robot's pose does not,
    and a height difference should not decide which place a robot is standing on.
    """
    target = np.asarray(pos, dtype=float)[:2]
    best_name, best_d = None, float("inf")
    for p in place_symbols:
        d = float(np.linalg.norm(np.asarray(p.position, dtype=float)[:2] - target))
        if d < best_d:
            best_d, best_name = d, p.symbol
    return best_name, best_d


# Beyond this the nearest place is probably not the one the robot is actually
# standing on -- nearest is measured straight-line, so a robot beside a partition
# can snap to a node on the other side of it. Worth a look in the log rather than
# a hard failure: a bad snap still plans, it just plans from the wrong doorway.
START_SNAP_WARN_M = 5.0


def robot_start_places(symbols: List[PddlSymbol], robot_states: Dict[str, np.ndarray]):
    """{robot id -> the place symbol it starts from}.

    A robot is generally not standing on a node. This used to be modelled with a
    synthetic `pstart<robot>` symbol at the exact pose, joined to every POI within
    a threshold -- edges the traversability graph does not contain. Worse, that
    threshold was compared against LayerPlanner distances, which count graph hops
    rather than metres, so a POI ten metres away could be admitted as readily as
    an adjacent one and priced the same. The planner then treated a distant goal
    as though the robot were next to it.

    Starting at the nearest real place removes all of that: the robot inherits
    the node's genuine edges and their genuine costs, and nothing is synthesised.
    The metre or two between the true pose and that node is handled downstream,
    where the first leg is parameterised from the robot's actual pose.
    """
    places = [s for s in symbols if s.layer == "place"]
    if not places:
        logger.warning("DSG has no place symbols; robots cannot be given a start")
        return {}

    starts = {}
    for robot_id, pose in robot_states.items():
        if pose is None:
            continue
        name, d = nearest_place_for_position(places, pose)
        starts[robot_id] = name
        if d > START_SNAP_WARN_M:
            logger.warning(
                "Robot %s starts at %s, %.1f m from its actual pose; the nearest "
                "place may be across an obstacle",
                robot_id,
                name,
                d,
            )
        else:
            logger.info("Robot %s starts at place %s (%.1f m away)", robot_id, name, d)
    return starts


# Multirobot init wrapper that reuses shared connectivity and containment helpers
def generate_dense_region_init_multirobot(
    G: spark_dsg.DynamicSceneGraph,
    symbols_of_interest: List[PddlSymbol],
    robot_states: Dict[str, np.ndarray],
    constraints=None,
) -> List[tuple]:
    connectivity = generate_dense_region_symbol_connectivity_multirobot(
        G, symbols_of_interest, robot_states, constraints=constraints
    )
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl: List[tuple] = [("=", ("total-cost",), 0)]
    initial_pddl += connectivity_pddl

    # Containment relations (objects and places -> regions)
    initial_pddl += generate_object_containment(G)
    initial_pddl += generate_place_containment(G)

    # Each robot starts at the real place nearest its pose.
    for robot_id, place in robot_start_places(
        symbols_of_interest, robot_states
    ).items():
        initial_pddl.append(("at-poi", robot_id, place))

    return initial_pddl


def diagnose_unreachable_goal(goal_string, init_facts, symbols):
    """Names the goal symbols nothing can reach, so a failure says what to fix.

    fast-downward only reports that a problem is unsolvable, and the operator is
    left with a 50k-line problem file. Almost always the cause is a constraint:
    a forbidden-poi on or beside a goal strips every `connected` fact the goal
    had, and no plan can reach it. That is cheap to detect here, where both the
    goal and the surviving connectivity are in hand.

    Returns the unreachable goal symbols, and logs them. Diagnostic only -- it
    changes nothing about what is planned.
    """
    connected = set()
    for fact in init_facts:
        if fact and fact[0] == "connected" and len(fact) >= 3:
            connected.add(fact[1])
            connected.add(fact[2])

    known = {sym.symbol for sym in symbols}
    named = {tok for tok in re.findall(r"[a-z]+[0-9]+", (goal_string or "").lower())}
    goal_symbols = named & known
    unreachable = sorted(goal_symbols - connected)

    if unreachable:
        logger.error(
            "Goal names %s, which %s no connectivity left in the problem: no plan "
            "can reach %s. A forbidden-poi on or within the expansion radius of a "
            "goal will do this -- check the constraints, or lower "
            "%s (currently %.1f).",
            ", ".join(unreachable),
            "has" if len(unreachable) == 1 else "have",
            "it" if len(unreachable) == 1 else "them",
            FORBIDDEN_RADIUS_ENV,
            _forbidden_radius_m(),
        )
    elif goal_symbols:
        logger.debug("All goal symbols %s still connected", sorted(goal_symbols))
    return unreachable


def filter_goal_for_available_objects(
    goal_string: str, available_objects: List[str]
) -> str:
    """Filter goal string to only include objects that are available in the problem."""
    import re

    # Extract object names from goal string (e.g., o2, o3, o21, etc.)
    object_pattern = r"\bo\d+\b"
    goal_objects = re.findall(object_pattern, goal_string)

    # Filter to only include available objects
    available_goal_objects = [obj for obj in goal_objects if obj in available_objects]

    # Create new goal string with only available objects
    if not available_goal_objects:
        return "(and)"  # Empty goal if no objects available

    # For safety goals, create individual safe predicates
    safe_goals = [f"(safe {obj})" for obj in available_goal_objects]
    return f"(and {' '.join(safe_goals)})"


def apply_world_state(init_facts, world_state, symbols, robot_ids):
    """Start the problem from what the mission has already done.

    Fast-downward plans from :init, so a replan only plans what is left once
    :init is true: places visited stay visited, and an object a robot holds is
    in its hand -- not lying where it was picked up, to be picked up again.
    """
    known = {s.symbol for s in symbols}
    held = {}
    for robot, objects in (world_state.holding or {}).items():
        if robot not in robot_ids:
            continue
        for obj in objects:
            if obj in known:
                held[obj] = robot
    facts = [
        f
        for f in init_facts
        if not (f and f[0] == "object-in-place" and len(f) >= 2 and f[1] in held)
    ]
    for obj, robot in sorted(held.items()):
        facts.append(("holding", robot, obj))
    for robot in sorted(set(held.values())):
        facts.append(("hand-full", robot))
    # (object-placed-by r o p): what a robot put down and left where it is.
    place_of = {
        f[1]: f[2] for f in facts if f and f[0] == "object-in-place" and len(f) >= 3
    }
    for robot in robot_ids:
        for obj in sorted(getattr(world_state, "placed_by", {}).get(robot, set())):
            if obj in place_of:
                facts.append(("object-placed-by", robot, obj, place_of[obj]))
    facts += [("visited-poi", s) for s in sorted(world_state.visited & known)]
    for robot in robot_ids:
        done = getattr(world_state, "visited_by", {}).get(robot, set())
        facts += [("visited-poi-by", robot, s) for s in sorted(done & known)]
        done = getattr(world_state, "inspected_by", {}).get(robot, set())
        facts += [("inspected-by", robot, s) for s in sorted(done & known)]
    if held or world_state.visited or world_state.inspected:
        logger.info(
            "Planning from mission state: holding=%s, %d visited, %d inspected",
            {o: r for o, r in held.items()},
            len(world_state.visited & known),
            len(world_state.inspected & known),
        )
    return facts


MANIPULATION_BY = ("object-placed-by", "inspected-by", "holding")


def _manipulation_robots(goal_string: str) -> set:
    """Robots a goal asks to inspect, pick or place: (object-placed-by r o p),
    (inspected-by r o), (holding r o)."""
    preds = "|".join(MANIPULATION_BY)
    return set(re.findall(rf"\(\s*(?:{preds})\s+([a-z0-9_]+)", goal_string.lower()))


def _unique_symbols(symbols):
    """One symbol per lowercase name. A room R0 and an open-vocab region r0
    both become r0 in PDDL, and a name declared twice breaks the problem."""
    seen, out = set(), []
    for s in symbols:
        if s.symbol in seen:
            logger.warning(
                "Two scene graph nodes are both %s; keeping the first", s.symbol
            )
            continue
        seen.add(s.symbol)
        out.append(s)
    return out


def generate_multirobot_region_pddl(
    G: spark_dsg.DynamicSceneGraph,
    raw_pddl_goal_string: str,
    robot_states: np.ndarray,
    constraints=None,
    world_state=None,
) -> Tuple[str, List[PddlSymbol], str]:
    """Generate a multi-robot PDDL problem for domain region-object-rearrangement-domain-multirobot-fd.

    If ``constraints`` is non-empty, the corresponding ``(connected ...)`` facts
    are dropped from the generated PDDL ``:init`` so the planner cannot path
    through forbidden POIs / edges. The domain itself is unchanged.
    """
    # Collect all places/objects/regions and positions
    symbols = extract_all_symbols(G)
    normalize_symbols(symbols)
    symbols = _unique_symbols(symbols)

    symbols_of_interest = symbols
    for robot, initial_position in robot_states.items():
        if initial_position is None:
            logger.warning(
                f"Skipping robot {robot} due to missing initial position (None)."
            )

    add_symbol_positions(G, symbols_of_interest)

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)
    goal_pddl = simplify(parsed_pddl_goal)

    # Robot starts at nearest places to their given 2D states
    robot_ids = [rid for rid, pose in robot_states.items() if pose is not None]
    # Build init facts via shared helpers
    init_facts_tuples: List[tuple] = generate_dense_region_init_multirobot(
        G, symbols_of_interest, robot_states, constraints=constraints
    )
    failure_hint = _failure_hint(
        raw_pddl_goal_string, init_facts_tuples, symbols_of_interest, robot_states
    )

    # Ensure robot symbols exist (with positions) for downstream planners
    for rid, pose in robot_states.items():
        if pose is None:
            continue
        symbols_of_interest.append(
            PddlSymbol(rid, "robot", [], position=np.array(pose[:2]))
        )

    # Only robots with an arm and gaze inspect, pick and place: a Husky's plan
    # would silently drop those actions (compile_phoenix_pddl_plan).
    no_arm = {
        str(r).lower() for r in getattr(world_state, "cannot_manipulate", None) or ()
    }
    asked = _manipulation_robots(raw_pddl_goal_string) & no_arm
    if asked:
        raise ValueError(
            f"{', '.join(sorted(asked))} cannot inspect, pick or place (no arm): "
            "name a robot that can, or leave the robot out"
        )
    init_facts_tuples += [("can-manipulate", r) for r in robot_ids if r not in no_arm]

    # Every object starts suspicious, except those the mission already inspected.
    object_symbols = [s for s in symbols_of_interest if s.layer == "object"]
    inspected = world_state.inspected if world_state is not None else set()
    init_facts_tuples += [
        ("suspicious", o.symbol) for o in object_symbols if o.symbol not in inspected
    ]
    if world_state is not None:
        init_facts_tuples = apply_world_state(
            init_facts_tuples, world_state, symbols_of_interest, robot_ids
        )

    # Build objects dict using shared generator and adding robots
    pddl_objects = generate_objects(symbols_of_interest)
    pddl_objects["robot"] = robot_ids
    problem = PddlProblem(
        name="multi-robot-problem",
        domain="region-object-rearrangement-domain-multirobot-fd",
        objects=pddl_objects,
        initial_facts=tuple(init_facts_tuples),
        goal=goal_pddl,
        optimizing=True,
    )
    problem_str = problem.to_string()
    _dump_problem(problem_str)
    return problem_str, symbols_of_interest, failure_hint


def _failure_hint(
    raw_pddl_goal_string, init_facts, symbols, robot_states, too_far=None
):
    """Why the problem likely has no plan ("" if nothing stands out).

    ``too_far``: {object: metres to the nearest place the robots can reach}.
    """
    too_far = too_far or {}
    unreachable = diagnose_unreachable_goal(raw_pddl_goal_string, init_facts, symbols)
    far = [s for s in unreachable if s in too_far]
    cut = [s for s in unreachable if s not in too_far]
    hints = []
    if cut:
        hints.append(
            f"{', '.join(cut)} cannot be reached: the constraints cut off every "
            f"path to it (a forbidden-poi on it or within {_forbidden_radius_m():.1f} m)"
        )
    if far:
        hints.append(
            "; ".join(
                f"{s} cannot be reached: it is {too_far[s]:.1f} m from the nearest place "
                "the robots can get to (no path to it, or a forbidden area in the way)"
                for s in far
            )
        )
    failure_hint = "; ".join(hints)
    # A goal that assigns a task to a robot we have no pose for cannot be
    # planned: that robot is not in the problem at all.
    goal_tokens = set(re.findall(r"[a-z0-9_]+", raw_pddl_goal_string.lower()))
    posless = sorted(r for r, pose in robot_states.items() if pose is None)
    missing = [r for r in posless if r in goal_tokens]
    if missing:
        failure_hint = "; ".join(
            filter(
                None,
                [
                    failure_hint,
                    f"the goal assigns work to {', '.join(missing)}, but there is "
                    "no pose for it (its map->base_link transform is missing)",
                ],
            )
        )
    return failure_hint


def _dump_problem(problem_str):
    try:
        dump_dir = os.environ.get(
            "PDDL_DUMP_DIR", os.path.join(os.getcwd(), "pddl_dumps")
        )
        os.makedirs(dump_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        persistent_problem = os.path.join(dump_dir, f"{ts}_mr_region_problem.pddl")
        latest_problem = os.path.join(dump_dir, "mr_region_problem_latest.pddl")
        with open(persistent_problem, "w") as f:
            f.write(problem_str)
        with open(latest_problem, "w") as f:
            f.write(problem_str)
        logger.info(f"Saved multi-robot region PDDL to {persistent_problem}")
    except Exception as e:
        logger.warning(f"Failed to persist multi-robot region PDDL dump: {e}")


# An object farther than this from every place the robots can reach is not
# reachable (env override for maps whose objects sit far from the ground graph).
MAX_OBJECT_SNAP_M = 5.0


def _max_object_snap_m():
    try:
        return float(os.getenv("OMNIPLANNER_MAX_OBJECT_SNAP_M", MAX_OBJECT_SNAP_M))
    except ValueError:
        return MAX_OBJECT_SNAP_M


# Fast Downward costs are integers: goal-relevant distances go in as decimetres,
# so near-equal routes are not rounded into ties.
DISTANCE_SCALE = 10.0

# Search budget (whole seconds) for goal-relevant problems, which are small
# enough for an anytime search to cut route length (see
# pddl_planning.ANYTIME_FD_SEARCH); 0 keeps the single greedy search.
ANYTIME_SEARCH_S = float(os.getenv("OMNIPLANNER_FD_ANYTIME_S", "1"))


def _predicate_heads(ast):
    """Every name in head position of a goal AST: predicates and connectives."""
    heads = set()

    def walk(node):
        if isinstance(node, (tuple, list)) and node:
            if isinstance(node[0], str):
                heads.add(node[0].lower())
            for child in node[1:]:
                walk(child)

    walk(ast)
    return heads


def _node_symbol_name(node_id):
    return spark_dsg.NodeSymbol(node_id).str(True).lower()


def constrained_places_planner(G, robot_states, constraints, by_name):
    """The places-graph planner a goal-relevant problem is measured and expanded on.

    Metres (``LayerPlanner.weighted``), runtime constraints taken out of the graph,
    and snapping restricted to the components the robots start in. Returns the
    planner and the POI symbols the constraints forbid outright.

    A forbidden-poi removes every place within the exclusion radius of it
    (``_forbidden_radius_m``, measured along the graph in metres) so routes go
    around it, and rules out the objects whose nearest place is one of them. A
    robot that starts inside a forbidden area keeps the shortest way out of it.
    A forbidden-edge between two places removes that edge.
    """
    starts = [np.asarray(p)[:2] for p in robot_states.values() if p is not None]
    planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES).weighted()
    forbidden_pois, forbidden_edges = _extract_forbidden_sets(constraints)
    forbidden_symbols = set(forbidden_pois)
    if forbidden_pois or forbidden_edges:
        radius = _forbidden_radius_m()
        protected = {planner.get_closest_node_id(xy) for xy in starts}
        drop = set()
        for name in forbidden_pois:
            if name not in by_name:
                logger.warning("forbidden-poi %s is not in the scene graph", name)
                continue
            node = G.get_node(pddl_symbol_to_node_symbol(G, name))
            anchor = planner.get_closest_node_id(
                np.asarray(node.attributes.position)[:2]
            )
            drop |= set(
                nx.single_source_dijkstra_path_length(
                    planner.nx_layer,
                    anchor,
                    cutoff=max(radius, 0.0),
                    weight=planner.weight,
                )
            )
        # A robot that starts inside a forbidden area keeps its shortest way out
        # (its start place and the places to the nearest one outside the area).
        exits = set()
        for start in sorted(drop & protected):
            lengths, paths = nx.single_source_dijkstra(
                planner.nx_layer, start, weight=planner.weight
            )
            outside = [n for n in lengths if n not in drop]
            if outside:
                exits |= set(paths[min(outside, key=lengths.get)])
            else:
                exits.add(start)
            logger.warning(
                "Robot start %s is inside a forbidden area; keeping its way out (%d places)",
                _node_symbol_name(start),
                len(exits),
            )
        drop -= exits
        forbidden_symbols |= {_node_symbol_name(n) for n in drop}
        forbidden_symbols -= {_node_symbol_name(n) for n in exits}
        # objects inside a forbidden area are off limits too (they would
        # otherwise snap to a place just outside it)
        for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
            xy = np.asarray(node.attributes.position)[:2]
            if planner.get_closest_node_id(xy) in drop:
                forbidden_symbols.add(node.id.str(True).lower())
        edges = []
        for s, t in forbidden_edges:
            if (
                s in by_name
                and t in by_name
                and by_name[s].layer == by_name[t].layer == "place"
            ):
                edges.append(
                    (
                        pddl_symbol_to_node_symbol(G, s).value,
                        pddl_symbol_to_node_symbol(G, t).value,
                    )
                )
        planner = planner.without(drop, edges)
        logger.info(
            "Constraints removed %d places and %d edges from the places graph",
            len(drop),
            len(edges),
        )
    return planner.restricted_to_components(starts), forbidden_symbols, forbidden_edges


def generate_goal_relevant_multirobot_pddl(
    G: spark_dsg.DynamicSceneGraph,
    raw_pddl_goal_string: str,
    robot_states: Dict[str, np.ndarray],
    constraints=None,
    world_state=None,
    max_region_places: int = GOAL_RELEVANT_MAX_REGION_PLACES,
):
    """A multi-robot problem over only the goal-relevant points of interest.

    The full encoding (``generate_multirobot_region_pddl``) gives Fast Downward
    every place, object and region; on a fused multi-session map its translator
    never finishes (place-object alone is robots x objects x places). Here the
    points of interest are:

    * each robot's start (the reachable place nearest its pose),
    * every object, place and region the goal names (any and/or branch),
    * each named object's current place, so it can be picked up,
    * every object a robot holds (else the planner thinks the hand is free),
    * for each named region a few representative places (the ones nearest its
      centre, plus the one nearest each robot), or all of its places when the
      goal says ``explored-region``,

    joined pairwise with shortest-path distances in metres over the places graph
    with the runtime constraints applied. One ``goto-poi`` is then a whole
    route; ``make_plan`` expands it on the same graph (``layer_planner``).

    Returns (problem string, symbols, failure hint, layer planner).
    """
    parsed_goal = lisp_string_to_ast(raw_pddl_goal_string)
    goal_pddl = simplify(parsed_goal)
    robot_ids = [rid for rid, pose in robot_states.items() if pose is not None]

    all_symbols = extract_all_symbols(G)
    normalize_symbols(all_symbols)
    by_name = {s.symbol: s for s in all_symbols}

    planner, forbidden_symbols, forbidden_edges = constrained_places_planner(
        G, robot_states, constraints, by_name
    )

    selected = {}

    def select(name):
        selected.setdefault(name, by_name[name])

    def nearest_place(xy):
        return _node_symbol_name(planner.get_closest_node_id(np.asarray(xy)[:2]))

    start_place = {r: nearest_place(robot_states[r]) for r in robot_ids}
    for place in start_place.values():
        select(place)

    held = set()
    if world_state is not None:
        for robot, objects in (world_state.holding or {}).items():
            if robot in robot_ids:
                held |= set(objects)

    # predicate names (v4's visited-object-by, inspected-by, ...) are not symbols
    referenced = collect_goal_symbol_names(parsed_goal) - _predicate_heads(parsed_goal)
    explored = {
        f[1] for f in extract_facts(parsed_goal, "explored-region") if len(f) > 1
    }
    robot_names = {r.lower() for r in robot_states}
    region_places = None
    object_place, region_membership, missing, too_far = {}, [], [], {}
    for name in sorted(referenced | held):
        sym = by_name.get(name)
        if sym is None:
            if name in robot_names or not could_name_a_scene_symbol(name):
                continue
            if name not in referenced:  # held, but gone from the map
                logger.warning(
                    "A robot holds %s, which is not in the scene graph", name
                )
                continue
            logger.warning(
                "Goal references symbol '%s' not present in the scene graph.", name
            )
            missing.append(MissingSymbol(name, kind_hint_for_symbol(name)))
            continue
        if sym.layer == "object":
            select(name)
            node = G.get_node(pddl_symbol_to_node_symbol(G, name))
            xy = np.asarray(node.attributes.position)[:2]
            object_place[name] = nearest_place(xy)
            select(object_place[name])
            # nearest of the places the robots can reach: far off means the
            # object is somewhere they cannot get to (another component, or
            # behind a closed area), not next to their ground
            off = float(np.linalg.norm(planner.get_closest_point(xy) - xy))
            if off > _max_object_snap_m():
                too_far[name] = off
        elif sym.layer == "place":
            select(name)
        elif sym.layer == "region":
            select(name)
            if region_places is None:
                region_places = {}
                for _, place, region in generate_place_containment(G):
                    region_places.setdefault(region, []).append(place)
            reachable = [
                p
                for p in region_places.get(name, [])
                if p in by_name
                and pddl_symbol_to_node_symbol(G, p).value
                in planner.node_value_to_position
            ]
            if name not in explored:
                xy = {
                    p: planner.node_value_to_position[
                        pddl_symbol_to_node_symbol(G, p).value
                    ]
                    for p in reachable
                }
                centre = np.asarray(
                    G.get_node(pddl_symbol_to_node_symbol(G, name)).attributes.position
                )[:2]
                # central places (an object put down there lands inside the region) ...
                chosen = sorted(
                    reachable, key=lambda p: float(np.linalg.norm(xy[p] - centre))
                )[:max_region_places]
                # ... and, so a visit does not drive to the middle, the member nearest each robot
                for r in robot_ids:
                    others = [p for p in reachable if p != start_place[r]]
                    if others:
                        pose = np.asarray(robot_states[r])[:2]
                        chosen.append(
                            min(
                                others,
                                key=lambda p: float(np.linalg.norm(xy[p] - pose)),
                            )
                        )
                reachable = list(dict.fromkeys(chosen))
            for p in reachable:
                select(p)
                region_membership.append((p, name))
    if missing:
        raise MissingSymbolError(
            sorted(missing, key=lambda m: m.name), raw_pddl_goal_string
        )

    add_symbol_positions(G, list(selected.values()))

    pois = [s for s in selected.values() if s.layer in ("place", "object")]
    init: List[tuple] = [("=", ("total-cost",), 0)]
    D = (
        planner.external_distance_matrix([s.position[:2] for s in pois])
        if len(pois) > 1
        else None
    )
    for i in range(len(pois)):
        for j in range(i + 1, len(pois)):
            s, t = pois[i].symbol, pois[j].symbol
            if {s, t} & (forbidden_symbols | set(too_far)):
                continue
            if tuple(sorted([s, t])) in forbidden_edges or not np.isfinite(D[i, j]):
                continue
            d = int(round(D[i, j] * DISTANCE_SCALE))
            init.append(("connected", s, t))
            init.append(("=", ("distance", s, t), d))
            init.append(("=", ("distance", t, s), d))
    init += [("object-in-place", o, p) for o, p in sorted(object_place.items())]
    init += [("place-in-region", p, r) for p, r in region_membership]
    init += [("at-poi", r, start_place[r]) for r in robot_ids]

    symbols = list(selected.values())
    for rid in robot_ids:
        symbols.append(
            PddlSymbol(rid, "robot", [], position=np.array(robot_states[rid][:2]))
        )
    inspected = world_state.inspected if world_state is not None else set()
    init += [
        ("suspicious", s.symbol)
        for s in symbols
        if s.layer == "object" and s.symbol not in inspected
    ]
    if world_state is not None:
        init = apply_world_state(init, world_state, symbols, robot_ids)

    pddl_objects = generate_objects(symbols)
    pddl_objects["robot"] = robot_ids
    problem_str = PddlProblem(
        name="multi-robot-problem",
        domain="region-object-rearrangement-domain-multirobot-fd",
        objects=pddl_objects,
        initial_facts=tuple(init),
        goal=goal_pddl,
        optimizing=True,
    ).to_string()
    logger.info(
        "Goal-relevant multi-robot problem: %d points of interest (of %d symbols), %d robots",
        len(pois),
        len(all_symbols),
        len(robot_ids),
    )
    _dump_problem(problem_str)
    failure_hint = _failure_hint(
        raw_pddl_goal_string, init, symbols, robot_states, too_far
    )
    return problem_str, symbols, failure_hint, planner


@dispatch
def ground_problem(
    domain: MultiRobotPddlDomain,
    dsg: spark_dsg.DynamicSceneGraph,
    robot_states: Mapping,
    goal: PddlGoal,
    feedback: Any = None,
) -> MultiRobotWrapper[GroundedPddlProblem]:
    logger.warning(f"Grounding PDDL Problem {domain.domain_name}")

    pddl_compliant_robot_states = {k.lower(): v for k, v in robot_states.items()}
    world_state = getattr(goal, "world_state", None)
    check_goal_symbols(
        goal.pddl_goal,
        predicate_argument_types(domain),
        pddl_compliant_robot_states,
        _unique_symbols(normalized(extract_all_symbols(dsg))),
    )
    # An additional task leaves busy robots alone: no plan, not even an empty
    # one, so whatever they are doing goes on.
    busy = unavailable_robots(
        goal.constraints, world_state, goal_tokens(goal.pddl_goal)
    ) & set(pddl_compliant_robot_states)
    if busy:
        logger.info("Additional task: leaving busy robots alone: %s", sorted(busy))
    # How much of the scene graph to encode (see DEFAULT_SCENE_SCOPE).
    scene_scope = getattr(domain, "scene_scope", None) or DEFAULT_SCENE_SCOPE
    layer_planner = None
    match domain.domain_name:
        # case "goto-object-domain-multirobot-fd":
        #     pddl_problem, symbols = generate_multirobot_inspection_pddl(
        #         dsg, goal.pddl_goal, robot_states
        #     )

        case "region-object-rearrangement-domain-multirobot-fd" if (
            scene_scope == "goal_relevant"
        ):
            pddl_problem, symbols, failure_hint, layer_planner = (
                generate_goal_relevant_multirobot_pddl(
                    dsg,
                    goal.pddl_goal,
                    planning_robot_states(
                        pddl_compliant_robot_states, goal.constraints, busy
                    ),
                    constraints=goal.constraints,
                    world_state=world_state,
                )
            )
        case "region-object-rearrangement-domain-multirobot-fd":
            # Robots left out by use-robots get no actions: an empty plan.
            pddl_problem, symbols, failure_hint = generate_multirobot_region_pddl(
                dsg,
                goal.pddl_goal,
                planning_robot_states(
                    pddl_compliant_robot_states, goal.constraints, busy
                ),
                constraints=goal.constraints,
                world_state=world_state,
            )
            # logger.warning(f"!!!!!!!!!!!!!!pddl_problem: {pddl_problem}")
        case _:
            raise NotImplementedError(
                f"I don't know how to ground a domain of type {domain.domain_name}!"
            )
    if busy:
        failure_hint = "; ".join(
            filter(
                None,
                [
                    failure_hint,
                    f"only free robots were planned for (busy: "
                    f"{', '.join(sorted(busy))})",
                ],
            )
        )

    symbol_dict = {s.symbol: s for s in symbols}

    # TODO: We don't actually want to rely on the robot_states, because
    # robot_states may contain information about robots we actually don't care
    # about for planning purposes.  Instead, we probably want this information
    # passed as part of the goal.

    valid_robot_names = [
        name
        for name, pose in robot_states.items()
        if pose is not None and name.lower() not in busy
    ]
    wrapper = MultiRobotWrapper(
        valid_robot_names,
        GroundedPddlProblem(
            domain,
            pddl_problem,
            symbol_dict,
            failure_hint,
            layer_planner,
            ANYTIME_SEARCH_S if layer_planner is not None else 0.0,
        ),
    )
    for outer_name in valid_robot_names:
        inner_name = outer_name.lower()
        wrapper.set_name_remap(outer_name, inner_name)
    return wrapper
