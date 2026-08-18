import logging
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import spark_dsg
from plum import dispatch

from dsg_pddl.pddl_grounding import (
    GroundedPddlProblem,
    PddlDomain,
    PddlGoal,
    PddlProblem,
    PddlSymbol,
)
from dsg_pddl.pddl_utils import extract_facts, lisp_string_to_ast, pddl_char_to_dsg_char
from omniplanner.omniplanner import RobotWrapper
from omniplanner.tsp import LayerPlanner

logger = logging.getLogger(__name__)

# How much of the scene graph to encode into the PDDL problem.
#   "goal_relevant" (default): only symbols referenced by the grounded goal,
#       their DSG containers, the robot start, and pairwise navigation distances
#       among them. Keeps the Fast Downward translation/instantiation tiny.
#   "full": legacy behavior -- encode every object/place/region in the scene.
# Default chosen per-domain in ground_problem; this env var overrides it.
DEFAULT_SCENE_SCOPE = os.getenv("OMNIPLANNER_SCENE_SCOPE", "goal_relevant")

# Max representative places to include per goal-referenced region (a region goal
# only needs to visit one of its places; a few gives the planner some slack).
GOAL_RELEVANT_MAX_REGION_PLACES = int(
    os.getenv("OMNIPLANNER_MAX_REGION_PLACES", "3")
)


def generate_symbol_connectivity(G, symbols):
    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)

    # De-duplicate by pddl symbol name: a goal that mentions the same object
    # twice (legal PDDL — e.g. two plan stages resolving to one instance)
    # otherwise emits each connectivity pair twice, and Fast Downward's
    # translator hard-errors on "PNE distance(a, b) ... specified twice in
    # initial state specification".
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s.symbol not in seen:
            seen.add(s.symbol)
            unique_symbols.append(s)

    connections = []
    for si in unique_symbols:
        for sj in unique_symbols:
            if si <= sj:
                continue

            distance = layer_planner.get_external_distance(si.position, sj.position)
            connections.append((si, sj, distance))

    return connections


def symbol_connectivity_to_pddl(connectivity):
    connections_init = []

    for info_s, info_t, dist in connectivity:
        s = info_s.symbol
        t = info_t.symbol
        # A disconnected pair (no path in the place graph) comes back as inf
        # from LayerPlanner.get_shortest_distance. Emitting a fabricated
        # connected/distance fact would both crash int(inf) and lie about
        # reachability, so skip the pair: the POI stays reachable via any other
        # finite connection, and a truly isolated POI becomes correctly
        # unreachable.
        if not np.isfinite(dist):
            logger.warning("Skipping disconnected POI pair %s <-> %s (inf distance)", s, t)
            continue
        d = int(dist)

        connected = ("connected", s, t)
        distance = ("=", ("distance", s, t), d)
        distance_rev = ("=", ("distance", t, s), d)

        connections_init.append(connected)
        connections_init.append(distance)
        connections_init.append(distance_rev)

    return connections_init


def generate_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_symbol_connectivity(G, symbols_of_interest)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl
    return initial_pddl


def explicit_edges_from_layer(
    symbol_lookup: dict, G: spark_dsg.DynamicSceneGraph, layer: spark_dsg.LayerView
):
    """Get the edges corresponding to layer's edges"""
    edges = []
    for node in layer.nodes:
        p1 = node.attributes.position
        normalized_symbol = symbol_lookup[normalize_symbol(node.id.str(True))]
        for neighbor in node.siblings():
            if node.id.value < neighbor:
                continue
            n = G.get_node(neighbor)
            p2 = n.attributes.position
            normalized_symbol2 = symbol_lookup[normalize_symbol(n.id.str(True))]
            edges.append(
                (normalized_symbol, normalized_symbol2, np.linalg.norm(p1 - p2))
            )
    return edges


def implicit_edges_from_layers(
    symbol_lookup: dict,
    layer1: spark_dsg.LayerView,
    layer2: spark_dsg.LayerView,
    same_layer,
    connection_threshold,
    layer_planner=None,
):
    edges = []
    for n1 in layer1.nodes:
        p1 = n1.attributes.position
        normalized_symbol = symbol_lookup[normalize_symbol(n1.id.str(True))]
        for n2 in layer2.nodes:
            if same_layer and n1.id.value <= n2.id.value:
                continue
            p2 = n2.attributes.position
            d = np.linalg.norm(p1 - p2)
            if d > connection_threshold:
                continue

            if layer_planner is not None:
                d = layer_planner.get_external_distance(p1[:2], p2[:2])
                if d > connection_threshold:
                    continue

            normalized_symbol2 = symbol_lookup[normalize_symbol(n2.id.str(True))]
            edges.append((normalized_symbol, normalized_symbol2, d))
    return edges


def generate_dense_symbol_connectivity(G, symbols):
    symbol_lookup = {s.symbol: s for s in symbols}

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

    start_symbol = symbol_lookup["pstart"]
    start_position = start_symbol.position

    # Connection between starting place and other symbols
    start_connection_threshold = 3
    for s in symbols:
        if s.symbol == "pstart":
            continue

        d = layer_planner.get_external_distance(start_position, s.position)
        if d < start_connection_threshold:
            edges.append((start_symbol, s, d))

    return edges


def generate_dense_region_symbol_connectivity(G, symbols):
    symbol_lookup = {s.symbol: s for s in symbols}

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

    # Region <-> Region Edges #TODO: currently, we don't actually utilize edges between regions?
    # region_layer =  G.get_layer(spark_dsg.DsgLayers.ROOMS)
    # edges += implicit_edges_from_layers(symbol_lookup, region_layer, region_layer, True, 20)

    start_symbol = symbol_lookup["pstart"]
    start_position = start_symbol.position

    # Connection between starting place and other symbols
    start_connection_threshold = 3
    for s in symbols:
        if s.symbol == "pstart":
            continue

        d = layer_planner.get_external_distance(start_position, s.position)
        if d < start_connection_threshold:
            edges.append((start_symbol, s, d))

    return edges


def generate_object_containment(G):
    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    containments = []

    centers = []
    symbols = []
    for node in places_layer.nodes:
        centers.append(node.attributes.position)
        symbols.append(normalize_symbol(node.id.str(True)))
    centers = np.array(centers)

    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        closest_idx = np.argmin(
            np.linalg.norm(centers - node.attributes.position, axis=1)
        )
        closest_place = symbols[closest_idx]
        containments.append(
            ("object-in-place", normalize_symbol(node.id.str(True)), closest_place)
        )

    return containments


def generate_place_containment(G):
    """Emit (place-in-region ?p ?reg) for every mesh place we can attribute to a room.

    Mesh places usually parent straight to a room, so we take that edge when it
    exists. Graphs that only connect rooms to the 3-D PLACES layer fall back to
    the nearest 3-D place and borrow its parent.
    """
    try:
        places_layer_2d = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer_2d = G.get_layer(20)

    containments = []
    unparented = []

    for node in places_layer_2d.nodes:
        if node.has_parent():
            containments.append(
                (
                    "place-in-region",
                    normalize_symbol(node.id.str(True)),
                    normalize_symbol(spark_dsg.NodeSymbol(node.get_parent()).str(True)),
                )
            )
        else:
            unparented.append(node)

    if unparented:
        places_layer_3d = G.get_layer(spark_dsg.DsgLayers.PLACES)
        place_centers = []
        place_nodes = []
        for node in places_layer_3d.nodes:
            place_centers.append(node.attributes.position)
            place_nodes.append(node)

        if place_centers:
            place_centers = np.array(place_centers)
            for node in unparented:
                closest_idx = np.argmin(
                    np.linalg.norm(place_centers - node.attributes.position, axis=1)
                )
                parent = place_nodes[closest_idx].get_parent()
                if parent is not None:
                    containments.append(
                        (
                            "place-in-region",
                            normalize_symbol(node.id.str(True)),
                            normalize_symbol(spark_dsg.NodeSymbol(parent).str(True)),
                        )
                    )

    n_regions = G.get_layer(spark_dsg.DsgLayers.ROOMS).num_nodes()
    if n_regions > 0 and not containments:
        # Region goals derive through place-in-region; with no facts every
        # (explored-region ?r) is vacuously true and the goal is silently free.
        logger.warning(
            "No place-in-region facts derivable for %d region(s): region goals "
            "will be vacuously satisfied. Check that mesh places parent to rooms.",
            n_regions,
        )

    return containments


def generate_dense_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_dense_symbol_connectivity(G, symbols_of_interest)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl

    containment_relations = generate_object_containment(G)
    initial_pddl += containment_relations
    return initial_pddl


def generate_dense_region_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_dense_region_symbol_connectivity(G, symbols_of_interest)

    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl

    containment_relations = generate_object_containment(G)
    containment_relations += generate_place_containment(G)
    initial_pddl += containment_relations
    return initial_pddl


def extract_symbols_of_interest(G, pddl_goal):
    place_facts = extract_facts(pddl_goal, "visited-place")
    place_facts += extract_facts(pddl_goal, "at-place")

    object_facts = extract_facts(pddl_goal, "visited-object")
    object_facts += extract_facts(pddl_goal, "at-object")

    place_symbols = [PddlSymbol(f[1], "place", []) for f in place_facts]
    object_symbols = [PddlSymbol(f[1], "object", []) for f in object_facts]

    # De-duplicate by symbol name (order-preserving): a goal may legally
    # mention the same object twice — e.g. "(and (visited-object o9)
    # (visited-object o9))" when two plan stages resolve to one instance —
    # but downstream every symbol becomes an :objects declaration and a
    # connectivity-matrix row, and Fast Downward's translator hard-errors on
    # the duplicated "PNE distance(a, b) specified twice" init facts.
    seen = set()
    unique = []
    for s in place_symbols + object_symbols:
        if s.symbol not in seen:
            seen.add(s.symbol)
            unique.append(s)
    return unique


def simplify(pddl):
    return pddl


def add_symbol_positions(G, symbols):
    for s in symbols:
        if s.position is not None:
            continue
        else:
            pddl_symbol_char = s.symbol[0]
            dsg_symbol_char = pddl_char_to_dsg_char(pddl_symbol_char)
            ns = spark_dsg.NodeSymbol(dsg_symbol_char, int(s.symbol[1:]))
            if not G.has_node(ns):
                # open-vocab regions use a lowercase r prefix
                ns = spark_dsg.NodeSymbol(pddl_symbol_char, int(s.symbol[1:]))
            # A copy: the attribute array is a view into the graph's storage.
            position = np.array(G.get_node(ns).attributes.position[:2])
            if position is None:
                raise Exception(f"Could not find node {ns} in DSG")
            s.position = position
    return symbols


def normalize_symbols(symbols):
    for s in symbols:
        s.symbol = normalize_symbol(s)


def normalized(symbols):
    normalize_symbols(symbols)
    return symbols


def normalize_symbol(symbol):
    if isinstance(symbol, str):
        return symbol.lower()
    else:
        return symbol.symbol.lower()


def generate_objects(symbols):
    type_dict = {"place": [], "dsg_object": [], "region": []}
    for s in symbols:
        if s.layer == "place":
            type_dict["place"].append(s.symbol)
        elif s.layer == "object":
            type_dict["dsg_object"].append(s.symbol)
        elif s.layer == "region":
            type_dict["region"].append(s.symbol)

    return type_dict


# PDDL tokens that are NOT scene symbols: logical operators, type names,
# predicate/function names from the rearrangement domains.
_NON_SYMBOL_TOKENS = {
    "and", "or", "not", "imply", "when", "exists", "forall", "=", "-",
    "increase", "decrease",
    "region", "place", "dsg_object", "point-of-interest", "object",
    "at-poi", "connected", "suspicious", "at-object", "at-place", "in-region",
    "holding", "hand-full", "object-in-place", "place-in-region",
    "visited-poi", "visited-place", "visited-object", "visited-region",
    "safe", "distance", "total-cost",
}


def collect_goal_symbol_names(ast):
    """Return the set of scene-symbol names referenced anywhere in a goal AST.

    Walks the parsed goal and collects leaf tokens that are not logical
    operators, type names, predicate/function names, or quantifier variables
    (``?x``). Works for ground goals (the usual LLM output) as well as goals
    that mix predicates and concrete symbols.
    """
    names = set()

    def walk(node):
        if isinstance(node, (tuple, list)):
            for child in node:
                walk(child)
        elif isinstance(node, str):
            tok = node.lower()
            if tok in _NON_SYMBOL_TOKENS or tok.startswith("?"):
                return
            names.add(tok)

    walk(ast)
    return names


def get_places_layer(G):
    try:
        return G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        return G.get_layer(20)


def generate_goal_relevant_pddl(
    G,
    raw_pddl_goal_string,
    initial_position,
    problem_name,
    problem_domain,
    max_region_places=GOAL_RELEVANT_MAX_REGION_PLACES,
):
    """Build a PDDL problem containing only goal-relevant symbols.

    Instead of encoding the whole scene graph, this includes the robot start,
    the objects/places/regions named by the grounded goal, the current place of
    each referenced object (so it can be picked), and a few representative
    places per referenced region (so it can be visited). Navigation distances
    are computed pairwise among just these POIs over the full places graph, so a
    single ``goto-poi`` hop represents a full multi-hop traversal whose waypoints
    are expanded later at plan-compile time.
    """
    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)
    goal_pddl = simplify(parsed_pddl_goal)
    referenced = collect_goal_symbol_names(parsed_pddl_goal)

    all_symbols = extract_all_symbols(G)
    normalize_symbols(all_symbols)
    by_name = {s.symbol: s for s in all_symbols}

    # One LayerPlanner over the full places graph, then restricted so that
    # every nearest-place snap lands in the component reachable from the
    # robot start: saved DSGs contain island place components, and snapping
    # an object (or the distance matrix's anchors) to one silently drops its
    # connectivity facts below -- a disconnected, unsolvable problem for a
    # physically reachable goal (motion-tier v1, floor3 s71 NL).
    layer_planner = LayerPlanner(
        G, spark_dsg.DsgLayers.MESH_PLACES
    ).restricted_to_component(np.asarray(initial_position))
    reachable_place_values = set(layer_planner.node_ids)

    # Cache place symbol/position arrays for nearest-place lookups, limited
    # to the same reachable component the planner snaps to.
    places_layer = get_places_layer(G)
    place_syms = []
    place_pos = []
    for node in places_layer.nodes:
        if node.id.value not in reachable_place_values:
            continue
        place_syms.append(normalize_symbol(node.id.str(True)))
        place_pos.append(node.attributes.position[:2])
    place_pos = np.array(place_pos) if place_pos else np.zeros((0, 2))
    place_sym_to_pos = dict(zip(place_syms, place_pos))

    start_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    selected = {"pstart": start_symbol}
    object_current_place = {}
    region_membership = []  # (place_symbol, region_symbol)

    # region -> member places, built lazily only if a region is referenced.
    region_to_places = None

    def obj_position(obj_sym):
        ns = spark_dsg.NodeSymbol(
            pddl_char_to_dsg_char(obj_sym[0]), int(obj_sym[1:])
        )
        return G.get_node(ns).attributes.position[:2]

    for name in referenced:
        sym = by_name.get(name)
        if sym is None:
            logger.warning(
                "Goal references symbol '%s' not present in the scene graph; "
                "skipping it during grounding.",
                name,
            )
            continue
        selected.setdefault(name, sym)

        if sym.layer == "object":
            if len(place_syms) == 0:
                continue
            opos = obj_position(name)
            idx = int(np.argmin(np.linalg.norm(place_pos - opos, axis=1)))
            pcur = place_syms[idx]
            object_current_place[name] = pcur
            selected.setdefault(pcur, by_name[pcur])
        elif sym.layer == "region":
            if region_to_places is None:
                region_to_places = {}
                for fact in generate_place_containment(G):
                    _, p, r = fact
                    region_to_places.setdefault(r, []).append(p)
            members = region_to_places.get(name, [])
            members = sorted(
                members,
                key=lambda p: float(
                    np.linalg.norm(place_sym_to_pos[p] - initial_position)
                ),
            )[:max_region_places]
            for p in members:
                selected.setdefault(p, by_name[p])
                region_membership.append((p, name))

    add_symbol_positions(G, list(selected.values()))

    # Build connectivity over POIs (places + objects), reusing the
    # component-restricted LayerPlanner so every anchor is reachable.
    pois = [s for s in selected.values() if s.layer in ("place", "object")]
    init = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    if len(pois) > 1:
        D = layer_planner.external_distance_matrix([p.position for p in pois])
        for i in range(len(pois)):
            for j in range(i + 1, len(pois)):
                d = D[i, j]
                if not np.isfinite(d):
                    continue
                s = pois[i].symbol
                t = pois[j].symbol
                di = int(d)
                init.append(("connected", s, t))
                init.append(("=", ("distance", s, t), di))
                init.append(("=", ("distance", t, s), di))

    for o, p in object_current_place.items():
        init.append(("object-in-place", o, p))
    for p, r in region_membership:
        init.append(("place-in-region", p, r))

    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=generate_objects(list(selected.values())),
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), list(selected.values())


def generate_inspection_pddl(G, raw_pddl_goal_string, initial_position):
    problem_name = "goto-object-problem"
    problem_domain = "goto-object-domain"

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)
    goal_symbols_of_interest = extract_symbols_of_interest(G, parsed_pddl_goal)
    normalize_symbols(goal_symbols_of_interest)
    logger.info(f"Extracted goal_symbols of interest: {goal_symbols_of_interest}")

    # ideally we check the goal here and see if we can run a more specialized planner based on the simplified goal
    goal_pddl = simplify(parsed_pddl_goal)

    start_place_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    symbols_of_interest = [start_place_symbol] + goal_symbols_of_interest

    add_symbol_positions(G, symbols_of_interest)

    logger.info(f"generate_objects: {generate_objects(symbols_of_interest)}")
    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=generate_objects(symbols_of_interest),
        initial_facts=generate_init(G, symbols_of_interest, start_place_symbol),
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), symbols_of_interest


def extract_all_symbols(G):
    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    # Positions come from the node itself: looking a lowercased PDDL name back
    # up in the graph guesses the prefix's case, and a wrong guess failed the
    # whole plan ("missing node 'R(0)'") for a goal that named no region.
    def symbol(node, layer):
        position = np.array(node.attributes.position[:2], dtype=float)
        return PddlSymbol(node.id.str(True), layer, [], position=position)

    place_symbols = [symbol(n, "place") for n in places_layer.nodes]
    region_symbols = [
        symbol(n, "region") for n in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes
    ]
    object_symbols = [
        symbol(n, "object") for n in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes
    ]

    return place_symbols + object_symbols + region_symbols


def generate_rearrangement_pddl(
    G, raw_pddl_goal_string, initial_position, scene_scope=DEFAULT_SCENE_SCOPE
):
    problem_name = "object-rearrangement-domain"
    problem_domain = "object-rearrangement-domain"

    if scene_scope == "goal_relevant":
        return generate_goal_relevant_pddl(
            G, raw_pddl_goal_string, initial_position, problem_name, problem_domain
        )

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    all_symbols = extract_all_symbols(G)
    symbols = [s for s in all_symbols if s.layer in ["place", "object"]]
    normalize_symbols(symbols)

    # ideally we check the goal here and see if we can run a more specialized planner based on the simplified goal
    goal_pddl = simplify(parsed_pddl_goal)

    start_place_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    symbols_of_interest = [start_place_symbol] + symbols

    add_symbol_positions(G, symbols_of_interest)

    pddl_objects = generate_objects(symbols_of_interest)
    init = generate_dense_init(G, symbols_of_interest, start_place_symbol)

    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=pddl_objects,
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), symbols_of_interest


def generate_region_pddl(
    G, raw_pddl_goal_string, initial_position, scene_scope=DEFAULT_SCENE_SCOPE
):
    problem_name = "region-object-rearrangement-domain"
    problem_domain = "region-object-rearrangement-domain"

    if scene_scope == "goal_relevant":
        return generate_goal_relevant_pddl(
            G, raw_pddl_goal_string, initial_position, problem_name, problem_domain
        )

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    symbols = extract_all_symbols(G)
    normalize_symbols(symbols)

    # ideally we check the goal here and see if we can run a more specialized planner based on the simplified goal
    goal_pddl = simplify(parsed_pddl_goal)

    start_place_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    symbols_of_interest = [start_place_symbol] + symbols

    add_symbol_positions(G, symbols_of_interest)

    pddl_objects = generate_objects(symbols_of_interest)
    init = generate_dense_region_init(G, symbols_of_interest, start_place_symbol)

    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=pddl_objects,
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), symbols_of_interest


@dispatch
def ground_problem(
    domain: PddlDomain,
    dsg: spark_dsg.DynamicSceneGraph,
    robot_states: Mapping,
    goal: PddlGoal,
    feedback: Any = None,
) -> RobotWrapper[GroundedPddlProblem]:
    logger.info(f"Grounding PDDL Problem {domain.domain_name}")

    robot_pose = robot_states[goal.robot_id]
    if robot_pose is None:
        raise RuntimeError(
            f"Cannot plan for robot '{goal.robot_id}': no transform available "
            "(is the robot online and publishing TF?)"
        )
    start = robot_pose[:2]

    # How much of the scene graph to encode (see DEFAULT_SCENE_SCOPE). The
    # domain may carry a config-provided scope; otherwise fall back to default.
    scene_scope = getattr(domain, "scene_scope", None) or DEFAULT_SCENE_SCOPE

    # TODO: TBD whether we want to check the domain here and choose how
    # to instantiate the PDDL problem, or if that should be in a separately
    # ground_problem function.
    match domain.domain_name:
        case "goto-object-domain":
            # Inspection grounding is already goal-relevant by construction.
            pddl_problem, symbols = generate_inspection_pddl(dsg, goal.pddl_goal, start)
        case "object-rearrangement-domain":
            pddl_problem, symbols = generate_rearrangement_pddl(
                dsg, goal.pddl_goal, start, scene_scope=scene_scope
            )
        case "region-object-rearrangement-domain":
            pddl_problem, symbols = generate_region_pddl(
                dsg, goal.pddl_goal, start, scene_scope=scene_scope
            )
        case _:
            raise NotImplementedError(
                f"I don't know how to ground a domain of type {domain.domain_name}!"
            )

    symbol_dict = {s.symbol: s for s in symbols}
    return RobotWrapper(
        goal.robot_id, GroundedPddlProblem(domain, pddl_problem, symbol_dict)
    )
