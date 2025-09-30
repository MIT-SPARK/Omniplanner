import logging
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


def generate_symbol_connectivity(G, symbols):
    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)

    connections = []
    for si in symbols:
        for sj in symbols:
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
    try:
        places_layer_2d = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer_2d = G.get_layer(20)

    places_layer = G.get_layer(spark_dsg.DsgLayers.PLACES)

    containments = []

    place_centers = []
    place_nodes = []
    for node in places_layer.nodes:
        place_centers.append(node.attributes.position)
        place_nodes.append(node)
    place_centers = np.array(place_centers)
    if len(place_centers) == 0:
        # there are no 3d places, so we don't need to worry about place/region containment
        return []

    for node in places_layer_2d.nodes:
        closest_idx = np.argmin(
            np.linalg.norm(place_centers - node.attributes.position, axis=1)
        )
        closest_place = place_nodes[closest_idx]
        parent = closest_place.get_parent()
        if parent is not None:
            containments.append(
                (
                    "place-in-region",
                    normalize_symbol(node.id.str(True)),
                    normalize_symbol(spark_dsg.NodeSymbol(parent).str(True)),
                )
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

    return place_symbols + object_symbols


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
            position = G.get_node(ns).attributes.position[:2]
            if position is None:
                raise Exception(f"Could not find node {ns} in DSG")
            s.position = position
    return symbols


def normalize_symbols(symbols):
    for s in symbols:
        s.symbol = normalize_symbol(s)


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

    place_symbols = []
    for node in places_layer.nodes:
        place_symbols.append(PddlSymbol(node.id.str(True), "place", []))

    region_symbols = []
    for node in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        region_symbols.append(PddlSymbol(node.id.str(True), "region", []))

    object_symbols = []
    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        object_symbols.append(PddlSymbol(node.id.str(True), "object", []))

    return place_symbols + object_symbols + region_symbols


def generate_rearrangement_pddl(G, raw_pddl_goal_string, initial_position):
    problem_name = "object-rearrangement-domain"
    problem_domain = "object-rearrangement-domain"

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


def generate_region_pddl(G, raw_pddl_goal_string, initial_position):
    problem_name = "region-object-rearrangement-domain"
    problem_domain = "region-object-rearrangement-domain"

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
    robot_states: dict,
    goal: PddlGoal,
    feedback: Any = None,
) -> RobotWrapper[GroundedPddlProblem]:
    logger.info(f"Grounding PDDL Problem {domain.domain_name}")

    start = robot_states[goal.robot_id][:2]

    # TODO: TBD whether we want to check the domain here and choose how
    # to instantiate the PDDL problem, or if that should be in a separately
    # ground_problem function.
    match domain.domain_name:
        case "goto-object-domain":
            pddl_problem, symbols = generate_inspection_pddl(dsg, goal.pddl_goal, start)
        case "object-rearrangement-domain":
            pddl_problem, symbols = generate_rearrangement_pddl(
                dsg, goal.pddl_goal, start
            )
        case "region-object-rearrangement-domain":
            pddl_problem, symbols = generate_region_pddl(dsg, goal.pddl_goal, start)
        case _:
            raise NotImplementedError(
                f"I don't know how to ground a domain of type {domain.domain_name}!"
            )

    symbol_dict = {s.symbol: s for s in symbols}
    return RobotWrapper(
        goal.robot_id, GroundedPddlProblem(domain, pddl_problem, symbol_dict)
    )