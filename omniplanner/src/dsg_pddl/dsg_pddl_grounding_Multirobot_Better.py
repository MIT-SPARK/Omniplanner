import logging
from typing import Dict, List, Tuple, Any

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
from omniplanner.omniplanner import RobotWrapper
from dsg_pddl.pddl_utils import pddl_char_to_dsg_char, lisp_string_to_ast

from dsg_pddl.dsg_pddl_grounding import (
    normalize_symbol,
    normalize_symbols,
    generate_objects,
    add_symbol_positions,
    generate_place_containment,
    symbol_connectivity_to_pddl,
    simplify,
    generate_dense_region_symbol_connectivity,
    generate_object_containment,
)
from omniplanner.tsp import LayerPlanner

import os
import time

logger = logging.getLogger(__name__)



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

def generate_dense_region_symbol_connectivity_multirobot(G, symbols, robot_states):
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
    start_connection_threshold = 3
    for robot_id in robot_states.keys():
        start_symbol_key = f"pstart{robot_id}"
        if start_symbol_key in symbol_lookup:
            
            start_symbol = symbol_lookup[start_symbol_key]
            start_position = start_symbol.position

            for s in symbols:
                if s.symbol.startswith("pstart"):  # Skip other robot start positions
                    continue
                d = layer_planner.get_external_distance(start_position, s.position)
                if d < start_connection_threshold:
                    edges.append((start_symbol, s, d))

    return edges

def extract_all_symbols(G: spark_dsg.DynamicSceneGraph) -> List[PddlSymbol]:
    symbols: List[PddlSymbol] = []

    # Places (use 2D layer if available; fallback to numeric layer id 20 used elsewhere)
    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    for node in places_layer.nodes:
        symbols.append(PddlSymbol(normalize_symbol(node.id.str(True)), "place", []))

    # Objects
    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        symbols.append(PddlSymbol(normalize_symbol(node.id.str(True)), "object", []))

    # Regions
    for node in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        symbols.append(PddlSymbol(normalize_symbol(node.id.str(True)), "region", []))

    return symbols


# Use a multirobot-specific connectivity generator (kept here due to different thresholds and lack of pstart)
def generate_dense_symbol_connectivity(
    G: spark_dsg.DynamicSceneGraph, symbols: List[PddlSymbol]
) -> List[Tuple[PddlSymbol, PddlSymbol, float]]:
    """Dense but pruned connectivity among symbols using DSG distances."""
    symbol_lookup = {s.symbol: s for s in symbols}

    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    edges: List[Tuple[PddlSymbol, PddlSymbol, float]] = []

    # Place <-> Place based on actual layer edges
    for node in places_layer.nodes:
        p1 = np.array(node.attributes.position[:2])
        s1 = symbol_lookup.get(normalize_symbol(node.id.str(True)))
        for neighbor in node.siblings():
            if node.id.value < neighbor:
                continue
            n = G.get_node(neighbor)
            if n is None:
                continue
            p2 = np.array(n.attributes.position[:2])
            s2 = symbol_lookup.get(normalize_symbol(n.id.str(True)))
            if s1 is None or s2 is None:
                continue
            dist = float(np.linalg.norm(p1 - p2))
            edges.append((s1, s2, dist))

    # Object <-> Object and Object <-> Place within thresholds using external distance
    from omniplanner.tsp import LayerPlanner

    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)

    object_symbols = [s for s in symbols if s.layer == "object"]
    place_symbols = [s for s in symbols if s.layer == "place"]

    # Object-Object (tight threshold)
    for i, si in enumerate(object_symbols):
        for sj in object_symbols[i + 1 :]:
            d = layer_planner.get_external_distance(si.position, sj.position)
            if d <= 3.0:
                edges.append((si, sj, float(d)))

    # Object-Place (looser threshold)
    for so in object_symbols:
        for sp in place_symbols:
            d = layer_planner.get_external_distance(so.position, sp.position)
            if d <= 10.0:
                edges.append((so, sp, float(d)))

    return edges



def nearest_place_for_position(place_symbols: List[PddlSymbol], pos: np.ndarray) -> str:
    best_name = place_symbols[0].symbol
    best_d = float("inf")
    for p in place_symbols:
        d = float(np.linalg.norm(p.position - pos))
        if d < best_d:
            best_d = d
            best_name = p.symbol
    return best_name


# Multirobot init wrapper that reuses shared connectivity and containment helpers
def generate_dense_region_init_multirobot(
    G: spark_dsg.DynamicSceneGraph,
    symbols_of_interest: List[PddlSymbol],
    robot_states: Dict[str, np.ndarray],
) -> List[tuple]:
    connectivity = generate_dense_region_symbol_connectivity_multirobot(G, symbols_of_interest, robot_states)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl: List[tuple] = [("=", ("total-cost",), 0)]
    initial_pddl += connectivity_pddl

    # Containment relations (objects and places -> regions)
    initial_pddl += generate_object_containment(G)
    initial_pddl += generate_place_containment(G)

    # Add each robot's starting place using the pstart{robot_id} symbols
    for robot_id in robot_states.keys():
        start_symbol_key = f"pstart{robot_id}"
        initial_pddl.append(("at-poi", robot_id, start_symbol_key))

    return initial_pddl



def filter_goal_for_available_objects(goal_string: str, available_objects: List[str]) -> str:
    """Filter goal string to only include objects that are available in the problem."""
    import re
    
    # Extract object names from goal string (e.g., o2, o3, o21, etc.)
    object_pattern = r'\bo\d+\b'
    goal_objects = re.findall(object_pattern, goal_string)
    
    # Filter to only include available objects
    available_goal_objects = [obj for obj in goal_objects if obj in available_objects]
    
    # Create new goal string with only available objects
    if not available_goal_objects:
        return "(and)"  # Empty goal if no objects available
    
    # For safety goals, create individual safe predicates
    safe_goals = [f"(safe {obj})" for obj in available_goal_objects]
    return f"(and {' '.join(safe_goals)})"



def generate_multirobot_inspection_pddl(
    G: spark_dsg.DynamicSceneGraph,
    raw_pddl_goal_string: str,
    robot_states: Dict[str, np.ndarray],
) -> Tuple[str, List[PddlSymbol]]:
    """Generate a multi-robot PDDL problem for domain goto-object-domain-multirobot-fd."""
    # Collect all places/objects and positions
    all_symbols = extract_all_symbols(G)
    add_symbol_positions(G, all_symbols)

    all_places = [s for s in all_symbols if s.layer == "place"]
    all_objects = [s for s in all_symbols if s.layer == "object"]
    print("length of all_objects: ", len(all_objects))
    print("length of all_places: ", len(all_places))
    # Apply optional limits to keep the problem manageable
    max_places = int(os.environ.get("MR_MAX_PLACES", "20"))
    max_objects = int(os.environ.get("MR_MAX_OBJECTS", "20"))
    place_symbols = all_places[: max_places if max_places > 0 else None]
    object_symbols = all_objects[: max_objects if max_objects > 0 else None]

    # Filter goal string to only include available objects
    available_object_names = [o.symbol for o in object_symbols]
    print(f"DEBUG: Available objects: {available_object_names}")
    filtered_goal_string = filter_goal_for_available_objects(raw_pddl_goal_string, available_object_names)
    print(f"DEBUG: Filtered goal result: {filtered_goal_string}")
    
    logger.info(f"Original goal: {raw_pddl_goal_string}")
    logger.info(f"Filtered goal: {filtered_goal_string}")
    logger.info(f"Available objects: {available_object_names}")

    # Selected symbol list for connectivity
    selected_symbols: List[PddlSymbol] = place_symbols + object_symbols
    # Ensure robots are present in the symbols map for downstream planners
    for rid, pose in robot_states.items():
        selected_symbols.append(PddlSymbol(rid, "robot", [], position=np.array(pose[:2])))

    # Objects section
    robot_ids = list(robot_states.keys())

    # Init facts
    connectivity = generate_dense_symbol_connectivity(G, selected_symbols)
    connected_facts = symbol_connectivity_to_pddl(connectivity)

    # Map objects to nearest place (object-at)
    object_at_facts: List[Tuple] = []
    for obj in object_symbols:
        nearest_place = nearest_place_for_position(place_symbols, obj.position)
        object_at_facts.append(("object-at", obj.symbol, nearest_place))

    # Suspicious all objects by default
    suspicious_facts = [("suspicious", o.symbol) for o in object_symbols]

    # Robot starts at nearest places to their given 2D states
    at_poi_facts = []
    for rid, pose in robot_states.items():
        nearest_place = nearest_place_for_position(place_symbols, np.array(pose[:2]))
        at_poi_facts.append(("at-poi", rid, nearest_place))

    # Compose PDDL
    def fmt_objects_line(names: List[str], typ: str) -> str:
        if not names:
            return ""
        return " ".join(names) + f" - {typ}\n"

    objects_section = ""
    objects_section += fmt_objects_line(robot_ids, "robot")
    objects_section += fmt_objects_line([p.symbol for p in place_symbols], "place")
    objects_section += fmt_objects_line([o.symbol for o in object_symbols], "dsg_object")

    # Serialize init facts
    def fmt_sexpr(expr: Any) -> str:
        if isinstance(expr, tuple):
            head = expr[0]
            args = expr[1:]
            return f"({head} {' '.join(fmt_sexpr(a) for a in args)})"
        return str(expr)

    def fmt_fact(f: Tuple) -> str:
        head = f[0]
        if head == "=":
            # Expect (= (<function> ...) <value>)
            return f"(= {fmt_sexpr(f[1])} {fmt_sexpr(f[2])})"
        return f"({head} {' '.join(fmt_sexpr(x) for x in f[1:])})"

    init_facts: List[str] = []
    # initialize total-cost
    init_facts.append("(= (total-cost) 0)")
    init_facts += [fmt_fact(f) for f in at_poi_facts]
    init_facts += [fmt_fact(f) for f in connected_facts]
    init_facts += [fmt_fact(f) for f in object_at_facts]
    init_facts += [fmt_fact(f) for f in suspicious_facts]

    problem_str = """
 (define (problem multi-robot-problem)
   (:domain goto-object-domain-multirobot-fd)
   (:objects
{objects}
   )
   (:init
{init}
   )
   (:goal
     {goal}
   )
   (:metric minimize (total-cost))
 )
""".strip().format(
        objects=("\n" + objects_section.strip() if objects_section.strip() else "").replace("\n", "\n  "),
        init=("\n" + "\n".join(init_facts) if init_facts else "").replace("\n", "\n  "),
        goal=filtered_goal_string,
    )

    # Persist copies of the generated multi-robot PDDL problem
    try:
        dump_dir = os.environ.get("PDDL_DUMP_DIR", os.path.join(os.getcwd(), "pddl_dumps"))
        os.makedirs(dump_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        persistent_problem = os.path.join(dump_dir, f"{ts}_mr_problem.pddl")
        latest_problem = os.path.join(dump_dir, "mr_problem_latest.pddl")
        with open(persistent_problem, "w") as f:
            f.write(problem_str)
        with open(latest_problem, "w") as f:
            f.write(problem_str)
        logger.info(f"Saved multi-robot PDDL to {persistent_problem}")
    except Exception as e:
        logger.warning(f"Failed to persist multi-robot PDDL dump: {e}")

    return problem_str, selected_symbols 



def generate_multirobot_region_pddl(
    G: spark_dsg.DynamicSceneGraph,
    raw_pddl_goal_string: str,
    robot_states: np.ndarray,
) -> Tuple[str, List[PddlSymbol]]:
    """Generate a multi-robot PDDL problem for domain region-object-rearrangement-domain-multirobot-fd."""
    # Collect all places/objects/regions and positions
    symbols = extract_all_symbols(G)
    normalize_symbols(symbols)
    
    symbols_of_interest=symbols
    print("robot_states: ", robot_states)
    for robot in robot_states.keys():
        initial_position = robot_states[robot]
        start_place_symbol= PddlSymbol(
            "pstart"+str(robot), "place", ["at-poi"], position=initial_position[:2]
        )
        print("start_place_symbol: ", start_place_symbol)
        print("initial_position: ", initial_position)
        symbols_of_interest= [start_place_symbol] + symbols_of_interest

    add_symbol_positions(G, symbols_of_interest)


    
    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)
    goal_pddl = simplify(parsed_pddl_goal)
    
    # Robot starts at nearest places to their given 2D states
    robot_ids = list(robot_states.keys())
    # Build init facts via shared helpers
    init_facts_tuples: List[tuple] = generate_dense_region_init_multirobot(
        G, symbols_of_interest, robot_states
    )

    # Ensure robot symbols exist (with positions) for downstream planners
    for rid, pose in robot_states.items():
        symbols_of_interest.append(PddlSymbol(rid, "robot", [], position=np.array(pose[:2])))

    # Add suspicious facts for all objects
    object_symbols = [s for s in symbols_of_interest if s.layer == "object"]
    init_facts_tuples += [("suspicious", o.symbol) for o in object_symbols]

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

    try:
        dump_dir = os.environ.get("PDDL_DUMP_DIR", os.path.join(os.getcwd(), "pddl_dumps"))
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

    return problem_str, symbols_of_interest


@dispatch
def ground_problem(
    domain: PddlDomain,
    dsg: spark_dsg.DynamicSceneGraph,
    robot_states: dict,
    goal: PddlGoal,
    feedback: Any = None,
    Multirobot: bool = True,
) -> RobotWrapper[GroundedPddlProblem]:
    logger.info(f"Grounding PDDL Problem {domain.domain_name}")
    print("goal.robot_id: ", goal.robot_id)
    print("robot_states: ", robot_states)
    start = robot_states[goal.robot_id][:2]

    match domain.domain_name:
        case "goto-object-domain-multirobot-fd":
            pddl_problem, symbols = generate_multirobot_inspection_pddl(dsg, goal.pddl_goal, robot_states)

        case "region-object-rearrangement-domain-multirobot-fd":
            pddl_problem, symbols = generate_multirobot_region_pddl(dsg, goal.pddl_goal, robot_states)
        case _:
            raise NotImplementedError(
                f"I don't know how to ground a domain of type {domain.domain_name}!"
            )

    symbol_dict = {s.symbol: s for s in symbols}
    return RobotWrapper(
        goal.robot_id, GroundedPddlProblem(domain, pddl_problem, symbol_dict)
    )
