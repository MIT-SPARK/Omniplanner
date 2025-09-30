import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import spark_dsg
from plum import dispatch

from dsg_pddl.dsg_pddl_grounding import (
    add_symbol_positions,
    explicit_edges_from_layer,
    extract_all_symbols,
    generate_object_containment,
    generate_objects,
    generate_place_containment,
    implicit_edges_from_layers,
    normalize_symbols,
    simplify,
    symbol_connectivity_to_pddl,
)
from dsg_pddl.pddl_grounding import (
    GroundedPddlProblem,
    MultiRobotPddlDomain,
    PddlGoal,
    PddlProblem,
    PddlSymbol,
)
from dsg_pddl.pddl_utils import lisp_string_to_ast
from omniplanner.omniplanner import MultiRobotWrapper
from omniplanner.tsp import LayerPlanner

logger = logging.getLogger(__name__)


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
    start_connection_threshold = 50
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
    connectivity = generate_dense_region_symbol_connectivity_multirobot(
        G, symbols_of_interest, robot_states
    )
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl: List[tuple] = [("=", ("total-cost",), 0)]
    initial_pddl += connectivity_pddl

    # Containment relations (objects and places -> regions)
    initial_pddl += generate_object_containment(G)
    initial_pddl += generate_place_containment(G)

    # Add each valid robot's starting place using the pstart{robot_id} symbols
    for robot_id, pose in robot_states.items():
        if pose is None:
            continue
        start_symbol_key = f"pstart{robot_id}"
        initial_pddl.append(("at-poi", robot_id, start_symbol_key))

    return initial_pddl


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


def generate_multirobot_region_pddl(
    G: spark_dsg.DynamicSceneGraph,
    raw_pddl_goal_string: str,
    robot_states: np.ndarray,
) -> Tuple[str, List[PddlSymbol]]:
    """Generate a multi-robot PDDL problem for domain region-object-rearrangement-domain-multirobot-fd."""
    # Collect all places/objects/regions and positions
    symbols = extract_all_symbols(G)
    normalize_symbols(symbols)

    symbols_of_interest = symbols
    for robot in robot_states.keys():
        initial_position = robot_states[robot]
        if initial_position is None:
            logger.warning(
                f"Skipping robot {robot} due to missing initial position (None)."
            )
            continue
        start_place_symbol = PddlSymbol(
            "pstart" + str(robot),
            "place",
            ["at-poi"],
            position=np.array(initial_position[:2]),
        )
        # print("start_place_symbol: ", start_place_symbol)
        # print("initial_position: ", initial_position)
        symbols_of_interest = [start_place_symbol] + symbols_of_interest

    add_symbol_positions(G, symbols_of_interest)

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)
    goal_pddl = simplify(parsed_pddl_goal)

    # Robot starts at nearest places to their given 2D states
    robot_ids = [rid for rid, pose in robot_states.items() if pose is not None]
    # Build init facts via shared helpers
    init_facts_tuples: List[tuple] = generate_dense_region_init_multirobot(
        G, symbols_of_interest, robot_states
    )

    # Ensure robot symbols exist (with positions) for downstream planners
    for rid, pose in robot_states.items():
        if pose is None:
            continue
        symbols_of_interest.append(
            PddlSymbol(rid, "robot", [], position=np.array(pose[:2]))
        )

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

    return problem_str, symbols_of_interest


@dispatch
def ground_problem(
    domain: MultiRobotPddlDomain,
    dsg: spark_dsg.DynamicSceneGraph,
    robot_states: dict,
    goal: PddlGoal,
    feedback: Any = None,
) -> MultiRobotWrapper[GroundedPddlProblem]:
    logger.warning(f"Grounding PDDL Problem {domain.domain_name}")

    pddl_compliant_robot_states = {k.lower(): v for k, v in robot_states.items()}
    match domain.domain_name:
        # case "goto-object-domain-multirobot-fd":
        #     pddl_problem, symbols = generate_multirobot_inspection_pddl(
        #         dsg, goal.pddl_goal, robot_states
        #     )

        case "region-object-rearrangement-domain-multirobot-fd":
            pddl_problem, symbols = generate_multirobot_region_pddl(
                dsg, goal.pddl_goal, pddl_compliant_robot_states
            )
            # logger.warning(f"!!!!!!!!!!!!!!pddl_problem: {pddl_problem}")
        case _:
            raise NotImplementedError(
                f"I don't know how to ground a domain of type {domain.domain_name}!"
            )

    symbol_dict = {s.symbol: s for s in symbols}

    # TODO: We don't actually want to rely on the robot_states, because
    # robot_states may contain information about robots we actually don't care
    # about for planning purposes.  Instead, we probably want this information
    # passed as part of the goal.

    valid_robot_names = [
        name for name, pose in robot_states.items() if pose is not None
    ]
    wrapper = MultiRobotWrapper(
        valid_robot_names, GroundedPddlProblem(domain, pddl_problem, symbol_dict)
    )
    for outer_name in valid_robot_names:
        inner_name = outer_name.lower()
        wrapper.set_name_remap(outer_name, inner_name)
    return wrapper
