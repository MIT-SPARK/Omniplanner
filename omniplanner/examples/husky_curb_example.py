#!/usr/bin/env python3
"""A Husky that avoids curbs, on a scene graph.

Plans one trip for one robot twice with the multi-robot PDDL planner: as a Spot
(no limit), then as a Husky that avoids places whose metadata marks them as a
curb, {"curb": true}. Prints both paths and PASS if the Spot's goes over a curb
and the Husky's does not.

A map without curb marks is marked here, in memory only: a target about
--trip-m away is picked, and the places within --curb-m of the middle of the
shortest route to it are marked as a curb.

Needs fast-downward on PATH; no ROS. Run from this directory:

    python husky_curb_example.py --scene-graph /path/to/dsg.json
"""

import argparse
import logging

import dsg_pddl.dsg_pddl_grounding_multirobot  # noqa: F401 (registers ground_problem)
import dsg_pddl.dsg_pddl_planning  # noqa: F401 (registers make_plan)
import networkx as nx
import numpy as np
import spark_dsg
from dsg_pddl.dsg_pddl_grounding_multirobot import PLATFORM_AVOIDS, places_labelled
from dsg_pddl.pddl_grounding import MultiRobotPddlDomain, PddlGoal
from utils import load_omniplanner_pddl_domain

from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner.tsp import LayerPlanner
from omniplanner.world_state import WorldState

DOMAIN = "RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl"


def _symbol(node_id):
    return spark_dsg.NodeSymbol(node_id).str(True).lower()


def _mesh_places(G):
    try:
        return G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        return G.get_layer(20)


def husky_curb_demo(G, domain, robot, pose, trip_m=20.0, curb_m=2.0):
    """One trip, planned for the same robot as a Spot, then as a Husky that
    avoids curbs. Returns True if the paths differ as they should (or the Husky
    is refused a trip it has no way to make)."""
    start = np.asarray(pose, dtype=float)[:2]
    planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES).weighted()
    planner = planner.restricted_to_component(start)
    graph, xy = planner.nx_layer, planner.node_value_to_position
    here = planner.get_closest_node_id(start)

    marked = places_labelled(G, PLATFORM_AVOIDS["husky"])
    lengths, routes = nx.single_source_dijkstra(graph, here, weight=planner.weight)
    if marked:
        print(f"The map marks {len(marked)} places as a curb: using them")
        # the target nearest trip_m away whose shortest route crosses a curb
        targets = [
            n
            for n in sorted(lengths, key=lambda n: (abs(lengths[n] - trip_m), n))
            if n not in marked and set(routes[n]) & marked
        ]
        if not targets:
            print("No shortest route from here crosses a curb: nothing to compare.")
            print("Start the Husky nearer a curb.")
            return False
        target, curb = targets[0], set(routes[targets[0]]) & marked
    else:
        far = [n for n in lengths if len(routes[n]) >= 5]
        if not far:
            print(f"No place far enough from {_symbol(here)} for a demo trip")
            return False
        target = min(far, key=lambda n: (abs(lengths[n] - trip_m), n))
        route = routes[target]
        middle = xy[route[len(route) // 2]]
        # the widest curb that still leaves the Husky a way around
        curb = set()
        for radius in (curb_m, curb_m / 2, 0.0):
            curb = {
                n
                for n in graph
                if np.linalg.norm(xy[n] - middle) <= radius and n not in (here, target)
            } or {route[len(route) // 2]}
            around = graph.copy()
            around.remove_nodes_from(curb)
            if nx.has_path(around, here, target):
                break
        else:
            print("Every way to the target crosses the curb: expect a refusal")
        for node in _mesh_places(G).nodes:
            if node.id.value in curb:
                node.attributes.metadata.add({"curb": True})
        print(
            f"The map has no curb marks: marked {len(curb)} places around "
            f"{_symbol(route[len(route) // 2])} as a curb (in memory only)"
        )

    name = robot.lower()
    goal_string = f"(visited-place-by {name} {_symbol(target)})"
    print(f"Robot {robot} at {start}, goal {goal_string}")
    print(f"Curb places on the way: {sorted(_symbol(n) for n in curb)}")
    curb_xy = np.array([xy[n] for n in curb])

    def plan_as(label, world_state):
        goal = PddlGoal(robot_id=name, pddl_goal=goal_string, world_state=world_state)
        req = PlanRequest(domain=domain, goal=goal, robot_states={robot: pose})
        try:
            plan = full_planning_pipeline(req, G).value.value
        except Exception as exc:
            print(f"  as a {label}: no plan ({exc})")
            return None
        path = [
            np.asarray(p, dtype=float)[:2]
            for action, hop in zip(plan.symbolic_actions, plan.parameterized_actions)
            if action[0] == "goto-poi"
            for p in hop
        ]
        metres = sum(float(np.linalg.norm(b - a)) for a, b in zip(path, path[1:]))
        over = sorted(
            _symbol(n)
            for n, p in zip(curb, curb_xy)
            if min(np.linalg.norm(q - p) for q in path) < 0.05
        )
        print(
            f"  as a {label}: {len(path)} waypoints, {metres:.1f} m, over "
            f"{over or 'none'} of the curb places"
        )
        print(f"    plan: {plan.symbolic_actions}")
        return metres, over

    spot = plan_as("Spot", None)
    husky = plan_as(
        "Husky", WorldState(cannot_manipulate={name}, platform={name: "husky"})
    )
    if spot is None:
        print("RESULT: FAIL -- no plan even without avoiding curbs")
        return False
    if husky is None:
        print("RESULT: the Husky is refused the trip the Spot can make (no way around)")
        return True
    # The Husky's way around can be as short as the Spot's (an open area), so
    # what counts is which places each path passes, not the length.
    ok = bool(spot[1]) and not husky[1]
    print(
        f"RESULT: {'PASS' if ok else 'FAIL'} -- the Spot's path "
        f"{'crosses' if spot[1] else 'does not cross'} the curb, the Husky's "
        f"{'crosses it too' if husky[1] else 'goes around'} "
        f"({husky[0] - spot[0]:+.1f} m)"
    )
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--scene-graph", required=True, help="scene graph to plan on")
    parser.add_argument(
        "--start",
        type=float,
        nargs=2,
        default=(0.0, 6.0),
        metavar=("X", "Y"),
        help="the robot's position in the map frame",
    )
    parser.add_argument(
        "--trip-m",
        type=float,
        default=20.0,
        help="about how far the target is from the start",
    )
    parser.add_argument(
        "--curb-m",
        type=float,
        default=2.0,
        help="radius of the curb marked on the Spot's route (maps with no marks)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    G = spark_dsg.DynamicSceneGraph.load(args.scene_graph)
    print(f"Loaded {args.scene_graph}: {G.num_nodes()} nodes")
    domain = MultiRobotPddlDomain(load_omniplanner_pddl_domain(DOMAIN))
    ok = husky_curb_demo(
        G,
        domain,
        "gauss",
        np.array(args.start),
        trip_m=args.trip_m,
        curb_m=args.curb_m,
    )
    raise SystemExit(0 if ok else 1)
