"""Time multi-robot PDDL planning on a real map: grounding, solve and expansion.

Runs ground_problem + make_plan (the planner node's path) for a few goals on a
saved scene graph, in the goal-relevant scene scope and optionally the full
one, and prints one line per run: seconds per stage, points of interest, plan
length in actions and metres driven.

    python multirobot_scale_benchmark.py DSG.json [--full] [--targets 5 10 20 30]

Two robots start next to each other at the first agent pose of the map (or
--start X Y). Goals: visit two objects, the same two assigned to the robots
(plan repair v4), inspect + pick + place, and N-target missions (half visits,
half inspections) over reachable objects picked with a fixed seed.
"""

import argparse
import random
import time
from importlib.resources import as_file, files

import dsg_pddl.domains
import dsg_pddl.dsg_pddl_grounding_multirobot  # noqa: F401 (registers ground_problem)
import numpy as np
import spark_dsg
from dsg_pddl.dsg_pddl_grounding_multirobot import ground_problem
from dsg_pddl.dsg_pddl_planning import make_plan
from dsg_pddl.pddl_grounding import MultiRobotPddlDomain, PddlGoal

from omniplanner.tsp import LayerPlanner

DOMAIN = "RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl"


def domain(scope):
    with as_file(files(dsg_pddl.domains).joinpath(DOMAIN)) as p:
        d = MultiRobotPddlDomain(open(p).read())
    d.scene_scope = scope
    return d


def reachable_objects(G, start):
    planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES).restricted_to_component(
        start
    )
    component = set(planner.node_ids)
    out = []
    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        pos = np.asarray(node.attributes.position)[:2]
        nearest = planner.get_closest_node_id(pos)
        if (
            nearest in component
            and np.linalg.norm(planner.node_value_to_position[nearest] - pos) < 3.0
        ):
            out.append(node.id.str(True).lower())
    return sorted(out, key=lambda s: int(s[1:]))


def travel(plan):
    total = 0.0
    for action, path in zip(plan.symbolic_actions, plan.parameterized_actions):
        if action[0] == "goto-poi":
            pts = [np.asarray(p)[:2] for p in path]
            total += sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:]))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dsg")
    ap.add_argument("--full", action="store_true", help="also run the full encoding")
    ap.add_argument("--targets", type=int, nargs="*", default=[5, 10, 20, 30])
    ap.add_argument("--start", type=float, nargs=2)
    a = ap.parse_args()

    t0 = time.perf_counter()
    G = spark_dsg.DynamicSceneGraph.load(a.dsg)
    print(f"loaded {a.dsg}: {G.num_nodes()} nodes in {time.perf_counter() - t0:.1f} s")
    if a.start:
        start = np.array(a.start)
    else:
        agent = next(iter(G.get_layer(spark_dsg.DsgLayers.AGENTS, "a").nodes))
        start = np.asarray(agent.attributes.position)[:2]
    robots = {"hamilton": start, "euclid": start + 0.1}
    objects = reachable_objects(G, start)
    rng = random.Random(0)
    picks = rng.sample(objects, max([2] + a.targets))
    home = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES).get_closest_node_id(start)
    home = spark_dsg.NodeSymbol(home).str(True).lower()
    o1, o2 = picks[:2]
    goals = {
        "visit2": f"(and (visited-object {o1}) (visited-object {o2}))",
        "assigned2": f"(and (visited-object-by euclid {o1}) (visited-object-by hamilton {o2}))",
        "inspect_pick_place": f"(and (inspected-by euclid {o1}) (object-in-place {o2} {home}))",
    }
    for n in a.targets:
        o = picks[:n]
        goals[f"targets{n}"] = (
            "(and "
            + " ".join(f"(visited-object {x})" for x in o[n // 2 :])
            + " "
            + " ".join(f"(safe {x})" for x in o[: n // 2])
            + ")"
        )

    for scope in ["goal_relevant"] + (["full"] if a.full else []):
        d = domain(scope)
        for name, goal in goals.items():
            t0 = time.perf_counter()
            try:
                grounded = ground_problem(
                    d, G, robots, PddlGoal(pddl_goal=goal, robot_id="hamilton")
                ).value
                t1 = time.perf_counter()
                plan = make_plan(grounded, G)
                t2 = time.perf_counter()
            except Exception as e:  # noqa: BLE001
                print(
                    f"{scope:13s} {name:18s} FAILED after {time.perf_counter() - t0:.1f} s: {e}"
                )
                continue
            pois = sum(
                1 for s in grounded.symbols.values() if s.layer in ("place", "object")
            )
            print(
                f"{scope:13s} {name:18s} ground {t1 - t0:6.2f} s  solve+expand {t2 - t1:6.2f} s  "
                f"total {t2 - t0:6.2f} s  POIs {pois:5d}  actions {len(plan.symbolic_actions):3d}  "
                f"driven {travel(plan):7.1f} m"
            )


if __name__ == "__main__":
    main()
