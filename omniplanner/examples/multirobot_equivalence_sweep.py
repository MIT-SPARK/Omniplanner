"""Goal-relevant vs full multi-robot grounding on random goals over a real map.

For each trial two robots start at random places of the largest component and
get a random goal (visits, robot assignment, inspect, inspect + pick + place,
``or``, end-at-place); both scene scopes plan it. Reports whether each solved,
metres driven by the expanded plan, and seconds, then a summary.

    python multirobot_equivalence_sweep.py DSG.json [--trials 30] [--seed 0]
"""

import argparse
import random
import statistics
import time
from importlib.resources import as_file, files

import dsg_pddl.domains
import networkx as nx
import numpy as np
import spark_dsg
from dsg_pddl.dsg_pddl_grounding_multirobot import ground_problem
from dsg_pddl.dsg_pddl_planning import make_plan
from dsg_pddl.pddl_grounding import MultiRobotPddlDomain, PddlGoal

from omniplanner.tsp import LayerPlanner

DOMAIN = "RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl"
ROBOTS = ("hamilton", "euclid")


def domain(scope):
    with as_file(files(dsg_pddl.domains).joinpath(DOMAIN)) as p:
        d = MultiRobotPddlDomain(open(p).read())
    d.scene_scope = scope
    return d


def travel(plan):
    total = 0.0
    for action, path in zip(plan.symbolic_actions, plan.parameterized_actions):
        if action[0] == "goto-poi":
            pts = [np.asarray(p)[:2] for p in path]
            total += sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:]))
    return total


def name(node_id):
    return spark_dsg.NodeSymbol(node_id).str(True).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dsg")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    G = spark_dsg.DynamicSceneGraph.load(a.dsg)
    planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)
    component = max(nx.connected_components(planner.nx_layer), key=len)
    places = sorted(component)
    objects = []
    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        pos = np.asarray(node.attributes.position)[:2]
        nearest = planner.get_closest_node_id(pos)
        if (
            nearest in component
            and np.linalg.norm(planner.node_value_to_position[nearest] - pos) < 3.0
        ):
            objects.append(node.id.str(True).lower())
    print(
        f"{a.dsg}: {len(places)} places in the main component, {len(objects)} reachable objects"
    )
    rng = random.Random(a.seed)
    domains = {s: domain(s) for s in ("goal_relevant", "full")}
    rows = []
    for trial in range(a.trials):
        robots = {r: planner.node_value_to_position[rng.choice(places)] for r in ROBOTS}
        o = rng.sample(objects, 3)
        p = name(rng.choice(places))
        r1, r2 = rng.sample(ROBOTS, 2)
        kind, goal = rng.choice(
            [
                (
                    "visit",
                    "(and "
                    + " ".join(f"(visited-object {x})" for x in o[: rng.randint(1, 3)])
                    + ")",
                ),
                (
                    "assigned",
                    f"(and (visited-object-by {r1} {o[0]}) (visited-object-by {r2} {o[1]}))",
                ),
                ("inspect", f"(and (inspected-by {r1} {o[0]}) (safe {o[1]}))"),
                (
                    "pick_place",
                    f"(and (inspected-by {r1} {o[0]}) (object-in-place {o[0]} {p}))",
                ),
                ("or", f"(or (visited-object {o[0]}) (visited-object {o[1]}))"),
                ("end_at", f"(and (at-place {r1} {p}) (visited-object {o[0]}))"),
            ]
        )
        row = {"trial": trial, "kind": kind}
        for scope, d in domains.items():
            t0 = time.perf_counter()
            try:
                plan = make_plan(
                    ground_problem(
                        d, G, robots, PddlGoal(pddl_goal=goal, robot_id=r1)
                    ).value,
                    G,
                )
                row[scope] = (True, travel(plan), time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001
                row[scope] = (False, None, time.perf_counter() - t0, type(e).__name__)
        rows.append(row)
        rel, full = row["goal_relevant"], row["full"]
        print(
            f"{trial:3d} {kind:10s} full: {'ok' if full[0] else full[3]:>18s} "
            f"{full[1] or 0:7.1f} m {full[2]:6.2f} s | goal_relevant: {'ok' if rel[0] else rel[3]:>18s} "
            f"{rel[1] or 0:7.1f} m {rel[2]:6.2f} s",
            flush=True,
        )
    both = [r for r in rows if r["full"][0] and r["goal_relevant"][0]]
    print("\nsummary")
    print(
        f"  solved: full {sum(r['full'][0] for r in rows)}/{len(rows)}, goal_relevant {sum(r['goal_relevant'][0] for r in rows)}/{len(rows)}"
    )
    print(
        f"  only full solved: {[r['trial'] for r in rows if r['full'][0] and not r['goal_relevant'][0]]}"
    )
    print(
        f"  only goal_relevant solved: {[r['trial'] for r in rows if r['goal_relevant'][0] and not r['full'][0]]}"
    )
    if both:
        ratio = [r["goal_relevant"][1] / r["full"][1] for r in both if r["full"][1] > 0]
        print(
            f"  driven, goal_relevant / full: median {statistics.median(ratio):.2f}, max {max(ratio):.2f}, min {min(ratio):.2f}"
        )
        print(
            f"  longer by >5%: {[r['trial'] for r in both if r['full'][1] > 0 and r['goal_relevant'][1] > 1.05 * r['full'][1]]}"
        )
        tf = [r["full"][2] for r in both]
        tr = [r["goal_relevant"][2] for r in both]
        print(
            f"  seconds, median: full {statistics.median(tf):.2f}, goal_relevant {statistics.median(tr):.2f}"
        )
        print(f"  seconds, max:    full {max(tf):.2f}, goal_relevant {max(tr):.2f}")


if __name__ == "__main__":
    main()
