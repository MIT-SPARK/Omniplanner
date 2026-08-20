#!/usr/bin/env python3
"""GRSTAPS-X end-to-end test with a real scene graph.

The counterpart to pddl_example_multirobot_FD_ominiplanner.py: the LLM is
skipped and the goal is handed in directly, so what is under test is the
planner rather than the prompt.

What this exercises that the fast-downward example cannot:

  traits       a heterogeneous fleet -- plain spot, spot with a manipulator,
               and a faster UAV -- where capability decides who CAN do a task
               and speed decides who SHOULD
  coalitions   an inspection whose trait requirement no single robot meets, so
               ITAGS must field a ground+air team on one task at one time
  precedence   inspect -> pick -> place, derived rather than declared
  ordering     "visit o2 before picking o42" -- an instruction-level ordering
               that PDDL goals cannot express at all
  scheduling   real start/finish timepoints from the MILP scheduler

Two entry points, selected by GrstapsDomain.entry:

  "pddl"   domain.pddl + problem.pddl -> grstapsx_example. The task planner
           decides which tasks exist and derives precedence causally.
  "itags"  itags_input.json -> itags. Tasks and precedence are enumerated by
           omniplanner, so an arbitrary ordering is just another index pair.
           Requires the `itags` cmake target to be built.

Run it with the workspace environment, not the system python -- the system
spark_dsg cannot read this graph format:

    ~/environments/dcist/spark_env/bin/python grstaps_example_omniplanner.py itags
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import spark_dsg

from omniplanner.grstaps_planner import (
    INSPECT_DURATION,
    PICK_DURATION,
    PLACE_DURATION,
    VISIT_DURATION,
    GrstapsDomain,
    GrstapsGoal,
    _symbol_position,
    enumerate_tasks,
    ordering_precedence,
    parse_goal_targets,
)
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline

logging.basicConfig(level=logging.INFO, force=True)

SCENE_GRAPH_PATH = (
    "/home/jaeyoun-choi/colcon_ws/assets/adt4_output/plan_repair_graph"
    "/hydra/backend/dsg_with_mesh.json"
)

# robot_name -> robot_type, exactly as omniplanner_plugins.yaml declares it.
# The plugin reads these off the node's adaptors at runtime; here we supply
# them directly, which is the only difference from the live path.
ROBOT_TYPES = {
    "scout": "spot",
    "worker": "spot-with-manipulator",
    "eye": "uav",
}


class ConstraintFact:
    """Stands in for omniplanner_msgs/ConstraintFact, which needs ROS.

    The planner only ever reads .predicate and .symbols.
    """

    def __init__(self, predicate, symbols):
        self.predicate = predicate
        self.symbols = list(symbols)

    def __repr__(self):
        return f"({self.predicate} {' '.join(self.symbols)})"


def unwrap(plan):
    """Strip the RobotWrapper / MultiRobotWrapper / SymbolicContext layers."""
    while hasattr(plan, "value"):
        plan = plan.value
    return plan


def robot_of_task(plan):
    """{task id -> [robot names]}, from the solver's own allocation."""
    out = {}
    for agent in plan.agents or []:
        for task_id in agent.get("individual_plan") or []:
            out.setdefault(task_id, []).append(agent.get("name"))
    return out


def print_schedule(plan):
    """The schedule, which is the whole point of using this planner."""
    owners = robot_of_task(plan)
    print(f"\n  makespan: {plan.makespan:.2f} s over {len(plan.tasks)} task(s)")
    print(f"  {'task':44} {'start':>8} {'finish':>8}  robots")
    print("  " + "-" * 78)
    for t in sorted(plan.tasks or [], key=lambda t: t.get("start_timepoint", 0.0)):
        print(
            f"  {t.get('name', '?'):44} "
            f"{t.get('start_timepoint', 0.0):8.2f} {t.get('finish_timepoint', 0.0):8.2f}"
            f"  {owners.get(t.get('id'), [])}"
        )


def check_chain(plan, links):
    """Verify each declared ordering, symbol by symbol.

    Comparing every task of one ACTION against every task of another is
    meaningless once a chain interleaves them -- the links are between
    symbols, so that is what has to be checked.
    """
    span = {}
    for t in plan.tasks or []:
        sym = (t.get("name") or "").split("::")[0].strip().split()[-1]
        a, b = span.get(sym, (1e9, -1e9))
        span[sym] = (
            min(a, t.get("start_timepoint", 0.0)),
            max(b, t.get("finish_timepoint", 0.0)),
        )
    # The solver stores timepoints as 32-bit floats, so at a makespan of a few
    # hundred seconds one ULP is ~3e-5. A back-to-back pair therefore differs by
    # representation noise, not by a real gap; a millisecond of slack is well
    # below anything physical and well above that noise.
    EPS = 1e-3
    for pre, post in links:
        if any(x not in span for x in list(pre) + [post]):
            continue
        last = max(span[x][1] for x in pre)
        first = span[post][0]
        ok = first >= last - EPS
        print(
            f"  {'OK  ' if ok else 'FAIL'} {'+'.join(pre):>10} ends {last:7.2f} "
            f"{'<=' if ok else '>'} {post} starts {first:7.2f}"
        )


def check_precedence(plan, earlier_action, later_action):
    """Assert one action finishes before another starts, and say so."""
    times = {}
    for t in plan.tasks or []:
        act = (t.get("name") or "").split()[0] if t.get("name") else ""
        times.setdefault(act, []).append(
            (t.get("start_timepoint", 0.0), t.get("finish_timepoint", 0.0))
        )
    if earlier_action not in times or later_action not in times:
        return
    last_end = max(f for _, f in times[earlier_action])
    first_start = min(s for s, _ in times[later_action])
    ok = last_end <= first_start + 1e-6
    print(
        f"  {'OK  ' if ok else 'FAIL'} {earlier_action} ends {last_end:.2f} "
        f"{'<=' if ok else '>'} {later_action} starts {first_start:.2f}"
    )


def fleet_start(G, names, spec):
    """Starting pose per robot.

    Defaults to one central place for everyone. `spread` scatters them to the
    map's extremes, which is useful for exercising allocation but puts robots
    in odd corners; explicit places are there for when the start matters.
    """
    places = list(G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes)
    by_sym = {n.id.str(True).lower(): n for n in places}
    if "=" in spec:
        out = {}
        for part in spec.split(","):
            name, _, sym = part.partition("=")
            node = by_sym.get(sym.strip().lower())
            if node is None:
                sys.exit(f"--start: no place {sym!r} in this graph")
            p = node.attributes.position
            out[name.strip()] = np.array([float(p[0]), float(p[1])])
        missing = [n for n in names if n not in out]
        if missing:
            sys.exit(f"--start: no position given for {', '.join(missing)}")
        return [out[n] for n in names]

    pts = [
        np.array([float(n.attributes.position[0]), float(n.attributes.position[1])])
        for n in places
    ]
    centre = min(pts, key=lambda q: np.linalg.norm(q - np.mean(pts, axis=0)))
    if spec == "centre":
        return [centre.copy() for _ in names]
    if spec == "spread":
        chosen = [centre]
        while len(chosen) < len(names):
            chosen.append(
                max(pts, key=lambda q: min(np.linalg.norm(q - c) for c in chosen))
            )
        return chosen
    sys.exit(f"--start: expected 'centre', 'spread', or name=place pairs, got {spec!r}")


def place_positions(G, n_robots):
    """Spread the fleet over the map so allocation has something to decide.

    Farthest-point sampling, not evenly spaced indices: the places layer is in
    no particular spatial order, so picking places[0], places[n/2], places[n]
    lands wherever those happen to be. On this map that put two robots 3.5 m
    apart and the third 17 m away, which reads as a bug in the plan when it is
    really just an arbitrary start.
    """
    places = list(G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes)
    pts = [
        np.array([float(n.attributes.position[0]), float(n.attributes.position[1])])
        for n in places
    ]
    centre = np.mean(pts, axis=0)
    chosen = [min(pts, key=lambda q: np.linalg.norm(q - centre))]  # start central
    while len(chosen) < n_robots:
        chosen.append(
            max(pts, key=lambda q: min(np.linalg.norm(q - c) for c in chosen))
        )
    return chosen


def goal_from_pddl(pddl_goal, robot_id, constraints=None):
    """Build a GrstapsGoal from a PDDL goal string, as the live plugin does.

    This is the boundary the LLM sits above: everything from the goal STRING
    downwards is exercised here. parse_goal_targets sorts the goal's predicates
    into task kinds -- visited-*/at-* are visits, safe/(not suspicious) is an
    inspection, holding/object-in-place is a manipulation -- using the same
    vocabulary the multirobot fast-downward domain already defines.
    """
    found = parse_goal_targets(pddl_goal)
    print(f"  goal: {pddl_goal}")
    print(
        f"  parsed -> visit={found['visit']} inspect={found['inspect']} "
        f"manipulate={found['manipulate']}"
    )
    if found["destinations"]:
        print(f"  destinations-> {found['destinations']}")
    return GrstapsGoal(
        goal_points=found["visit"],
        inspect_points=found["inspect"],
        manipulate_points=found["manipulate"],
        # Without these a relocation collapses to "put it back where you found
        # it": the place task keeps the pick location as its terminal
        # configuration, so the carry is never planned or charged.
        manipulate_destinations=found["destinations"],
        robot_id=robot_id,
        robot_types=ROBOT_TYPES,
        constraints=constraints or [],
    )


def print_symbolic_plan(scenario):
    """Show the ITAGS problem omniplanner generated, and where it lives.

    This is the symbolic plan: the task list with its trait requirements and
    durations, plus the precedence pairs. Reading it is how you check that a
    goal and its ordering became what you meant, without inferring it from a
    schedule.
    """
    root = os.environ.get("ADT4_GRSTAPS_ROOT", os.path.expanduser("~/grstapsx"))
    d = os.path.join(root, "data", "grstapsx", scenario)
    inp = os.path.join(d, "itags_input.json")
    if not os.path.exists(inp):
        print(f"  (no itags_input.json at {inp}; entry='itags' writes it)")
        return
    j = json.load(open(inp))
    print(
        f"\n  symbolic plan  ({len(j['tasks'])} tasks, "
        f"{len(j['precedence_constraints'])} precedence pairs)"
    )
    print(f"  {'#':>2}  {'task':44} {'dur':>5}  requires")
    for i, t in enumerate(j["tasks"]):
        req = [f"{n}x d{k}" for k, n in enumerate(t["desired_traits"]) if n]
        print(f"  {i:>2}  {t['name']:44} {t['duration']:5.1f}  {', '.join(req) or '-'}")
    if j["precedence_constraints"]:
        print("  precedence (a must finish before b starts):")
        for a, b in j["precedence_constraints"]:
            print(
                f"      {a} -> {b}    {j['tasks'][a]['name'].split('::')[0].strip()}"
                f"  ->  {j['tasks'][b]['name'].split('::')[0].strip()}"
            )
    print(f"  files: {inp}")
    print(f"         {os.path.join(d, 'itags_solution.json')}")


def solve(G, goal, entry, scenario):
    request = PlanRequest(
        domain=GrstapsDomain(
            scenario_name=scenario,
            run_mode="native",  # native binary rather than the docker image
            entry=entry,
        ),
        goal=goal,
        robot_states=ROBOT_STATES,
    )
    return unwrap(full_planning_pipeline(request, G))


parser = argparse.ArgumentParser(
    description="GRSTAPS-X end-to-end test. With no --goal it runs the built-in cases.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""examples:
  # the built-in cases
  %(prog)s itags

  # your own goal, ordered by symbol
  %(prog)s itags --goal "(and (object-in-place o4 t598)(visited-object o8))" \
                 --before o4 o8

  # enumerate the tasks WITHOUT solving, to find the indices
  %(prog)s itags --goal "..." --list-tasks

  # then order them by index, the way GRSTAPS-X itself does
  %(prog)s itags --goal "..." --precedence 3 0 --precedence 5 0
""",
)
parser.add_argument("entry", nargs="?", default="pddl", choices=["pddl", "itags"])
parser.add_argument("--goal", help="PDDL goal string; omit to run the built-in cases")
parser.add_argument(
    "--before",
    nargs=2,
    metavar=("A", "B"),
    action="append",
    default=[],
    help="every task of A finishes before any task of B (repeatable)",
)
parser.add_argument(
    "--precedence",
    nargs=2,
    type=int,
    metavar=("I", "J"),
    action="append",
    default=[],
    help="raw task-index pair, i before j (repeatable)",
)
parser.add_argument("--robot", default=None, help="robot the goal is addressed to")
parser.add_argument(
    "--start",
    default="centre",
    help="where the fleet begins: 'centre' (all at the most central place), "
    "'spread' (farthest-point over the map), or explicit places such as "
    "'scout=t3,worker=t4316,eye=t598'",
)
parser.add_argument(
    "--scenario", default="custom", help="scenario name under data/grstapsx"
)
parser.add_argument(
    "--list-tasks",
    action="store_true",
    help="enumerate the tasks and exit, so index-based precedence can be written",
)
args = parser.parse_args()
entry_mode = args.entry

print("GRSTAPS-X End-to-End Test with Real Scene Graph")
print("=" * 80)
print(f"Scene graph: {SCENE_GRAPH_PATH}")
print(f"Entry point: {entry_mode}")

try:
    G = spark_dsg.DynamicSceneGraph.load(SCENE_GRAPH_PATH)
except RuntimeError as exc:
    # The system spark_dsg predates this graph format and fails on 'layer_ids'.
    # The workspace env has the build that reads it.
    sys.exit(
        f"\nCould not load the scene graph: {exc}\n\n"
        f"This usually means the wrong interpreter. Run it with the workspace\n"
        f"environment, which has the spark_dsg build that reads this format:\n\n"
        f"  $ADT4_ENV/spark_env/bin/python {os.path.basename(__file__)} ...\n"
        f"  (typically ~/environments/dcist/spark_env/bin/python)\n"
    )
objects = [n.id.str(True) for n in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes]
n_places = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).num_nodes()
print(f"DSG: {n_places} mesh places, {len(objects)} objects")

ROBOT_STATES = dict(zip(ROBOT_TYPES, fleet_start(G, list(ROBOT_TYPES), args.start)))
for name, kind in ROBOT_TYPES.items():
    print(f"  {name:8} {kind:24} at {np.round(ROBOT_STATES[name], 1)}")

# Pick targets that exist in this graph rather than hard-coding ids.
visit_target, inspect_target, manipulate_target = objects[1], objects[4], objects[7]
print(
    f"\nTargets: visit {visit_target}, inspect {inspect_target}, "
    f"manipulate {manipulate_target}"
)


if args.goal:
    # ---------------------------------------------------------------------
    # A mission given on the command line. Nothing about it is baked in here:
    # the goal, the ordering and the target robot all come from arguments.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Custom mission")
    print("=" * 80)
    cons = [ConstraintFact("before", [a, b]) for a, b in args.before]
    robot = args.robot or next(iter(ROBOT_TYPES))
    goal = goal_from_pddl(args.goal, robot, cons)
    goal.extra_precedence = [list(p) for p in args.precedence]
    if cons:
        print(f"  before      : {cons}")
    if goal.extra_precedence:
        print(f"  precedence  : {goal.extra_precedence}  (raw task indices)")

    if args.list_tasks:
        # Enumerate exactly as grounding would, without running the solver, so
        # the indices printed here are the ones --precedence expects.
        durations = {
            "visit-location": VISIT_DURATION,
            "inspect-object": INSPECT_DURATION,
            "pick-object": PICK_DURATION,
            "place-object": PLACE_DURATION,
        }
        syms = (
            list(goal.goal_points)
            + list(goal.manipulate_points)
            + list(goal.inspect_points)
            + list(goal.manipulate_destinations.values())
        )
        vertex_of = {}
        for sym in syms:
            xy = _symbol_position(G, sym)
            if xy is None:
                print(f"  {sym}: NOT IN THE GRAPH")
                continue
            vertex_of[sym.lower()] = {"id": -1, "x": xy[0], "y": xy[1]}
        tasks, of_symbol, intrinsic = enumerate_tasks(goal, durations, vertex_of)
        print(f"\n  {len(tasks)} task(s):")
        for i, t in enumerate(tasks):
            print(f"   {i:>2}  {t['name']}")
        print(f"\n  intrinsic precedence (always applied): {intrinsic}")
        print(
            f"  from --before                        : "
            f"{ordering_precedence(cons, of_symbol)}"
        )
        print("\n  Pass --precedence I J to add your own pairs on top.")
        sys.exit(0)

    plan = solve(G, goal, entry_mode, args.scenario)
    print_symbolic_plan(args.scenario)
    print_schedule(plan)
    print("\n" + "=" * 80)
    sys.exit(0)


# --------------------------------------------------------------------------
# 1. Plain visits -- the baseline every planner can do
# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print("1. Plain visits (speed decides who goes)")
print("=" * 80)
plan = solve(
    G,
    goal_from_pddl(
        "(and "
        f"(visited-object {visit_target.lower()})"
        f"(visited-object {objects[2].lower()})"
        f"(visited-object {objects[5].lower()}))",
        "scout",
    ),
    entry_mode,
    "ex_visits",
)
print_schedule(plan)


# --------------------------------------------------------------------------
# 2. Capability -- only the manipulator robot can pick and place
# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. Manipulation (capability decides who CAN)")
print("=" * 80)
plan = solve(
    G,
    goal_from_pddl(f"(holding worker {manipulate_target.lower()})", "worker"),
    entry_mode,
    "ex_manip",
)
print_schedule(plan)
check_precedence(plan, "pick-object", "place-object")


# --------------------------------------------------------------------------
# 3. Coalition + the full chain: inspect -> pick -> place
# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. Coalition inspection, then the full inspect -> pick -> place chain")
print("=" * 80)
plan = solve(
    G,
    goal_from_pddl(
        f"(and (not (suspicious {manipulate_target.lower()}))"
        f"(holding worker {manipulate_target.lower()}))",
        "worker",
    ),
    entry_mode,
    "ex_chain",
)
print_schedule(plan)
check_precedence(plan, "inspect-object", "pick-object")
check_precedence(plan, "pick-object", "place-object")
print(
    "  (the inspection is ONE task with two robots: its trait requirement\n"
    "   is met only by a ground+air pair, and traits add up across a coalition)"
)


# --------------------------------------------------------------------------
# 4. Runtime constraints -- keep the fleet away from a POI
# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. Forbidden POI (constraints prune the motion graph)")
print("=" * 80)
plan = solve(
    G,
    goal_from_pddl(
        f"(visited-object {visit_target.lower()})",
        "scout",
        constraints=[ConstraintFact("forbidden-poi", [objects[3].lower()])],
    ),
    entry_mode,
    "ex_forbid",
)
print_schedule(plan)


# --------------------------------------------------------------------------
# 5. Instruction-level ordering -- "visit X first, then pick Y"
#    This is the one thing a PDDL goal cannot express: a goal is a conjunction
#    of facts that must hold at the END, with no notion of order. The ordering
#    rides alongside as a constraint and becomes a precedence pair.
# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"5. Ordering: visit {visit_target} BEFORE manipulating {manipulate_target}")
print("=" * 80)
if entry_mode != "itags":
    print(
        "  SKIPPED on the pddl entry point.\n"
        "  Ordering between two unrelated targets has no causal link in a\n"
        "  domain quantified over ?l, so it cannot be encoded there. Re-run\n"
        "  with `itags` to see it honoured."
    )
else:
    plan = solve(
        G,
        goal_from_pddl(
            f"(and (visited-object {visit_target.lower()})"
            f"(holding worker {manipulate_target.lower()}))",
            "worker",
            constraints=[
                ConstraintFact(
                    "before", [visit_target.lower(), manipulate_target.lower()]
                )
            ],
        ),
        entry_mode,
        "ex_order",
    )
    print_schedule(plan)
    check_precedence(plan, "visit-location", "pick-object")

# --------------------------------------------------------------------------
# 6. A chained mission: clear objects out of the way, then go in.
#    Ordering rides alongside the goal as `before` constraints, because a PDDL
#    goal is a conjunction of end-state facts and cannot express sequence.
# --------------------------------------------------------------------------
print("\n" + "=" * 80)
print("6. Chained mission: clear, visit, clear, visit")
print("=" * 80)
CHAIN_GOAL = (
    "(and (object-in-place o4 t598)(object-in-place o5 t598)"
    "(visited-place t299)(visited-object o8)"
    "(object-in-place o15 t4316)(object-in-place o16 t4316)"
    "(visited-place t4812)(visited-place t4214))"
)
# {1,2} -> 3 -> {4,5} -> 6.  Symbol-level: every task of a precedes every task of b.
CHAIN_CONS = [
    ConstraintFact("before", ["o4", "o8"]),
    ConstraintFact("before", ["o5", "o8"]),
    ConstraintFact("before", ["o8", "o15"]),
    ConstraintFact("before", ["o8", "o16"]),
    ConstraintFact("before", ["o15", "t4214"]),
    ConstraintFact("before", ["o16", "t4214"]),
]
if entry_mode != "itags":
    print(
        "  SKIPPED on the pddl entry point: `before` needs the itags entry,\n"
        "  where precedence is an input rather than derived from the domain."
    )
else:
    plan = solve(
        G, goal_from_pddl(CHAIN_GOAL, "worker", CHAIN_CONS), entry_mode, "chain"
    )
    print_symbolic_plan("chain")
    print_schedule(plan)
    check_chain(
        plan,
        [
            (["o4", "o5"], "o8"),
            (["o8"], "o15"),
            (["o8"], "o16"),
            (["o15", "o16"], "t4214"),
        ],
    )

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
