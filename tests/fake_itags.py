"""A stand-in for the itags binary: allocates by traits alone, greedily.

Reads itags_input.json and writes an itags_solution.json in the solver's
format. Each task (in precedence order) goes to the smallest coalition whose
summed species traits cover the task's desired_traits -- one robot if one
suffices, else a pair -- preferring the robot with the fewest tasks so far,
then the nearest. It knows nothing about names: like ITAGS it can only give a
task to robots whose traits meet it, which is what the trait encoding of
use-robots / holding has to get right.

Usage: fake_itags.py <itags_input.json> <itags_solution.json>
"""

import itertools
import json
import math
import sys


def covers(supply, demand):
    return all(s >= d - 1e-9 for s, d in zip(supply, demand))


def order(n, pairs):
    after = {i: set() for i in range(n)}
    for a, b in pairs:
        after[b].add(a)
    done, out = set(), []
    while len(out) < n:
        ready = [i for i in range(n) if i not in done and after[i] <= done]
        if not ready:
            raise SystemExit("cyclic precedence")
        out.append(ready[0])
        done.add(ready[0])
    return out


def main(in_path, out_path):
    with open(in_path) as f:
        data = json.load(f)
    species = {s["name"]: s["traits"] for s in data["species"]}
    robots = data["robots"]
    supply = [species[r["species"]] for r in robots]
    at = [
        (r["initial_configuration"]["x"], r["initial_configuration"]["y"])
        for r in robots
    ]
    plans = [[] for _ in robots]
    legs = [[] for _ in robots]
    free = [0.0] * len(robots)
    finish = {}
    tasks_out = []
    tasks = data["tasks"]
    for i in order(len(tasks), data["precedence_constraints"]):
        t = tasks[i]
        demand = t["desired_traits"]
        here = (t["initial_configuration"]["x"], t["initial_configuration"]["y"])
        coalition = None
        for size in (1, 2):
            options = [
                c
                for c in itertools.combinations(range(len(robots)), size)
                if covers([sum(v) for v in zip(*(supply[k] for k in c))], demand)
            ]
            if options:
                coalition = min(
                    options,
                    key=lambda c: (
                        sum(len(plans[k]) for k in c),
                        sum(math.dist(at[k], here) for k in c),
                    ),
                )
                break
        if coalition is None:
            raise SystemExit(f"no coalition covers task {t['name']}")
        start = max(
            [free[k] + math.dist(at[k], here) for k in coalition]
            + [finish[a] for a, b in data["precedence_constraints"] if b == i]
        )
        end = start + t["duration"]
        finish[i] = end
        for k in coalition:
            plans[k].append(i)
            legs[k].append(
                {
                    "path": [
                        {"x": at[k][0], "y": at[k][1]},
                        {"x": here[0], "y": here[1]},
                    ]
                }
            )
            at[k] = (
                t["terminal_configuration"]["x"],
                t["terminal_configuration"]["y"],
            )
            free[k] = end
        tasks_out.append(
            {
                "id": i,
                "name": t["name"],
                "coalition": list(coalition),
                "start_timepoint": start,
                "finish_timepoint": end,
            }
        )
    solution = {
        "makespan": max(finish.values(), default=0.0),
        "tasks": sorted(tasks_out, key=lambda t: t["id"]),
        "agents": [
            {"name": r["name"], "individual_plan": plans[k], "transitions": legs[k]}
            for k, r in enumerate(robots)
        ],
        "precedence_constraints": data["precedence_constraints"],
    }
    with open(out_path, "w") as f:
        json.dump({"solution": solution}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
