# GRSTAPS-X in omniplanner

## In one paragraph

GRSTAPS-X is an external C++ solver that does task planning, allocation, MILP
scheduling and motion planning in one pass. It is wired in as **another
planning domain**, dispatched exactly like the PDDL and TSP planners, so no
existing planner was modified. Give it a `GrstapsDomain` and it grounds the DSG
into a small scenario on disk, shells out to the solver, and parses the
schedule back. It can be driven either by a plain list of points or by the
plan-repair flow's `ConstrainedPddlGoalMsg`, which is what lets the heracles
agent talk to it.

**Constraints, short answer:** GRSTAPS-X itself has no notion of constraints.
Omniplanner enforces them *before* the solver runs, by cutting the forbidden
region out of the motion graph. See [Constraints](#constraints) — including
what is **not** honoured.

---

## The 30-second version

```
heracles agent ──ConstrainedPddlGoalMsg──▶ goal_manager ──▶ omniplanner_node
                                               │                    │
                                    "do I even need to replan?"      ▼
                                               ▲            grstaps_planner
                                               │                    │
                                     plan_visited_pois ◀────────────┘
                                                            ground → solve → compile
```

The goal manager decides *whether* to replan; it does not care *who* plans.
Pointing it at GRSTAPS-X instead of fast-downward is one launch argument
(`planner_goal_topic`). The agent is unchanged and never names a planner.

---

## How it plugs in

Omniplanner dispatches on the domain type via `plum`. GRSTAPS-X adds two
methods and touches nothing else:

```python
ground_problem(GrstapsDomain, dsg, robot_states, goal) -> RobotWrapper[GroundedGrstapsProblem]
make_plan(GroundedGrstapsProblem, map_context)         -> GrstapsPlan
compile_plan(adaptor, frame, GrstapsPlan)              -> ActionSequence
```

| file | role |
|---|---|
| `omniplanner/src/omniplanner/grstaps_planner.py` | domain, grounding, solver invocation, plan parsing |
| `omniplanner_ros/src/omniplanner_ros/grstaps_planner_ros.py` | ROS plugin, goal callbacks, `compile_plan`, repair hook |

### Where it differs from the PDDL planners

This is the substantive design difference, not an implementation detail.

The PDDL planners compile the DSG *into* the problem file — every place becomes
a symbol, every adjacency a `connected` fact, every pair a `distance`. The
state space therefore grows with the map, which is why fast-downward peaked at
5.1 GB on a 1279-place graph.

GRSTAPS-X keeps geometry **out** of the PDDL. The mission PDDL knows only
"there are locations, visit them"; the metric world is handed over separately
as a euclidean motion graph. Grounding is consequently map-independent — the
same code planned unmodified against 96-, 202- and 1279-place graphs.

---

## What grounding writes

`ground_problem` writes a scenario under
`$ADT4_GRSTAPS_ROOT/data/grstapsx/<scenario_name>/`:

| file | contents |
|---|---|
| `domain.pddl` | mission logic only — a single `visit-location` durative action |
| `problem.pddl` | the locations to visit, one per goal symbol |
| `action_trait_config.json` | robots, species traits, action→subtask traits |
| `maps/ground_graph.json` | the DSG mesh-places layer as a euclidean motion graph |

Two non-obvious invariants, both learned the hard way:

- A location's `x`/`y` in `nodes` must be its **snapped motion-graph vertex's**
  coordinates, not the object's own. The solver treats `x/y` and `id` as the
  same point; a mismatch fails allocation with *"Searched the entire space, but
  couldn't find a solution."*
- `nodes` must contain a `_start_<id>` entry per robot, or config parsing fails
  with *"json is missing field 'x'."*

Goal symbols become PDDL location names by lowercasing (`O8` → `o8`), matching
what the PDDL planners do, so plans from either planner name the same symbols
and stay comparable. Symbol lookup into the DSG is case-insensitive, since the
DSG stores `O8` but a PDDL goal always arrives lowercase.

---

## Bridging the domain gap

The fast-downward domain and GRSTAPS-X's own domains are not two dialects of
the same thing — they are structurally incompatible. Nothing translates one
into the other. Instead we **write a third, minimal domain** in the GRSTAPS-X
idiom and carry only the goal targets across.

### Three domains, side by side

| | FD multirobot (`...MultiRobot_FD_Explore.pddl`) | GRSTAPS-X native (`wpc_spread`) | ours (`dsg-visit`) |
|---|---|---|---|
| actions | 4 — `goto-poi`, `inspect`, `pick-object`, `place-object` | 12 durative — recon, scan, strike, capture, rescue, supply | **1** — `visit-location` |
| robots in PDDL | yes: every predicate takes `?r - robot` | **none at all** | none |
| movement | `goto-poi` + `connected` / `distance` functions | not in PDDL | not in PDDL |
| who allocates | the planner, as part of search | ITAGS downstream, from trait vectors | ITAGS (but traits are trivial) |
| coalitions | not expressible | one action, split by config `action_templates` | none |
| objective | `minimize (total-cost)` | durative, constant durations | `minimize (total-time)` |

### Why the FD domain cannot simply be handed over

- **Robot parameters defeat the point.** GRSTAPS-X domains are robot-agnostic
  *by design* — robots enter through the config's trait vectors, and deciding
  who does what is ITAGS' job. A domain that binds `?r - robot` into every
  action has already made the allocation decision the solver exists to make.
- **The motion graph is in the wrong place.** `connected` and `distance` put
  geometry inside the PDDL. GRSTAPS-X takes the motion graph separately and
  would be planning over a duplicated, and much larger, world.
- **Ordering is expressed differently.** FD chains actions through robot state
  (`at-poi ?r ?p`); GRSTAPS-X chains them through *location* state
  (`reconed` → `ground-scanned` → `ground-threat-neutralized` → `secured`),
  precisely so that no robot needs naming.

### What actually crosses the boundary

Only the goal targets. `_visit_targets` walks the incoming PDDL goal, accepts
`visited-{place,object,poi}` and `at-{place,object}`, and takes each atom's
last argument. Grounding then emits a fresh problem:

```
(and (visited-object o8))        ── _visit_targets ──▶  ["o8"]

(define (problem omniplanner_grstaps)
  (:domain dsg-visit)
  (:objects o8 - location)
  (:init   (is-location o8))
  (:goal   (and (visited o8)))
  (:metric minimize (total-time)))
```

Two idioms in `dsg-visit` are copied deliberately from `wpc_spread`, and both
are load-bearing rather than stylistic:

- **`is-location` as a static marker.** It exists only to give the action a
  *positive* precondition. Without one, the grounder's reachability analysis
  prunes the action and the whole chain collapses.
- **Latching positives only.** Never write `not` on a predicate absent from
  `:init` — it breaks the SAS translator's binary-variable assumption.

### What the simplification costs

- `pick-object`, `place-object` and `inspect` are gone. The only thing
  expressible is "be at this location".
- Region-level goals (`explored-region`), and the `safe` / `suspicious`
  predicates, have no equivalent.
- **The robot binding is discarded.** Taking the last argument turns
  `(at-object hilbert o15)` into `o15` — "robot hilbert must reach o15" becomes
  "someone reaches o15". For GRSTAPS-X that is arguably correct, since
  assignment is its job, but a goal naming a specific robot is not honoured as
  written.

The ceiling here is what we generate, not what the solver accepts: GRSTAPS-X's
own `b45` example runs 12 actions. Richer behaviour means extending
`DOMAIN_PDDL` **and** adding matching `action_trait_config.json` entries — which
is also the point at which trait vectors and coalitions start to mean something.

---

## Constraints

**GRSTAPS-X never sees a constraint.** There is no constraint syntax in the
mission PDDL and none is passed to the solver. Instead, `ground_problem` prunes
the motion graph before writing it:

1. `_forbidden_positions` resolves each `forbidden-poi` symbol to a position.
2. `_prune_graph` finds every vertex within `forbidden_radius_m` (default 5.0 m)
   of any of those positions and **drops all of its edges**.

Vertices are kept but isolated, because vertex ids are indices into the vertex
list — deleting entries would renumber every edge in the file.

The result is stronger than the PDDL path's approach. Dropping `connected`
facts between POIs still leaves the metric path free to cut straight through
the forbidden area; here A\* physically cannot route through an isolated vertex.

### What is not honoured — read this before trusting a constraint

- **`forbidden-edge` is silently ignored.** The PDDL grounder handles it
  (`_extract_forbidden_sets` returns both sets); the GRSTAPS path reads only
  `forbidden-poi`. A `forbidden-edge` constraint will appear to be accepted and
  will have no effect.
- **The radius is euclidean here, path distance in the PDDL grounder.** A place
  that is metrically near a forbidden POI but only reachable the long way round
  is pruned here and kept there. The two planners will not always agree.
- **A forbidden POI near a goal makes the goal unreachable.** Pruning runs
  before the goal is snapped to a vertex, so if the target's vertex is isolated
  the solver fails with *"Searched the entire space"* rather than reporting the
  conflict.
- A symbol not present in the DSG logs a warning and is skipped, not rejected.

Constraints reach the planner from two places and both are merged: persistent
ones accumulated by `omniplanner_repair_node` on the node, and per-message ones
carried on the goal. This matches `MultiRobotPddlConstrained`, so switching
planners does not silently drop constraints a user already stated.

---

## The repair loop

`goal_manager` skips a replan when the current plan already satisfies a new
goal. That decision needs to know which POIs the active plan visits, which the
planner publishes:

```python
def on_plan_compiled(self, plans, plan_dict):   # grstaps_planner_ros.py
    # -> /<robot>/omniplanner_node/plan_visited_pois
```

Per-robot coverage is exact rather than inferred: `agents[].individual_plan`
holds the solver's own task ordering, and each task's name (`"visit-location o8
:: visit"`) names its target.

Without this hook the repair flow still behaves *correctly* — an empty cache
reads as "no plan yet" and forces a replan — but it never skips anything, which
is the entire point of plan repair.

---

## Configuration

Two configs, differing only in how a goal arrives:

| config | `goal_format` | subscribes | driven by |
|---|---|---|---|
| `grstaps` | `points` (default) | `GotoPointsGoalMsg` on `grstaps_planner/grstaps_goal` | a manual point list |
| `grstaps_repair` | `constrained_pddl` | `ConstrainedPddlGoalMsg` on `grstaps_planner/pddl_goal` | goal_manager / heracles agent |

Experiments: `spot_grstaps`, `spot_grstaps_repair`,
`spot_grstaps_heracles_repair`.

The repair experiments pass `planner_goal_topic:=grstaps_planner/pddl_goal` to
`master.launch.yaml`, which is the whole of the planner switch. The argument
defaults to `multi_robot_pddl_constrained/pddl_goal`, so every existing
experiment is unaffected.

> `grstaps_repair` is a standalone config, not an inheritor of `grstaps`.
> `resolve_override_dirs` collects only *leaf* keys, so a non-leaf parent
> contributes no override files — `grstaps_repair: [grstaps]` would silently
> generate a config with no GRSTAPS plugin in it at all.

### Environment

Nothing machine-specific belongs in a tracked config; paths come from the
environment, and every one has a default.

| variable | default | meaning |
|---|---|---|
| `ADT4_GRSTAPS_ROOT` | `~/grstapsx` | repo root; scenarios are written under `data/` |
| `ADT4_GRSTAPS_BINARY` | `~/grstapsx/build-native/grstapsx_example` | native solver binary |
| `ADT4_GRSTAPS_CONDA_PREFIX` | `~/miniconda3/envs/grstapsx` | env supplying OMPL etc. |
| `ADT4_GUROBI_LICENSE` | `~/gurobi.lic` | Gurobi licence |
| `ADT4_GUROBI_HOME` | `~/opt/gurobi1103/linux64` | Gurobi install |

`run_mode` selects `native` (Named-User licence, no network) or `docker` (WLS
licence, 2 concurrent sessions — hence the licence retry logic in
`_run_solver`). With none of this installed the plugin simply fails at goal
time; the rest of omniplanner is unaffected.

Building the solver natively needs OMPL **1.6.x** — `find_package(ompl 1.6.0)`
with `SameMajorVersion` rejects 2.x, and the upstream `environment.yml` does not
pin it.

---

## Reading the log

```
GRSTAPS-X solved: makespan=49.834, 1 task(s), 1 agent(s)
  visit-location o8 :: visit      44.83 ->   49.83 s  coalition=[0]
  robot hilbert: task order [0], 1 leg(s), 18 waypoint(s)
      1. visit-location o8       -> o8     (O8)   44.83 -> 49.83 s
      leg 1 (18 pts): t4330 -> t4907 -> t35 -> t4871 -> t3 -> t4887 -> t73
                   -> t140 -> t283 -> t824 -> t417 -> t299 -> t604 -> t990
                   -> t1405 -> t1139 -> t1104 -> t1015
```

The numbered list is the solver's actual sequencing. The leg is the metric
route rendered as DSG place symbols — `symbol_of_vertex` inverts the vertex
numbering the solver works in. Reading the route as `t####` symbols is how you
tell a genuine traversal from one that cuts through a wall, which bare
coordinates will not show you.

---

## Measured against fast-downward

Same DSG, robot poses and targets; targets farthest-point sampled so allocation
actually matters.

| case | planner | solve | total | max robot |
|---|---|---|---|---|
| 1279 places, 3 robots, 9 targets | fast-downward | 46.5 s (5.1 GB peak) | 363.8 m | 194.0 m |
| 1279 places, 3 robots, 9 targets | GRSTAPS-X | **1.2 s** | **277.7 m** | **156.4 m** |
| 96 places, 3 robots, 9 targets | fast-downward | 1.9 s | — | 54.7 m |
| 96 places, 3 robots, 9 targets | GRSTAPS-X | **0.3 s** | — | **45.7 m** |

The objectives genuinely differ: the multirobot PDDL domain minimises
`total-cost`, a sum, so concentrating work on one robot is free; GRSTAPS-X
minimises **makespan**, so it has a reason to spread work across the fleet.

---

## Limitations

This is a proof of concept and is not ready to replace fast-downward.

- **One action type.** The mission domain has only `visit-location`. No pick,
  place, or ordering beyond what the scheduler derives.
- **Trait vectors are trivial.** Every robot and subtask is `[1,0,...]`, so
  allocation has no capability signal to reason about — the fleet is
  homogeneous by construction.
- **The schedule is discarded.** `compile_plan` emits route geometry only,
  because `ActionSequenceMsg` cannot carry the start/finish times GRSTAPS-X
  computes — which is most of what the solver is for.
- **Single-robot compilation.** `compile_plan` handles one robot's assignment;
  multi-robot output would need a `MultiRobotWrapper` path.
- **One `Follow` per route, not per leg**, which is the likely cause of the
  executor cutting corners between waypoints.
- `forbidden-edge` unsupported — see [Constraints](#constraints).
