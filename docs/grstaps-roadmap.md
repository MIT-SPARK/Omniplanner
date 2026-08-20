# GRSTAPS-X: status, and what is left to reach run-adt4

The reason to run GRSTAPS-X instead of fast-downward is that it reasons about
**heterogeneous capability** (which robot *can* do this) and **time** (when, in
what order, how long). Both now work. What remains is wiring it to the live
system so a natural-language instruction reaches it.

## Where it stands

| capability | state | where |
|---|---|---|
| DSG → euclidean motion graph | done | `_mesh_place_graph` |
| species and trait vectors from `robot_type` | done | `SPECIES`, `species_for` |
| speed per species (UAV 2×) | done | `SPECIES` |
| capability gating (only `spot_arm` manipulates) | done | `base_traits` per subtask |
| coalitions (one task, several robots at once) | done | `inspect-object`, additive traits |
| task durations, configurable | done | `GrstapsDomain.*_duration_s` |
| MILP schedule with real timepoints | done | solver |
| precedence, intrinsic | done | `enumerate_tasks` |
| precedence, instruction-level (`before`) | done | `ordering_precedence` |
| precedence, raw index pairs | done | `GrstapsGoal.extra_precedence` |
| object relocation to a destination | done | `terminal_configuration` |
| one robot carries one object | done | `atomic_manipulation` |
| constraints: `forbidden-poi`, `forbidden-edge` | done | `_forbidden_sets`, `_prune_graph` |
| ITAGS entry point (skips PDDL task planning) | done | `entry="itags"`, `_run_itags` |
| multi-robot compile to `ActionSequence` | done | `compile_plan` on `MultiRobotWrapper` |
| schedule published for inspection | done | `TaskScheduleMsg` on `~/task_schedule` |
| schedule-aware replan decision | done | `goal_manager` `replan_horizon_s` |
| portable across maps | done | verified on two DSGs, one arg (`--map`) |
| **natural language → this planner, live** | **not done** | see below |

Two examples exercise everything above without ROS:
`examples/grstaps_example_omniplanner.py` (schedules, symbolic plan, both entry
points) and `examples/render_mission_video.py` (video).

## Deadlines: out of scope

Ordering is what this deployment needs, and precedence provides it. The solver
has deadline machinery (`RelativeDeadline`) but no config exposes it, so it
would be an upstream change. Revisit only if hard time windows appear.

---

# Remaining work: natural language to GRSTAPS-X under run-adt4

The target flow, and what each stage already does:

```
  "clear o4 and o5 out of the way, then go look at o8"
        │
        ▼  heracles agent (LLM)                          NEEDS PROMPT WORK
  goal:        (and (object-in-place o4 t598)(visited-object o8))
  constraints: [["before","o4","o8"], ["before","o5","o8"]]
        │
        ▼  pddl_repair_tool  →  ConstrainedPddlGoalMsg    works today
  /<robot>/commanded_goal
        │
        ▼  goal_manager                                   works today
  forwards to planner_goal_topic = grstaps_planner/pddl_goal
        │
        ▼  GrstapsRos.constrained_pddl_callback           works today
  parse_goal_targets  →  visit / inspect / manipulate + destinations
  constraints carried through untouched
        │
        ▼  ground_problem(entry="itags")                  NEEDS CONFIG
  enumerate_tasks   → tasks + intrinsic precedence
  ordering_precedence → `before` compiled to task-index pairs
        │
        ▼  itags binary                                   NEEDS ENV ON TARGET
  itags_input.json → itags_solution.json
        │
        ▼  compile_plan / on_plan_compiled                works today
  ActionSequence per robot + TaskScheduleMsg
```

## What the LLM emits, and what it does not

The agent works in **symbols**, never task indices. `before(a, b)` means every
task of symbol *a* finishes before any task of symbol *b*. Indices come from our
enumeration, which the agent cannot know and must not guess.

> **The high-level → detailed precedence conversion is already done, and is
> simpler than expected.** `ordering_precedence` expands a symbol-level
> `before` into every task-index pair it implies. And since a relocation is now
> **one atomic task** (`relocate-object`, pick and place indivisible), there is
> no pick/place split left to expand. A symbol maps to exactly one task except
> when it is both inspected and manipulated.

Trait vectors, durations and the domain stay **predefined**, exactly as the
fast-downward `.pddl` file is. The LLM never invents them.

## Steps

**1. Teach the agent the vocabulary it may use.** The repair prompt documents
only `forbidden-poi` and `forbidden-edge`. It needs:

- `before` as a constraint, with the symbol-level meaning spelled out
- the manipulation goal predicates that already parse: `(object-in-place ?o ?p)`
  for relocation, `(holding ?r ?o)` for a pick, `(not (suspicious ?o))` for a
  coalition inspection

These are the multirobot fast-downward domain's own predicates, so no new
vocabulary is invented — the agent is only permitted to use more of what the
domain already defines. Edit the repair-specific variants
(`pddl_domain_description_with_constraints.yaml`,
`pddl_in_context_examples_with_constraints.yaml`), never the shared files.

**2. Turn on the ITAGS entry point in the repair config.** The
`grstaps_repair` overlay sets `goal_format: constrained_pddl` but not `entry`,
so it would default to the PDDL path — where `before` cannot be expressed at
all. One line.

**3. Make the `itags` binary reachable.** It is built from a `tools/` target
added to grstapsx (`add_executable(itags ...)`), so any machine running this
needs that build. `ADT4_ITAGS_BINARY` overrides the default path.

**4. Test through the live node**, not just offline: agent → goal_manager →
plugin → solver → published plan and schedule.

## Risks worth naming

- **A cycle in the ordering aborts the solver** (`SIGABRT`, not a clean error).
  An LLM can easily emit contradictory `before` facts. A topological check on
  our side before writing the file would turn that into a readable message; we
  build the pairs, so it is cheap.
- **A symbol the graph lacks is dropped with a warning.** Combined with an LLM
  that invents ids, an ordering can silently not apply. The existing "confirm
  the id with run_cypher_query first" instruction matters more here than for
  `forbidden-poi`.
- **Goals are not portable between maps.** Place ids carry a prefix that
  differs by Hydra encoding (`t####` on one map, `P####` on another), and object
  ids do not correspond. A mission written for one DSG will not resolve on
  another — which is also why the agent must look ids up rather than recall them.
- **Only one manipulator serialises everything.** With four relocations and one
  `spot_arm`, the fleet cannot parallelise and the makespan is dominated by that
  one robot. This is the constraint doing its job, not the allocator failing.
