# OmniPlanner + GRSTAPS-X

OmniPlanner turns goals grounded in a Hydra **dynamic scene graph (DSG)** into
robot actions. It supports Fast Downward/PDDL, TSP, point navigation, and
**GRSTAPS-X** (the solver's repository name). GRSTAPS-X adds capability-based
fleet allocation, coalition tasks, precedence, and scheduling.

**Validation:** both native solver entry points and the ROS adapter-to-action
pipeline have been exercised on a real saved DSG and in the two-robot
`run-adt4` sim. This is not yet a verified live LLM-to-robot deployment. See
[remaining work](docs/grstaps-roadmap.md).

## Pipeline

```text
Hydra DSG + robot poses/types                      User instruction
            │                                            │
            │                              Heracles: look up DSG symbols,
            │                              emit goal + constraints
            │                                            │
            │                               /<robot>/commanded_goal
            │                                            ▼
            │                              goal_manager: keep plan or replan
            │                                            │
            └───────────────────────► OmniPlanner ROS plugin
                                             │
                               ground_problem → make_plan → compile_plan
                                             │
                              per-robot ActionSequence + fleet schedule
                                             │
                                Spot executors ↔ /plan_progress
```

1. **Ground:** resolve symbols, snap robot starts and task locations to DSG
   mesh-place vertices, and build the fleet and constrained motion graph.
2. **Solve:** dispatch by problem type. Fast Downward searches a grounded PDDL
   problem; GRSTAPS-X allocates and schedules tasks using a separate motion graph.
3. **Compile:** preserve each robot's task order and turn routes/tasks into
   `Follow`, `Gaze`, `Pick`, and `Place` actions. Every participating robot gets
   a sequence, including an empty sequence if it receives no work.
4. **Execute:** Spot gates actions on predecessor completion and coalition
   readiness. Robots share a plan ID and publish `READY`/`DONE` progress.
   Scheduled start/finish times are published for inspection; execution does
   **not** wait for those timestamps.
5. **Repair:** the goal manager can keep an existing plan for a covered visit
   goal with unchanged constraints. Richer predicates trigger replanning.
   This is a coverage check, not proof that physical work has completed.

## What GRSTAPS-X receives and returns

| Boundary | Data |
| --- | --- |
| OmniPlanner input | DSG; robot poses and types; visit/inspect/relocate targets; constraints; planner configuration |
| Solver input | Tasks with durations, required trait vectors and start/end locations; robot species, traits, speeds and starts; weighted motion graph; precedence pairs |
| Solver output | `itags_solution.json`: makespan, task start/finish times, task coalitions, each agent's ordered task IDs and transition paths, precedence |
| Runtime output | One `ActionSequence` per robot; `~/task_schedule`; per-robot `plan_visited_pois` and fleet `~/plan_covered_pois` |

There are **two entry points**:

| Entry | Task generation | Use |
| --- | --- | --- |
| `entry: itags`, `run_mode: native` | OmniPlanner enumerates tasks and precedence into `itags_input.json`; no PDDL task search runs | **Configured repair/deployment path**; supports `before` and relocation destinations |
| `entry: pddl` | Generated `domain.pddl`, `problem.pddl`, and `action_trait_config.json` enter `grstapsx_example` | Native/Docker task-planning path; explicit ordering and relocation destinations are rejected by this adapter |

Both write `maps/ground_graph.json` under
`$ADT4_GRSTAPS_ROOT/data/grstapsx/<scenario_name>/`. The ITAGS entry also writes
PDDL files for inspection, but the solver does not read them.

## Which algorithms run?

- **Task planning:** on the PDDL entry, GRSTAPS-X's imported temporal task planner
  grounds PDDL into SAS and searches task plans with causal ordering. The
  configured ITAGS entry skips this search: visit → one task, inspection → one
  coalition task, relocation → one atomic pick–carry–place task.
- **Allocation:** ITAGS (*Incremental Task Allocation Graph Search*) uses greedy
  best-first search over robot–task assignments. Coalition traits add together
  to satisfy a task's requirements. Its TETAQ score is
  `α × APR + (1 − α) × NSQ`: remaining unmet traits versus normalized schedule
  makespan. The default builder weight is `α = 0.5`.
- **Scheduling:** a deterministic **Gurobi MILP** schedules the allocation using
  task/travel durations, precedence, and shared-robot resource constraints,
  optimizing makespan (the last task's finish time). Greedy allocation and
  solver time limits mean the fleet result is not a global-optimality guarantee.
- **Routing:** **A\*** on the weighted Euclidean DSG graph supplies routes and
  travel times. Carry paths use weighted shortest paths on that same constrained
  graph. This adapter does not enable collision-free multi-agent path planning.

The adapter keeps geometry outside mission PDDL. All species currently share
one 2D ground graph, including a configured UAV; UAV speed/traits do not provide
an aerial navigation or execution stack.

## Goals, capabilities, and constraints

| Request | Result |
| --- | --- |
| `(visited-object o8)` / `(visited-place t299)` | Visit a target; requires a sensor |
| `(safe o4)` / `(not (suspicious o4))` | Coalition inspection; compiles to `Gaze` |
| `(visited-object-by hamilton o8)` / `(visited-place-by hamilton t299)` | A visit only that robot may do; requires `entry: itags` |
| `(inspected-by hamilton o4)` | A coalition inspection whose team includes that robot; requires `entry: itags` |
| `(object-in-place o4 t598)` | Atomic relocation by a manipulator; requires `entry: itags` |
| `[["before", "o4", "o8"]]` | Every task for `o4` must finish before any task for `o8`; ITAGS only |
| `[["forbidden-poi", "o5"]]` | Isolate graph vertices within the configured path-distance radius |
| `[["forbidden-edge", "t3", "t4"]]` | Remove the undirected edge between the snapped endpoints |

Example: move an object, then visit another:

```text
goal:        (and (object-in-place o4 t598) (visited-object o8))
constraints: [["before", "o4", "o8"]]
```

IDs are map-specific; look them up in the current DSG. Missing goal/destination
IDs and contradictory precedence are rejected. A `forbidden-poi` radius uses
weighted graph distance in metres (default 1 m, shared with fast-downward and
overridden by `ADT4_FORBIDDEN_RADIUS_M`); it removes incident edges,
not continuous geometric regions. Invalid constraint symbols are still warned
and skipped, and an absent forbidden edge has no effect.

Species profiles are `spot` (ground + sensor, 1 m/s), `spot_arm` (also a
manipulator), and `uav` (air + sensor, 2 m/s). Default inspection requires ground
+ air + two sensors. The repair overlay instead requests **two ground robots**
and maps every Spot to `spot_arm`. Huskies are excluded because their executor
lacks the gates.

A task pinned to a robot by a `-by` predicate gets that robot's *identity
trait*: grounding gives each robot in the fleet one extra trait dimension only
it carries, and the pinned task demands it, so the allocator must hand the task
to that robot (or include it in the inspection team). Unpinned tasks are left
to the allocator. A goal that pins work to a robot with no pose fails and says
so; the robots then stop, as for any goal that cannot be planned.

**Goal-language limits:** this is a restricted PDDL adapter. Disjunctions (`or`),
robot-specific final poses (`at-*`), pick-and-hold (`holding`), unknown predicates,
and negative goals other than `(not (suspicious ...))` are rejected. The GRSTAPS
agent prompt uses the supported forms above. Inspection commands a gaze; it
does not itself prove an object is safe. See [remaining work](docs/grstaps-roadmap.md).

## Setup

GRSTAPS-X itself lives in its own repository,
[jinwoop/grstapsx](https://github.com/jinwoop/grstapsx); its README explains the
solver, its data formats and its other build options (Docker, conda). In this
workspace it is built like any other package:

1. `vcs import src < src/awesome_dcist_t4/install/packages.yaml` clones it to
   `src/grstapsx` (branch `feature/colcon`).
2. `rosdep install --from-paths src --ignore-src -r -y` installs its apt
   dependencies (Eigen, Boost, fmt, spdlog, yaml-cpp, nlohmann-json, range-v3,
   `ros-jazzy-ompl`).
3. `colcon build` builds it. Gurobi and cppcoro are downloaded during the build
   when they are not installed, and `itags` is installed on `PATH` with Gurobi
   next to it, so nothing else needs to be set.
4. **Gurobi license (once per machine, by hand).** The solver needs a license
   that the build cannot fetch for you. Academic licenses are free: create one
   at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/)
   and activate it **on the machine that runs omniplanner** (the base station),
   from the university network:

   ```bash
   $ADT4_WS/build/grstapsx/_deps/gurobi_dist-src/linux64/bin/grbgetkey <key>
   ```

   It writes `~/gurobi.lic`, which Gurobi finds by default. A named-user
   academic license is tied to that machine's hostname: copying it to another
   computer does not work. Elsewhere, set `GRB_LICENSE_FILE` or
   `ADT4_GUROBI_LICENSE` to the license file.

Check the setup before a run:

```bash
which itags        # installed by colcon on PATH
```

A failed solve is reported on the planner's alert topic and the robots stop;
"No Gurobi license" in that message means step 4.

## Configuration

The launch overlay in
[`dcist_launch_system`](../dcist_launch_system/config_generation/experiment_overrides/grstaps_repair/omniplanner_plugins_overlay.yaml)
selects:

```yaml
planners:
  grstaps_planner:
    plugin:
      type: Grstaps
      run_mode: native
      entry: itags
      goal_format: constrained_pddl
      robot_type_species: {spot: spot_arm, simulated-spot: spot_arm}
      inspect_requirement: {ground: 2, sensor: 2}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `entry` | `pddl` | `itags` sends tasks and ordering straight to the solver; needed for `before`, relocation destinations and `-by` |
| `run_mode` | `docker` | `native` runs the local `itags` binary (required by `entry: itags`) |
| `robot_type_species` | `{}` | `{robot_type: species}` for the whole fleet, read from `omniplanner_plugins.yaml`'s `robot_type` |
| `robot_species` | `{}` | `{robot_name: species}` for one robot; wins over the type |
| `inspect_requirement` | ground 1, air 1, sensor 2 | What an inspection's team must cover together. `{ground: 2, sensor: 2}` needs two Spots **with poses** when the goal is planned |
| `solver_timeout_s` | `120` | Longest a solve may take; past it the solve fails instead of blocking the planner |
| `work_dir` | `~/.cache/omniplanner/grstaps` | Where problem and solution files go, one subdirectory per robot |
| `itags_binary` | `itags` on `PATH` | Path to the solver binary |
| `visit/inspect/pick/place_duration_s` | 5 / 10 / 10 / 10 | Nominal task durations used for scheduling |
| `forbidden_radius_m` | `ADT4_FORBIDDEN_RADIUS_M` or 1 m | How far `forbidden-poi` reaches along the graph |

Environment variables override the defaults without editing a config:
`ADT4_ITAGS_BINARY`, `ADT4_GRSTAPS_WORK_DIR`, `ADT4_GUROBI_LICENSE`. The
`pddl` entry and Docker mode still read a grstapsx checkout
(`ADT4_GRSTAPS_ROOT`, `ADT4_GRSTAPS_BINARY`); see
[path defaults and source map](docs/grstaps.md).

The repair launch sets `planner_goal_topic:=grstaps_planner/pddl_goal`.
On hardware, run `base_station_grstaps_heracles_repair` on the base station and
the robots' usual `spot-default` sessions. The `spot_grstaps*` sessions are for
simulation only: their executor drives a fake Spot.

Before a hardware run:

- Build `spot_tools` from the same branch on **every robot and the base
  station**. `ActionMsg` gained ordering fields and `PlanProgressMsg` is new;
  mismatched message definitions do not interoperate.
- Robots wait for each other over the global `/plan_progress` topic. Check it
  crosses between robots: `ros2 topic echo /plan_progress` on one robot while
  the other finishes a task.
- A robot waiting on a partner gives up after `plan_wait_timeout_s` (Spot
  executor parameter, default 900 s; 0 waits forever) and logs every 30 s
  what it is waiting on.

Run the unit and solver tests from the colcon workspace root (the solver tests
skip without the saved map or the solver):

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
"${ADT4_ENV:-$HOME/environments/dcist}/spark_env/bin/python" -m pytest \
  src/awesome_dcist_t4/omniplanner/tests -q
```

## Plan repair: mission progress and constraints

`omniplanner_repair_node` is a drop-in replacement for `omniplanner_node` that
remembers what the robots have already done during a mission. When a new goal
or constraint arrives (e.g. "avoid object o14"), it replans only the part of
the mission that is left, instead of starting over. In Awesome-DCIST-T4 it is
launched with `launch_omniplanner_repair:=true`, which also starts the
`goal_manager` that forwards operator goals to it.

The node adds the following interfaces. Private names (`~/...`) resolve under
the node's namespace and name; for example, a node named `omniplanner_node` in
namespace `base_station` exposes
`/base_station/omniplanner_node/reset_mission_state`.

| Interface | Type | Purpose |
| --- | --- | --- |
| `~/<plugin>/pddl_goal` (subscription) | `omniplanner_msgs/msg/ConstrainedPddlGoalMsg` | A goal plus constraints for that goal only, for a `MultiRobotPddlConstrained` plugin (e.g. `~/multi_robot_pddl_constrained/pddl_goal`). The goal's constraints are combined with the persistent list below. They are not remembered for later goals. |
| `~/constraints` (subscription) | `omniplanner_msgs/msg/ConstraintList` | Sets the persistent constraint list applied to every later goal. Each message replaces the previous list; publish an empty list to clear it. |
| `/action_done` (subscription) | `robot_executor_msgs/msg/ActionDoneMsg` | Records actions the executors finished successfully (see below). This topic is global unless remapped. |
| `~/reset_mission_state` (service) | `std_srvs/srv/Trigger` | Forgets visited places and inspected objects, for starting a new mission. Holding state and constraints are kept. It does not stop execution or trigger replanning. |

What each successful action on `/action_done` records (failed actions record
nothing):

* `FOLLOW`: places and objects along the finished path are marked visited:
  those within `visited_radius_m` (parameter, default `1.0` m) of a path
  waypoint, plus any object whose nearest place is a waypoint. (A route to an
  object stops at its nearest place, which can be more than 1 m away.)
* `GAZE`: the object is marked inspected.
* `PICK` / `PLACE`: the robot is marked as holding, or no longer holding, the
  object. This is only a fallback: when the scene graph says what a robot
  holds, the scene graph wins.

This progress is kept in memory, so restarting the node forgets it.

Reset progress before starting a new mission (replace the namespace and node
name to match your launch):

```bash
ros2 service call /base_station/omniplanner_node/reset_mission_state std_srvs/srv/Trigger '{}'
```

The reply is `success: true` with the message `forgot visited places and
inspected objects`. To clear the persistent constraints:

```bash
ros2 topic pub --once /base_station/omniplanner_node/constraints omniplanner_msgs/msg/ConstraintList '{constraints: []}'
```
