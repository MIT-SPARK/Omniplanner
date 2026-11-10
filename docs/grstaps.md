# GRSTAPS-X integration reference

The [README](../README.md) describes the current pipeline, algorithms, and
supported behavior. This replaces the earlier visit-only prototype guide.

## Source map

| Component | Source |
| --- | --- |
| Domain, goal parsing, grounding, solver invocation | [`grstaps_planner.py`](../omniplanner/src/omniplanner/grstaps_planner.py) |
| ROS callbacks, action compilation, coverage/schedule | [`grstaps_planner_ros.py`](../omniplanner_ros/src/omniplanner_ros/grstaps_planner_ros.py) |
| Generic dispatch and wrappers | [`omniplanner.py`](../omniplanner/src/omniplanner/omniplanner.py) |
| Keep/replan decision | [`goal_manager_node.py`](../omniplanner_ros/src/omniplanner_ros/goal_manager_node.py) |
| Executor gate state | [`plan_progress.py`](../../spot_tools/robot_executor_interface/robot_executor_interface/src/robot_executor_interface/plan_progress.py) |
| Spot execution | [`spot_executor.py`](../../spot_tools/spot_tools/src/spot_executor/spot_executor.py) |
| Agent prompt | [`pddl_domain_description_grstaps.yaml`](../../heracles_agents/examples/prompts/common/pddl_domain_description_grstaps.yaml) |
| Robot pinning, mission state, setup knobs | [`tests/test_grstaps_*.py`](../tests/) |
| Solver build in a colcon workspace | [`grstapsx` `cmake/FetchDependencies.cmake`](https://github.com/jinwoop/grstapsx/blob/feature/colcon/cmake/FetchDependencies.cmake) |

Algorithm descriptions were checked against the installed GRSTAPS-X source:
`include/grstapsx/task_planning_full/task_plan_search.hpp`,
`include/grstapsx/task_allocation/itags/itags.hpp`,
`src/task_allocation/itags/time_extended_task_allocation_quality.cpp`,
`normalized_schedule_quality.cpp`, and
`src/geometric_planning/motion_planners/euclidean_graph_motion_planner.cpp`.
The copied `wpc_spread/action_trait_config.json` selects deterministic MILP
scheduling. OmniPlanner does not expose all upstream solver modes.

## Paths and artifacts

The `itags` entry needs nothing from a grstapsx checkout: the binary comes
from `PATH` (a colcon build installs it), the solver parameters are built in
(`SOLVER_PARAMS`, copied from GRSTAPS-X's `wpc_spread` scenario), and problem
files go to `work_dir`. The `pddl` entry and Docker mode still run from a
checkout:

| Setting (environment variable) | Default | Used by |
| --- | --- | --- |
| `itags_binary` (`ADT4_ITAGS_BINARY`) | `itags` on `PATH`, else `~/grstapsx/build-native/itags` | `itags` |
| `work_dir` (`ADT4_GRSTAPS_WORK_DIR`) | `~/.cache/omniplanner/grstaps` | `itags` |
| `license_path` (`ADT4_GUROBI_LICENSE`) | `~/gurobi.lic`; unset if missing, so Gurobi searches its defaults | both |
| `repo_root` (`ADT4_GRSTAPS_ROOT`) | `~/grstapsx` | `pddl`, Docker |
| `native_binary` (`ADT4_GRSTAPS_BINARY`) | `~/grstapsx/build-native/grstapsx_example` | `pddl` |
| `gurobi_home` / `conda_prefix` (`ADT4_GUROBI_HOME`, `ADT4_GRSTAPS_CONDA_PREFIX`) | added to `LD_LIBRARY_PATH` only if they exist | hand-built binaries |

The `repo_root`, `native_binary`, `gurobi_home` and `conda_prefix` defaults read
the environment when Python starts, so set those before launching. `run_mode`
defaults to Docker; the overlays set native mode, which the ITAGS entry
requires. `solver_params_from` names a scenario in the checkout to copy solver
parameters from instead of the built-in ones. Every solver call is limited by
`solver_timeout_s`: the search and MILP timeouts are capped to it, and the
process is killed 30 s after it.

Grounding writes:

- `maps/ground_graph.json`: vertices, weighted edges, and applied exclusions.
- `action_trait_config.json`: species, starts, traits, action templates, and
  copied allocation/scheduling parameters.
- `domain.pddl` / `problem.pddl`: generated mission logic and goals.
- `itags_input.json`: explicit tasks and precedence, for `entry: itags`.

ITAGS returns `<scenario_dir>/itags_solution.json`. The PDDL example returns
`<binary_build_dir>/solutions/<scenario_name>/problem/itags_files/itags_solution.json`
(native) or the equivalent path under `build/` (Docker). Stale results are
removed before solving. The `itags` entry writes to
`<work_dir>/<scenario_name>_<robot>`, so planners for different robots never
share files; for the `pddl` entry use distinct `scenario_name` values.

A graph location's coordinates must equal its snapped vertex coordinates;
robot starts also need entries in the PDDL configuration's `nodes` table.
The ITAGS JSON embeds complete start configurations directly.

## Execution contract

Coalition inspection is **one task assigned to several robots**, not separate
independent observer tasks. Each participant approaches, waits for predecessors,
announces READY, and waits for every partner. Dependent tasks use qualified
completion IDs such as `0@hilbert` so all coalition members must finish.
Both qualified and ordinary DONE events are emitted for compatibility with solo
predecessors. Deploy planner and Spot executor changes together.

Zero-travel tasks retain a one-point `Follow` checkpoint. The executor checks
actual arrival and navigates from its current pose if it has drifted. Atomic
relocation emits `Pick → Follow(carry) → Place` with a shared task ID; carry
routing uses the pruned solver graph. An action that fails its retries is
skipped, as on the other planners, and its task still counts as done, so
partners are not left waiting on it. A robot waiting on a partner gives up the
plan after the executor's `plan_wait_timeout_s` (default 900 s).

`TaskScheduleMsg` has a scalar `robot_name`, so a coalition is published as one
schedule row per participant with the same target and timepoints. The reported
makespan is unchanged. Timepoints, nominal durations, and species speeds are
planning estimates, not execution deadlines.

See [remaining work](grstaps-roadmap.md).
