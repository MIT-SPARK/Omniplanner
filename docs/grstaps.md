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
| Regression/native integration checks | [`test_grstaps_integration.py`](../tests/test_grstaps_integration.py) |

Algorithm descriptions were checked against the installed GRSTAPS-X source:
`include/grstapsx/task_planning_full/task_plan_search.hpp`,
`include/grstapsx/task_allocation/itags/itags.hpp`,
`src/task_allocation/itags/time_extended_task_allocation_quality.cpp`,
`normalized_schedule_quality.cpp`, and
`src/geometric_planning/motion_planners/euclidean_graph_motion_planner.cpp`.
The copied `wpc_spread/action_trait_config.json` selects deterministic MILP
scheduling. OmniPlanner does not expose all upstream solver modes.

## Paths and artifacts

| Environment variable | Default |
| --- | --- |
| `ADT4_GRSTAPS_ROOT` | `~/grstapsx` |
| `ADT4_GRSTAPS_BINARY` | `~/grstapsx/build-native/grstapsx_example` |
| `ADT4_ITAGS_BINARY` | `~/grstapsx/build-native/itags` |
| `ADT4_GRSTAPS_CONDA_PREFIX` | `~/miniconda3/envs/grstapsx` |
| `ADT4_GUROBI_HOME` | `~/opt/gurobi1103/linux64` |
| `ADT4_GUROBI_LICENSE` | `~/gurobi.lic` |

Set these **before starting Python/ROS**: dataclass defaults read the environment
at import time. Changing the repository root does not automatically change the
binary defaults. `run_mode` defaults to Docker; the repair overlay explicitly
sets native mode. The ITAGS adapter invokes a local binary and requires native
mode. Native linking uses Gurobi and conda library directories.

`solver_params_from` defaults to `wpc_spread`; its configuration must exist in
the solver checkout. Grounding writes:

- `maps/ground_graph.json`: vertices, weighted edges, and applied exclusions.
- `action_trait_config.json`: species, starts, traits, action templates, and
  copied allocation/scheduling parameters.
- `domain.pddl` / `problem.pddl`: generated mission logic and goals.
- `itags_input.json`: explicit tasks and precedence, for `entry: itags`.

ITAGS returns `<scenario_dir>/itags_solution.json`. The PDDL example returns
`<binary_build_dir>/solutions/<scenario_name>/problem/itags_files/itags_solution.json`
(native) or the equivalent path under `build/` (Docker). Stale results are
removed before solving. Use distinct `scenario_name` values for concurrent
planner instances: the default directory is shared, not request-isolated.

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
routing uses the pruned solver graph. A scheduled action that exhausts retries
stops the sequence without announcing DONE.

`TaskScheduleMsg` has a scalar `robot_name`, so a coalition is published as one
schedule row per participant with the same target and timepoints. The reported
makespan is unchanged. Timepoints, nominal durations, and species speeds are
planning estimates, not execution deadlines.

See [validation evidence](integration-audit.md) and [remaining work](grstaps-roadmap.md).
