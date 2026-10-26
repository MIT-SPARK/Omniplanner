import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import numpy as np
import spark_dsg
from plum import dispatch

from dsg_pddl.dsg_pddl_grounding import GroundedPddlProblem, PddlDomain, PddlSymbol
from dsg_pddl.pddl_planning import solve_pddl
from omniplanner.tsp import LayerPlanner

logger = logging.getLogger(__name__)


@dataclass
class PddlPlan:
    domain: PddlDomain
    symbolic_actions: List[tuple]
    parameterized_actions: List
    symbols: Dict[str, PddlSymbol]
    # robot -> place symbols its goto paths pass through (goal-relevant plans,
    # whose one goto-poi is a whole route); the targets are in symbolic_actions
    traversed: Dict[str, Set[str]] = field(default_factory=dict)


def drop_index(t, k):
    return t[:k] + t[k + 1 :]


def parameterize_goto_poi(layer_planner, symbols, action):
    path = layer_planner.get_external_path(
        symbols[action[1]].position,
        symbols[action[2]].position,
    )
    last_pose = path[-1]
    return path, last_pose


def parameterize_goto_poi_multirobot(layer_planner, symbols, action):
    return parameterize_goto_poi(layer_planner, symbols, drop_index(action, 1))


def parameterize_inspect(layer_planner, symbols, action, last_pose):
    return [last_pose, symbols[action[1]].position]


def parameterize_inspect_multirobot(layer_planner, symbols, action, last_pose):
    return parameterize_inspect(
        layer_planner, symbols, drop_index(action, 1), last_pose
    )


def parameterize_pick_object(layer_planner, symbols, action, last_pose):
    # The nominal parameter for where the robot should be to pick the object
    # is the place that the object is in
    place_position = symbols[action[2]].position
    return [last_pose, place_position]


def parameterize_pick_object_multirobot(layer_planner, symbols, action, last_pose):
    return parameterize_pick_object(
        layer_planner, symbols, drop_index(action, 1), last_pose
    )


def parameterize_place_object(layer_planner, symbols, action, last_pose):
    place_position = symbols[action[2]].position
    return [last_pose, place_position]


def parameterize_place_object_multirobot(layer_planner, symbols, action, last_pose):
    return parameterize_place_object(
        layer_planner, symbols, drop_index(action, 1), last_pose
    )


@dispatch
def make_plan(grounded_problem: GroundedPddlProblem, map_context: Any) -> PddlPlan:
    plan = solve_pddl(grounded_problem)

    logger.debug(f"Made plan {plan}")

    parameterized_plan = []

    # TODO: generalize this post-processing (to be based on either the pddl
    # problem name, or some other post-processing type associated with the
    # grounded pddl problem?)

    # A goal-relevant grounding hands over the planner it measured distances on
    # (metres, constraints applied): its goto hops are whole routes and must be
    # expanded on that same graph, or a route could cross a forbidden area.
    grounded_planner = getattr(grounded_problem, "layer_planner", None)
    if grounded_planner is not None:
        layer_planner = grounded_planner
    else:
        layer_planner = LayerPlanner(map_context, spark_dsg.DsgLayers.MESH_PLACES)
    # Waypoint expansion must snap inside the same reachable component the
    # grounding used, or a goal position near an island place raises
    # NetworkXNoPath at compile time for a plan the solver already proved
    # (motion-tier v1 nl_s71: grounded fine, gate 1 verified, crashed here).
    # Single-robot grounding keys the start as "pstart"; the multirobot
    # grounding keys one start per robot as "pstart{robot_id}", so an exact
    # "pstart" lookup silently no-opped the restriction for every multirobot
    # plan. Take every pstart* symbol and restrict to the UNION of their
    # components: the robots' starts may live in different components, and
    # restricting to just one of them would make the other robots' places
    # unsnappable.
    start_positions = [
        start.position
        for name, start in grounded_problem.symbols.items()
        if name.startswith("pstart")
        and start is not None
        and getattr(start, "position", None) is not None
    ]
    if start_positions and grounded_planner is None:
        layer_planner = layer_planner.restricted_to_components(start_positions)
    last_pose = np.zeros(2)
    multirobot = "multirobot" in grounded_problem.domain.domain_name
    # Multi-robot plans interleave the robots' actions: each robot's next action
    # starts where that robot is, not where the previous action left another one.
    last_poses = {}
    traversed = {}
    for p in plan:
        if multirobot and len(p) > 1:
            robot = grounded_problem.symbols.get(p[1])
            last_pose = last_poses.get(
                p[1], robot.position if robot is not None else np.zeros(2)
            )
        match p[0]:
            case "goto-poi":
                if multirobot:
                    path, last_pose = parameterize_goto_poi_multirobot(
                        layer_planner, grounded_problem.symbols, p
                    )
                    if grounded_planner is not None:
                        a, b = (grounded_problem.symbols[x].position for x in p[2:4])
                        traversed.setdefault(p[1], set()).update(
                            spark_dsg.NodeSymbol(n).str(True).lower()
                            for n in grounded_planner.get_external_path_nodes(a, b)
                        )
                else:
                    path, last_pose = parameterize_goto_poi(
                        layer_planner, grounded_problem.symbols, p
                    )
                parameterized_plan.append(path)
            case "inspect":
                if multirobot:
                    path = parameterize_inspect_multirobot(
                        layer_planner, grounded_problem.symbols, p, last_pose
                    )
                else:
                    path = parameterize_inspect(
                        layer_planner, grounded_problem.symbols, p, last_pose
                    )
                parameterized_plan.append(path)
            case "pick-object":
                # The nominal parameter for where the robot should be to pick the object
                # is the place that the object is in
                if multirobot:
                    path = parameterize_pick_object_multirobot(
                        layer_planner, grounded_problem.symbols, p, last_pose
                    )
                else:
                    path = parameterize_pick_object(
                        layer_planner, grounded_problem.symbols, p, last_pose
                    )
                parameterized_plan.append(path)
            case "place-object":
                if multirobot:
                    path = parameterize_place_object_multirobot(
                        layer_planner, grounded_problem.symbols, p, last_pose
                    )
                else:
                    path = parameterize_place_object(
                        layer_planner, grounded_problem.symbols, p, last_pose
                    )
                parameterized_plan.append(path)
            case _:
                raise Exception(
                    f"Plan contains {p[0]} action, but I don't know how to parameterize!"
                )
        if multirobot and len(p) > 1:
            last_poses[p[1]] = last_pose

    return PddlPlan(
        grounded_problem.domain,
        plan,
        parameterized_plan,
        grounded_problem.symbols,
        traversed,
    )
