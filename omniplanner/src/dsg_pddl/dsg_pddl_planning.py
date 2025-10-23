import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import spark_dsg
from plum import dispatch

from dsg_pddl.dsg_pddl_grounding import GroundedPddlProblem, PddlDomain, PddlSymbol
from dsg_pddl.pddl_utils import lisp_string_to_ast
from omniplanner.tsp import LayerPlanner

logger = logging.getLogger(__name__)


@dataclass
class PddlPlan:
    domain: PddlDomain
    symbolic_actions: List[tuple]
    parameterized_actions: List
    symbols: Dict[str, PddlSymbol]


def solve_pddl(problem: GroundedPddlProblem):
    """Use fast-downward to solve the given pddl problem"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        problem_fn = os.path.join(tmpdirname, "problem.pddl")
        domain_fn = os.path.join(tmpdirname, "domain.pddl")
        plan_fn = os.path.join(tmpdirname, "plan.txt")

        with open(problem_fn, "w") as fo:
            fo.write(problem.problem_str)

        with open(domain_fn, "w") as fo:
            fo.write(problem.domain.to_string())

        command = ["fast-downward"]
        command += ["--plan-file", plan_fn]
        command += [domain_fn]
        command += [problem_fn]
        command += [
            "--search",
            "let(hff, ff(), let(hcea, cea(), lazy_greedy([hff, hcea], preferred=[hff, hcea])))",
        ]

        logger.warning(f"Calling: {command}")
        return_code = subprocess.run(command)
        logger.warning(f"Return code: {return_code}")

        if os.path.exists(plan_fn):
            with open(plan_fn, "r") as fo:
                lines = fo.readlines()
        else:
            output_dir = os.getenv("ADT4_OUTPUT_DIR", "")
            debug_fn = os.path.join(output_dir, "pddl_problem_debugging.pddl")
            logger.warning(
                f"Planning failed. Please see {debug_fn} for the failed problem file."
            )
            with open(debug_fn, "w") as fo:
                fo.write(problem.problem_str)
            raise Exception(
                f"Planning failed, please see {debug_fn} for failed problem file."
            )

    plan = [lisp_string_to_ast(line) for line in lines[:-1]]
    return plan


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

    logger.warning(f"Made plan {plan}")

    parameterized_plan = []

    # TODO: generalize this post-processing (to be based on either the pddl
    # problem name, or some other post-processing type associated with the
    # grounded pddl problem?)

    layer_planner = LayerPlanner(map_context, spark_dsg.DsgLayers.MESH_PLACES)
    last_pose = np.zeros(2)
    multirobot = "multirobot" in grounded_problem.domain.domain_name
    for p in plan:
        match p[0]:
            case "goto-poi":
                if multirobot:
                    path, last_pose = parameterize_goto_poi_multirobot(
                        layer_planner, grounded_problem.symbols, p
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

    return PddlPlan(
        grounded_problem.domain, plan, parameterized_plan, grounded_problem.symbols
    )
