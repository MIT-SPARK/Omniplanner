import logging
import os
import subprocess
import tempfile

from dsg_pddl.pddl_grounding import GroundedPddlProblem
from dsg_pddl.pddl_utils import lisp_string_to_ast

logger = logging.getLogger(__name__)


def solve_pddl(problem: GroundedPddlProblem):
    """Use fast-downward to solve the given pddl problem"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        problem_fn = os.path.join(tmpdirname, "problem.pddl")
        domain_fn = os.path.join(tmpdirname, "domain.pddl")
        plan_fn = os.path.join(tmpdirname, "plan.txt")

        with open(problem_fn, "w") as fo:
            fo.write(problem.problem_str)

        with open(domain_fn, "w") as fo:
            fo.write(problem.domain.to_pddl_string())

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
