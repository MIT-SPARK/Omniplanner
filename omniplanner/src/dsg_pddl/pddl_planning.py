import logging
import os
import subprocess
import tempfile
import uuid
from datetime import datetime

from dsg_pddl.pddl_grounding import GroundedPddlProblem
from dsg_pddl.pddl_utils import lisp_string_to_ast

logger = logging.getLogger(__name__)


def solve_pddl(problem: GroundedPddlProblem):
    """Use fast-downward to solve the given pddl problem"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        now = datetime.now()
        key = str(uuid.uuid4())[:8]
        formatted_str = now.strftime("%Y-%m-%d_%H_%M_%S")

        problem_fn = os.path.join(tmpdirname, "problem.pddl")
        debug_problem_fn = os.path.expanduser(
            f"~/omniplanner_problem_{formatted_str}_{key}.pddl"
        )
        domain_fn = os.path.join(tmpdirname, "domain.pddl")
        debug_domain_fn = os.path.expanduser(
            f"~/omniplanner_domain_{formatted_str}_{key}.pddl"
        )
        plan_fn = os.path.join(tmpdirname, "plan.txt")
        debug_plan_fn = os.path.expanduser(
            f"~/omniplanner_plan_{formatted_str}_{key}.pddl"
        )

        with open(problem_fn, "w") as fo:
            fo.write(problem.problem_str)

        with open(debug_problem_fn, "w") as fo:
            fo.write(problem.problem_str)

        with open(domain_fn, "w") as fo:
            fo.write(problem.domain.to_string())

        with open(debug_domain_fn, "w") as fo:
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
            with open(debug_plan_fn, "w") as fo:
                fo.write(lines)
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
