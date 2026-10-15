import logging
import os
import subprocess
import tempfile
import uuid
from datetime import datetime

from dsg_pddl.pddl_grounding import GroundedPddlProblem
from dsg_pddl.pddl_utils import lisp_string_to_ast

logger = logging.getLogger(__name__)


DEFAULT_FD_SEARCH = (
    "let(hff, ff(), let(hcea, cea(), lazy_greedy([hff, hcea], preferred=[hff, hcea])))"
)


# fast-downward exit codes (driver/returncodes.py).
FD_UNSOLVABLE = (10, 11, 12)  # proved there is no plan, or searched it all
FD_OUT_OF_TIME = (21, 23)
FD_OUT_OF_MEMORY = (20, 22, 24)
FD_BAD_INPUT = (30, 31)  # the translator rejected the problem


def failure_reason(returncode: int, problem: GroundedPddlProblem) -> str:
    """Says why fast-downward found no plan, in words the operator can act on."""
    hint = getattr(problem, "failure_hint", "")
    if hint:
        return f"Cannot satisfy the goal and constraints: {hint}"
    if returncode in FD_UNSOLVABLE:
        return "Cannot satisfy the goal and constraints: no plan achieves the goal"
    if returncode in FD_OUT_OF_TIME:
        return "The planner ran out of time before finding a plan"
    if returncode in FD_OUT_OF_MEMORY:
        return "The planner ran out of memory before finding a plan"
    if returncode in FD_BAD_INPUT:
        return (
            "The goal could not be read: check it names objects and places "
            "in the scene graph"
        )
    return f"The planner failed (fast-downward exit code {returncode})"


def solve_pddl(problem: GroundedPddlProblem):
    """Use fast-downward to solve the given pddl problem.

    The Fast Downward invocation can be overridden via environment variables:
        ADT4_FD_ALIAS              - use a Fast Downward alias (e.g. "seq-sat-lama-2011").
                                     Takes precedence over ADT4_FD_SEARCH when set.
        ADT4_FD_SEARCH             - raw value passed after `--search`.
        ADT4_FD_OVERALL_TIME_LIMIT - value passed via `--overall-time-limit`.
    When unset, the historical lazy-greedy(ff + cea) search is used.
    """
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

        fd_alias = os.getenv("ADT4_FD_ALIAS", "").strip()
        fd_search = os.getenv("ADT4_FD_SEARCH", "").strip()
        fd_time_limit = os.getenv("ADT4_FD_OVERALL_TIME_LIMIT", "").strip()

        command = ["fast-downward"]
        command += ["--plan-file", plan_fn]
        if fd_time_limit:
            command += ["--overall-time-limit", fd_time_limit]
        if fd_alias:
            command += ["--alias", fd_alias]
        command += [domain_fn]
        command += [problem_fn]
        if not fd_alias:
            command += ["--search", fd_search or DEFAULT_FD_SEARCH]

        logger.warning(f"Calling: {command}")
        return_code = subprocess.run(command)
        logger.warning(f"Return code: {return_code}")

        if os.path.exists(plan_fn):
            with open(plan_fn, "r") as fo:
                lines = fo.readlines()
            with open(debug_plan_fn, "w") as fo:
                fo.writelines(lines)
        else:
            output_dir = os.getenv("ADT4_OUTPUT_DIR", "")
            debug_fn = os.path.join(output_dir, "pddl_problem_debugging.pddl")
            logger.warning(
                f"Planning failed. Please see {debug_fn} for the failed problem file."
            )
            with open(debug_fn, "w") as fo:
                fo.write(problem.problem_str)
            raise Exception(failure_reason(return_code.returncode, problem))

    plan = [lisp_string_to_ast(line) for line in lines[:-1]]
    return plan
