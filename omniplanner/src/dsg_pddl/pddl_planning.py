import glob
import logging
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from datetime import datetime

from dsg_pddl.grounding_errors import error_for_fd_returncode
from dsg_pddl.pddl_grounding import GroundedPddlProblem
from dsg_pddl.pddl_utils import lisp_string_to_ast

logger = logging.getLogger(__name__)

DEFAULT_FD_SEARCH = (
    "let(hff, ff(), let(hcea, cea(), lazy_greedy([hff, hcea], preferred=[hff, hcea])))"
)

# Anytime search for problems that ask for it (GroundedPddlProblem.anytime_search_s,
# the small goal-relevant ones): the greedy plan above first, then weighted A*
# with falling weights, each pass bounded by the best cost so far, until the
# search space is exhausted or the budget runs out. The greedy plan alone ignores
# cost: on 10-30 targets it drives 20-45% further than what this finds in 1-2 s.
# (A cost-optimal A* would need an admissible heuristic that supports the
# domains' derived predicates; LM-cut does not.)
ANYTIME_FD_SEARCH = (
    "let(hff, ff(), let(hcea, cea(), iterated(["
    "lazy_greedy([hff, hcea], preferred=[hff, hcea]), "
    "lazy_wastar([hff, hcea], preferred=[hff, hcea], w=5), "
    "lazy_wastar([hff, hcea], preferred=[hff, hcea], w=3), "
    "lazy_wastar([hff, hcea], preferred=[hff, hcea], w=2), "
    "lazy_wastar([hff, hcea], preferred=[hff, hcea], w=1)"
    "], repeat_last=true, continue_on_fail=true)))"
)


# Lines of Fast Downward output kept on a failure (PddlSolverError.fd_log).
FD_LOG_TAIL = 40

_FD_SIGNALS = (
    (re.compile(r"No relaxed solution"), "the goal is unreachable from the start (no relaxed solution)"),
    (re.compile(r"Completely explored state space"), "explored every state: no plan"),
    (re.compile(r"Search stopped without finding a solution"), "search stopped without a plan"),
    (re.compile(r"Time limit has been reached"), "search time limit reached"),
    (re.compile(r"Memory limit has been reached"), "memory limit reached"),
    (re.compile(r"Solution found"), "solution found"),
)


def fd_digest(output: str) -> str:
    """One line on what Fast Downward did: where it stopped and how big the task was.

    e.g. "translate killed (exit -9: time or memory limit) after 61.4 s; 21651110
    queue pushes" or "search time limit reached; 2112 operators, 410 facts;
    6091 states expanded; peak 30 MB".
    """
    if not output:
        return ""
    parts = []
    m = re.search(r"translate exit code: (-?\d+)", output)
    if m and m.group(1) != "0":
        code = int(m.group(1))
        parts.append(
            f"translate killed (exit {code}: time or memory limit)" if code < 0 else f"translate exit {code}"
        )
    for pattern, text in _FD_SIGNALS:
        if pattern.search(output):
            parts.append(text)
    if _FD_SIGNALS[0][1] in parts:  # proved unreachable: the search lines say the same
        parts = [p for p in parts if p not in (_FD_SIGNALS[1][1], _FD_SIGNALS[2][1])]
    for line in output.splitlines():  # the translator's own errors
        if re.match(r"\s*(Error|error:|Undefined|.*Exception)", line):
            parts.append(line.strip()[:160])
            break
    size = []
    for key, label in (("operators", "operators"), ("facts", "facts"), ("axioms", "axioms")):
        m = re.search(rf"Translator {key}: (\d+)", output)
        if m:
            size.append(f"{m.group(1)} {label}")
    if size:
        parts.append(", ".join(size))
    pushes = re.findall(r"(\d+) total queue pushes", output)
    if pushes and not size:  # killed while instantiating: how far it got
        parts.append(f"{pushes[-1]} queue pushes")
    expanded = re.findall(r"(\d+) expanded", output)
    if expanded:
        parts.append(f"{expanded[-1]} states expanded")
    m = re.findall(r"Peak memory: (\d+) KB", output)
    if m:
        parts.append(f"peak {int(m[-1]) // 1024} MB")
    m = re.findall(r"Planner time: ([\d.]+)s", output)
    if m:
        parts.append(f"{float(m[-1]):.1f} s")
    return "; ".join(parts)


def _log_tail(output: str, n: int = FD_LOG_TAIL) -> list[str]:
    """The last ``n`` lines of Fast Downward's output, without its driver's INFO chatter."""
    lines = [line for line in output.splitlines() if line.strip() and not line.startswith("INFO ")]
    return lines[-n:]


def _best_plan_file(plan_fn):
    """The last (cheapest) plan an anytime search wrote: plan, or plan.1, plan.2, ..."""
    numbered = sorted(glob.glob(plan_fn + ".*"), key=lambda f: int(f.rsplit(".", 1)[1]))
    if numbered:
        return numbered[-1]
    return plan_fn if os.path.exists(plan_fn) else None


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


def resolve_dump_dir():
    """Directory where the debug domain/problem/plan PDDL dumps are written.

    Defaults to ~/adt4_output/omniplanner/; override with the
    OMNIPLANNER_DUMP_DIR env var. Created if it doesn't already exist.
    """
    dump_dir = os.environ.get("OMNIPLANNER_DUMP_DIR")
    dump_dir = (
        os.path.expanduser(dump_dir)
        if dump_dir
        else os.path.expanduser("~/adt4_output/omniplanner")
    )
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir


def solve_pddl(problem: GroundedPddlProblem):
    """Use fast-downward to solve the given pddl problem.

    The Fast Downward invocation can be overridden via environment variables:
        ADT4_FD_ALIAS              - use a Fast Downward alias (e.g. "seq-sat-lama-2011").
                                     Takes precedence over ADT4_FD_SEARCH when set.
        ADT4_FD_SEARCH             - raw value passed after `--search`.
        ADT4_FD_OVERALL_TIME_LIMIT - value passed via `--overall-time-limit`.
        OMNIPLANNER_FD_TRANSLATE_TIME_LIMIT - optional translation limit.
        OMNIPLANNER_FD_SEARCH_TIME_LIMIT - optional greedy-search limit.
    Translation and greedy search are unlimited unless configured (0 or an
    empty value disables either phase limit). Goal-relevant problems may use
    a short anytime search first, then retry greedy if no plan was found.
    When unset, the historical lazy-greedy(ff + cea) search is used.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        now = datetime.now()
        key = str(uuid.uuid4())[:8]
        formatted_str = now.strftime("%Y-%m-%d_%H_%M_%S")

        dump_dir = resolve_dump_dir()

        problem_fn = os.path.join(tmpdirname, "problem.pddl")
        debug_problem_fn = os.path.join(
            dump_dir, f"omniplanner_problem_{formatted_str}_{key}.pddl"
        )
        domain_fn = os.path.join(tmpdirname, "domain.pddl")
        debug_domain_fn = os.path.join(
            dump_dir, f"omniplanner_domain_{formatted_str}_{key}.pddl"
        )
        plan_fn = os.path.join(tmpdirname, "plan.txt")
        debug_plan_fn = os.path.join(
            dump_dir, f"omniplanner_plan_{formatted_str}_{key}.pddl"
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
        translate_time_limit = os.getenv(
            "OMNIPLANNER_FD_TRANSLATE_TIME_LIMIT", ""
        ).strip()
        search_time_limit = os.getenv("OMNIPLANNER_FD_SEARCH_TIME_LIMIT", "").strip()
        # an explicit alias / search from the environment wins over anytime search
        anytime_s = getattr(problem, "anytime_search_s", 0.0) or 0.0
        anytime = anytime_s > 0 and not fd_alias and not fd_search

        def fd_command(search, search_time_limit, inputs=None):
            """``inputs``: a translated task ([output.sas]) runs the search alone."""
            command = ["fast-downward"]
            if inputs is None and translate_time_limit not in (None, "", "0"):
                command += ["--translate-time-limit", str(translate_time_limit)]
            if search_time_limit not in (None, "", "0"):
                command += ["--search-time-limit", str(search_time_limit)]
            command += ["--plan-file", plan_fn]
            if fd_time_limit:
                command += ["--overall-time-limit", fd_time_limit]
            if fd_alias:
                command += ["--alias", fd_alias]
            command += inputs or [domain_fn, problem_fn]
            if not fd_alias:
                command += ["--search", search]
            return command

        if anytime:
            # fast-downward takes whole seconds ("1.0" is a usage error)
            command = fd_command(ANYTIME_FD_SEARCH, max(1, math.ceil(anytime_s)))
            # FD deletes output.sas after a run; keep it for the greedy retry below
            command.insert(1, "--keep-sas-file")
        else:
            command = fd_command(fd_search or DEFAULT_FD_SEARCH, search_time_limit)

        logger.info(f"Calling: {command}")
        fd_start = time.perf_counter()
        # Capture FD output instead of letting it stream to the terminal/log;
        # surface it only on failure or at DEBUG.
        #
        # cwd=tmpdirname: Fast Downward's translate step writes its intermediate
        # `output.sas` to the PROCESS CWD (a relative path -- see the FD driver's
        # `--sas-file output.sas`), and the search step reads it back from there.
        # Under `ros2 launch` the node's CWD is typically `/` (or another
        # non-writable dir), so translate dies with
        # `FileNotFoundError: 'output.sas'` (exit 30) and the whole solve -- and
        # the omniplanner_node process -- crashes on an otherwise trivially
        # solvable problem. Pin FD's CWD to the per-solve TemporaryDirectory we
        # already own (domain/problem/plan live there too) so output.sas lands
        # somewhere writable and isolated. Verified: identical domain+problem
        # solves in ~0.13 s from a writable CWD but reproduces the exact
        # FileNotFoundError from a non-writable one.
        proc = subprocess.run(command, capture_output=True, text=True, cwd=tmpdirname)
        fd_elapsed = time.perf_counter() - fd_start
        logger.info(
            f"fast-downward finished in {fd_elapsed:.3f}s (return code {proc.returncode})"
        )
        logger.debug("fast-downward stdout:\n%s", proc.stdout)
        if proc.stderr:
            logger.debug("fast-downward stderr:\n%s", proc.stderr)

        # An anytime search that ran out of its budget before a first plan (or
        # failed for a reason other than unsolvable/bad input/out of memory)
        # gets the plain greedy search. A memory failure is terminal: another
        # translation/search can exhaust the same machine again.
        if (
            anytime
            and _best_plan_file(plan_fn) is None
            and proc.returncode not in FD_UNSOLVABLE + FD_BAD_INPUT + FD_OUT_OF_MEMORY
        ):
            # the first run's translation (output.sas in the cwd) is reused: on
            # big problems translating is most of the time (Penn, 98 goals: 12 s)
            sas_fn = os.path.join(tmpdirname, "output.sas")
            translated = "translate exit code: 0" in (proc.stdout or "")
            inputs = [sas_fn] if translated and os.path.isfile(sas_fn) else None
            logger.warning(
                "Anytime search found no plan within %.1f s (return code %d); "
                "retrying with the greedy search%s",
                anytime_s,
                proc.returncode,
                " on the translated task" if inputs else "",
            )
            # the translator's part only: the failed search's lines would read as this run's
            translate_out = (proc.stdout or "").split("translate exit code: 0", 1)[0]
            proc = subprocess.run(
                fd_command(DEFAULT_FD_SEARCH, search_time_limit, inputs),
                capture_output=True,
                text=True,
                cwd=tmpdirname,
            )
            if inputs:  # the translator's stats (operators, facts) are in the first run
                proc.stdout = translate_out + (proc.stdout or "")

        best_plan_fn = _best_plan_file(plan_fn)
        if best_plan_fn is not None:
            with open(best_plan_fn, "r") as fo:
                lines = fo.readlines()
            with open(debug_plan_fn, "w") as fo:
                fo.writelines(lines)
        else:
            output_dir = os.getenv("ADT4_OUTPUT_DIR", "")
            debug_fn = os.path.join(output_dir, "pddl_problem_debugging.pddl")
            logger.warning(
                f"Planning failed. Please see {debug_fn} for the failed problem file."
            )
            logger.warning("fast-downward stdout:\n%s", proc.stdout)
            logger.warning("fast-downward stderr:\n%s", proc.stderr)
            try:
                with open(debug_fn, "w") as fo:
                    fo.write(problem.problem_str)
            except OSError as e:
                # An unwritable dump path must not mask the real failure with
                # an OSError -- callers triage on the typed error below.
                logger.warning(f"Could not write {debug_fn}: {e}")
            # Triage the FD exit code so a caller can distinguish "no plan
            # exists" from "the problem file is malformed" from "we ran out of
            # time". Unknown codes stay a plain PddlSolverError. The message is
            # the operator-facing failure_reason (uses the grounder's
            # failure_hint when present); the debug path is logged above.
            # Fast Downward's own account goes along: a digest in the message
            # (it reaches the operator, e.g. on plan_failed) and the log tail
            # and problem file on the error, for consoles that show them.
            output = (proc.stdout or "") + (proc.stderr or "")
            digest = fd_digest(output)
            reason = failure_reason(proc.returncode, problem)
            err = error_for_fd_returncode(
                proc.returncode,
                f"{reason} [Fast Downward: {digest}]" if digest else reason,
            )
            err.fd_digest = digest
            err.fd_log = _log_tail(output)
            err.problem_file = debug_problem_fn
            raise err
        # what the search did, for callers that report it
        problem.fd_stats = {
            "seconds": round(time.perf_counter() - fd_start, 3),
            "plans": len(glob.glob(plan_fn + ".*")) or 1,
            "digest": fd_digest((proc.stdout or "") + (proc.stderr or "")),
        }

    plan = [lisp_string_to_ast(line) for line in lines[:-1]]
    return plan
