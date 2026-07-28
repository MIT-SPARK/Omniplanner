"""Tests for typed grounding + solver errors.

Two seams are covered:

  * ``generate_goal_relevant_pddl`` must raise ``MissingSymbolError`` (listing
    *every* unresolved name) instead of silently dropping goal symbols that are
    absent from the scene graph -- the silent skip left the symbol in the goal
    string, which Fast Downward then rejected as an undeclared object (rc 31)
    and surfaced as a bare ``Exception``.
  * ``solve_pddl`` must triage ``proc.returncode`` into a typed
    ``PddlSolverError`` subclass so a caller can tell "unsolvable" from
    "malformed" from "timed out".

Resolvable goals must still ground byte-identically (snapshot test below).

Runnable both under pytest and directly:
    PYTHONPATH=omniplanner/src python -m pytest omniplanner/tests/test_grounding_errors.py
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from utils import build_test_dsg  # noqa: E402

from dsg_pddl.dsg_pddl_grounding import generate_goal_relevant_pddl  # noqa: E402
from dsg_pddl.grounding_errors import (  # noqa: E402
    GroundingError,
    MissingSymbol,
    MissingSymbolError,
    PddlMalformedError,
    PddlSolverError,
    PddlTimeoutError,
    PddlUnsolvableError,
    error_for_fd_returncode,
)
from dsg_pddl.pddl_grounding import GroundedPddlProblem  # noqa: E402
from dsg_pddl.pddl_planning import solve_pddl  # noqa: E402

PROBLEM_NAME = "object-rearrangement-domain"


def _ground(goal_str, G=None):
    G = build_test_dsg() if G is None else G
    return generate_goal_relevant_pddl(
        G, goal_str, np.array([0.0, 0.0]), PROBLEM_NAME, PROBLEM_NAME
    )


# --------------------------------------------------------------------------
# error type shape
# --------------------------------------------------------------------------


def test_missing_symbol_error_shape():
    missing = [MissingSymbol("o42", "object"), MissingSymbol("r9", "region")]
    err = MissingSymbolError(missing, "(and (object-in-region o42 r9))")
    assert isinstance(err, GroundingError)
    assert isinstance(err, Exception)
    assert err.missing is missing
    assert err.goal == "(and (object-in-region o42 r9))"
    assert "['o42', 'r9']" in str(err)


def test_missing_symbol_defaults():
    m = MissingSymbol("o1")
    assert m.name == "o1"
    assert m.kind_hint == ""


def test_solver_error_hierarchy():
    for cls in (PddlUnsolvableError, PddlTimeoutError, PddlMalformedError):
        assert issubclass(cls, PddlSolverError)
        assert issubclass(cls, Exception)
    err = PddlSolverError("boom", 99)
    assert err.returncode == 99
    assert str(err) == "boom"
    assert PddlSolverError("boom").returncode is None


# --------------------------------------------------------------------------
# seam 1: generate_goal_relevant_pddl
# --------------------------------------------------------------------------


def test_missing_object_raises_instead_of_silent_skip():
    goal = "(and (visited-object o42))"
    with pytest.raises(MissingSymbolError) as exc_info:
        _ground(goal)
    err = exc_info.value
    assert [m.name for m in err.missing] == ["o42"]
    assert err.missing[0].kind_hint == "object"
    assert err.goal == goal


def test_all_missing_symbols_are_reported():
    # both o42 and r9 are absent; the error must list BOTH (not just the first)
    with pytest.raises(MissingSymbolError) as exc_info:
        _ground("(and (object-in-region o42 r9))")
    names = sorted(m.name for m in exc_info.value.missing)
    assert names == ["o42", "r9"]
    hints = {m.name: m.kind_hint for m in exc_info.value.missing}
    assert hints == {"o42": "object", "r9": "region"}


def test_partially_missing_goal_still_raises():
    # o0 exists in the fixture, p99 does not -> only p99 reported
    with pytest.raises(MissingSymbolError) as exc_info:
        _ground("(and (object-in-place o0 p99))")
    assert [(m.name, m.kind_hint) for m in exc_info.value.missing] == [
        ("p99", "place")
    ]


def test_kind_hint_heuristic_unknown_leading_char():
    with pytest.raises(MissingSymbolError) as exc_info:
        _ground("(and (visited-object zzz1))")
    assert [(m.name, m.kind_hint) for m in exc_info.value.missing] == [("zzz1", "")]


# The exact problem string emitted for a resolvable goal, captured verbatim
# from the pre-change implementation (whitespace included, so it lives in a
# data file rather than a source literal). Grounding a resolvable goal must
# stay byte-identical: the typed error only replaces the silent-skip branch.
SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__), "data", "goal_relevant_resolvable_problem.pddl"
)


def test_resolvable_goal_grounds_byte_identically():
    with open(SNAPSHOT_PATH) as fo:
        expected = fo.read()
    problem_str, symbols = _ground("(and (visited-object o0))")
    assert problem_str == expected
    assert sorted(s.symbol for s in symbols) == ["o0", "p0", "pstart"]


# --------------------------------------------------------------------------
# seam 2: solve_pddl returncode triage
# --------------------------------------------------------------------------

TRIVIAL_DOMAIN = """(define (domain trivial)
 (:requirements :strips)
 (:predicates (p) (q))
 (:action noop
  :parameters ()
  :precondition (p)
  :effect (p))
)"""

UNSOLVABLE_PROBLEM = """(define (problem trivial-unsolvable)
 (:domain trivial)
 (:init (p))
 (:goal (q))
)"""

# The exact shape of the original bug: the goal names an object that was never
# declared in (:objects) -- FD's translator rejects the input (rc 31).
MALFORMED_PROBLEM = """(define (problem trivial-malformed)
 (:domain trivial)
 (:objects )
 (:init (p))
 (:goal (holds o42))
)"""


class _StubDomain:
    """Minimal stand-in for PddlDomain: solve_pddl only needs to_string()."""

    def __init__(self, domain_str):
        self._domain_str = domain_str

    def to_string(self):
        return self._domain_str


def _problem(problem_str, domain_str=TRIVIAL_DOMAIN):
    return GroundedPddlProblem(
        domain=_StubDomain(domain_str), problem_str=problem_str, symbols={}
    )


@pytest.fixture
def isolated_dumps(tmp_path, monkeypatch):
    """Point both debug-dump destinations at a tmp dir."""
    monkeypatch.setenv("OMNIPLANNER_DUMP_DIR", str(tmp_path / "dumps"))
    monkeypatch.setenv("ADT4_OUTPUT_DIR", str(tmp_path))
    return tmp_path


needs_fd = pytest.mark.skipif(
    shutil.which("fast-downward") is None, reason="fast-downward not installed"
)


@needs_fd
def test_solve_pddl_unsolvable_raises_typed_error(isolated_dumps):
    with pytest.raises(PddlUnsolvableError) as exc_info:
        solve_pddl(_problem(UNSOLVABLE_PROBLEM))
    assert exc_info.value.returncode in (10, 11, 12)
    # the debug problem file is still written before raising
    assert (isolated_dumps / "pddl_problem_debugging.pddl").exists()


@needs_fd
def test_solve_pddl_malformed_raises_typed_error(isolated_dumps):
    with pytest.raises(PddlMalformedError) as exc_info:
        solve_pddl(_problem(MALFORMED_PROBLEM))
    assert exc_info.value.returncode == 31


@pytest.mark.parametrize(
    "returncode,expected",
    [
        (10, PddlUnsolvableError),
        (11, PddlUnsolvableError),
        (12, PddlUnsolvableError),
        (21, PddlTimeoutError),
        (23, PddlTimeoutError),
        (24, PddlTimeoutError),
        (31, PddlMalformedError),
        (30, PddlSolverError),
        (33, PddlSolverError),
        (None, PddlSolverError),
    ],
)
def test_error_for_fd_returncode_map(returncode, expected):
    err = error_for_fd_returncode(returncode, "msg")
    assert type(err) is expected
    assert err.returncode == returncode
    assert "msg" in str(err)


@needs_fd
def test_unwritable_debug_path_does_not_mask_typed_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIPLANNER_DUMP_DIR", str(tmp_path / "dumps"))
    monkeypatch.setenv("ADT4_OUTPUT_DIR", str(tmp_path / "does" / "not" / "exist"))
    with pytest.raises(PddlUnsolvableError):
        solve_pddl(_problem(UNSOLVABLE_PROBLEM))


@pytest.mark.parametrize(
    "returncode,expected",
    [
        (10, PddlUnsolvableError),
        (11, PddlUnsolvableError),
        (12, PddlUnsolvableError),
        (21, PddlTimeoutError),
        (23, PddlTimeoutError),
        (24, PddlTimeoutError),
        (31, PddlMalformedError),
        (32, PddlSolverError),
    ],
)
def test_solve_pddl_maps_returncode(returncode, expected, isolated_dumps, monkeypatch):
    """No plan file + a given FD returncode -> the matching typed error."""

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, returncode, stdout="fake stdout", stderr="fake stderr"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(expected) as exc_info:
        solve_pddl(_problem(UNSOLVABLE_PROBLEM))
    assert type(exc_info.value) is expected
    assert exc_info.value.returncode == returncode
    debug_fn = isolated_dumps / "pddl_problem_debugging.pddl"
    assert debug_fn.exists()
    assert debug_fn.read_text() == UNSOLVABLE_PROBLEM
