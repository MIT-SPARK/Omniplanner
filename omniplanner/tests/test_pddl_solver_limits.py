"""Solver limits are opt-in; an exhausted anytime search retries only safely."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from dsg_pddl import pddl_planning


@pytest.fixture
def solver(monkeypatch, tmp_path):
    for name in (
        "ADT4_FD_ALIAS",
        "ADT4_FD_SEARCH",
        "ADT4_FD_OVERALL_TIME_LIMIT",
        "OMNIPLANNER_FD_TRANSLATE_TIME_LIMIT",
        "OMNIPLANNER_FD_SEARCH_TIME_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMNIPLANNER_DUMP_DIR", str(tmp_path / "dumps"))
    monkeypatch.setenv("ADT4_OUTPUT_DIR", str(tmp_path))
    problem = SimpleNamespace(
        problem_str="(problem)",
        domain=SimpleNamespace(to_string=lambda: "(domain)"),
        anytime_search_s=0,
        failure_hint="",
    )
    calls = []

    def run(returncode=0, *, write_plan=True, translated=True):
        def fake_run(command, **kwargs):
            calls.append(command)
            if translated:
                Path(kwargs["cwd"], "output.sas").write_text("translated task")
            if write_plan:
                plan_fn = command[command.index("--plan-file") + 1]
                Path(plan_fn).write_text("(goto-poi t1 t2)\n; cost = 1\n")
            return subprocess.CompletedProcess(
                command,
                returncode,
                stdout="translate exit code: 0\n" if translated else "",
                stderr="",
            )

        monkeypatch.setattr(pddl_planning.subprocess, "run", fake_run)

    return problem, calls, run


def test_default_solve_has_no_phase_time_limits(solver):
    problem, calls, run = solver
    run()
    assert pddl_planning.solve_pddl(problem) == [("goto-poi", "t1", "t2")]
    assert "--translate-time-limit" not in calls[0]
    assert "--search-time-limit" not in calls[0]
    assert "--overall-time-limit" not in calls[0]


@pytest.mark.parametrize("value", ["", "0", "120"])
def test_explicit_phase_limits(solver, monkeypatch, value):
    problem, calls, run = solver
    monkeypatch.setenv("OMNIPLANNER_FD_TRANSLATE_TIME_LIMIT", value)
    monkeypatch.setenv("OMNIPLANNER_FD_SEARCH_TIME_LIMIT", value)
    run()
    pddl_planning.solve_pddl(problem)
    for flag in ("--translate-time-limit", "--search-time-limit"):
        if value == "120":
            assert calls[0][calls[0].index(flag) + 1] == value
        else:
            assert flag not in calls[0]


def test_overall_limit_is_preserved(solver, monkeypatch):
    problem, calls, run = solver
    monkeypatch.setenv("ADT4_FD_OVERALL_TIME_LIMIT", "180")
    run()
    pddl_planning.solve_pddl(problem)
    assert calls[0][calls[0].index("--overall-time-limit") + 1] == "180"


@pytest.mark.parametrize("returncode", [20, 22, 24])
def test_out_of_memory_does_not_retry(solver, returncode):
    problem, calls, run = solver
    problem.anytime_search_s = 1
    run(returncode, write_plan=False)
    with pytest.raises(Exception, match="ran out of memory"):
        pddl_planning.solve_pddl(problem)
    assert len(calls) == 1


@pytest.mark.parametrize("greedy_limit", ["", "120"])
def test_anytime_timeout_reuses_translation_for_greedy(
    solver, monkeypatch, greedy_limit
):
    problem, calls, run = solver
    problem.anytime_search_s = 1
    run(23, write_plan=False)
    monkeypatch.setenv("OMNIPLANNER_FD_SEARCH_TIME_LIMIT", greedy_limit)
    monkeypatch.setenv("OMNIPLANNER_FD_TRANSLATE_TIME_LIMIT", "120")
    first_run = pddl_planning.subprocess.run

    def timeout_then_success(command, **kwargs):
        result = first_run(command, **kwargs)
        if len(calls) == 2:
            plan_fn = command[command.index("--plan-file") + 1]
            Path(plan_fn).write_text("(goto-poi t1 t2)\n; cost = 1\n")
            result.returncode = 0
        return result

    monkeypatch.setattr(pddl_planning.subprocess, "run", timeout_then_success)
    pddl_planning.solve_pddl(problem)
    assert len(calls) == 2
    assert calls[0][calls[0].index("--search-time-limit") + 1] == "1"
    assert any(arg.endswith("output.sas") for arg in calls[1])
    if greedy_limit:
        assert calls[1][calls[1].index("--search-time-limit") + 1] == greedy_limit
    else:
        assert "--search-time-limit" not in calls[1]
    assert "--translate-time-limit" not in calls[1]


@pytest.mark.parametrize("returncode", [1, 2, 3])
def test_resource_limit_keeps_an_existing_plan(solver, returncode):
    problem, calls, run = solver
    problem.anytime_search_s = 1
    run(returncode)
    assert pddl_planning.solve_pddl(problem) == [("goto-poi", "t1", "t2")]
    assert len(calls) == 1
