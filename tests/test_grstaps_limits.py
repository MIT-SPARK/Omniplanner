"""GRSTAPS-X says plainly when a goal is too big or the solver ran out of time."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from test_grstaps_regions import G, _domain  # noqa: F401

from omniplanner import grstaps_planner as gp


def test_too_many_tasks_is_refused_before_solving(G, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("OMNIPLANNER_GRSTAPS_MAX_TASKS", "3")
    goal = gp.GrstapsGoal(goal_points=["p1", "p2", "p3", "p5"], robot_id="spot1")
    with pytest.raises(
        ValueError,
        match=r"makes 4 GRSTAPS-X tasks.*limit 3.*OMNIPLANNER_GRSTAPS_MAX_TASKS",
    ):
        gp.ground_problem(_domain(tmp_path), G, {"spot1": np.zeros(3)}, goal)


def _problem(tmp_path, n_tasks, aprs):
    scen = tmp_path / "scen"
    (scen / "logs").mkdir(parents=True)
    (scen / "itags_input.json").write_text(json.dumps({"tasks": [{}] * n_tasks}))
    (scen / "logs" / "run.txt").write_text(
        "".join(f"[x] [debug] APR: {a}\n" for a in aprs)
    )
    return gp.GroundedGrstapsProblem(
        scenario_dir=str(scen), domain=NS(solver_timeout_s=60.0)
    )


def test_progress_comes_from_the_last_apr(tmp_path):
    assert gp._itags_progress(_problem(tmp_path, 172, [0.9942, 0.9651])) == (6, 172)


@pytest.mark.parametrize(
    "out",
    [
        "[error] ITAGS found no solution: the search timed out before every task was allocated\n",
        "terminate called after throwing an instance of 'std::bad_optional_access'\n",  # an older itags
    ],
)
def test_a_timeout_is_said_in_words(tmp_path, monkeypatch, out):
    problem = _problem(tmp_path, 172, [0.9651])
    problem.domain.entry = "itags"
    monkeypatch.setattr(
        gp,
        "_run_itags",
        lambda p: (
            NS(stdout=out, stderr="", returncode=2),
            str(tmp_path / "none.json"),
        ),
    )
    with pytest.raises(
        Exception,
        match=r"ran out of time.*after 60 s with 6 of 172 task\(s\) allocated.*split the goal",
    ):
        gp.make_plan(problem, None)
