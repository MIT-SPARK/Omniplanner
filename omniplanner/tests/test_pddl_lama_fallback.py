"""A goal the greedy ff+cea search cannot steer through is planned by LAMA.

The problem is the console's, 2026-11-17: husky1 visits the 42 objects of a
courtyard (R6) while two Spots carry three of them, the cones o757-o759, to
t7318. Either half alone plans in milliseconds; together, greedy ff+cea found
no plan in 13 minutes. lama-first plans it in well under a second.

Needs fast-downward on PATH (as the other planning tests do).
"""

import gzip
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from dsg_pddl import pddl_planning

DATA = Path(__file__).parent / "data"

pytestmark = pytest.mark.skipif(
    shutil.which("fast-downward") is None, reason="fast-downward not installed"
)


@pytest.fixture
def r6_problem(monkeypatch, tmp_path):
    for name in (
        "ADT4_FD_ALIAS",
        "ADT4_FD_SEARCH",
        "ADT4_FD_OVERALL_TIME_LIMIT",
        "OMNIPLANNER_FD_TRANSLATE_TIME_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    # the greedy retry fails here whatever its limit; keep the test short
    monkeypatch.setenv("OMNIPLANNER_FD_SEARCH_TIME_LIMIT", "3")
    monkeypatch.setenv("OMNIPLANNER_DUMP_DIR", str(tmp_path / "dumps"))
    monkeypatch.setenv("ADT4_OUTPUT_DIR", str(tmp_path))
    problem_str = gzip.decompress(
        (DATA / "r6_visits_and_cone_moves_problem.pddl.gz").read_bytes()
    ).decode()
    domain_str = (DATA / "region_multirobot_fd_domain.pddl").read_text()
    return SimpleNamespace(
        problem_str=problem_str,
        domain=SimpleNamespace(to_string=lambda: domain_str),
        anytime_search_s=1,
        failure_hint="",
    )


def test_visits_and_carries_of_the_same_objects_get_a_plan(r6_problem):
    goal = r6_problem.problem_str[r6_problem.problem_str.index("(:goal") :]
    visits = set(re.findall(r"\(visited-object-by husky1 (\S+)\)", goal))
    carries = set(re.findall(r"\(object-in-place (\S+) t7318\)", goal))
    assert len(visits) == 42 and carries == {"o757", "o758", "o759"}

    start = time.perf_counter()
    plan = pddl_planning.solve_pddl(r6_problem)
    assert time.perf_counter() - start < 30

    husky_reached = {a[3] for a in plan if a[0] == "goto-poi" and a[1] == "husky1"}
    assert visits <= husky_reached
    placed = {a[2] for a in plan if a[0] == "place-object" and a[3] == "t7318"}
    assert placed == carries
