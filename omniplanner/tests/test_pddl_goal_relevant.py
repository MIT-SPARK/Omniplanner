"""Regression tests for goal-relevant PDDL grounding (scene_scope).

These verify that:
  * dynamic, goal-based pruning produces a much smaller PDDL problem than the
    legacy "full" scope,
  * the pruned problem still yields a valid plan that achieves the goal, and
  * the legacy "full" scope still works.

Runnable both under pytest and directly:
    PYTHONPATH=omniplanner/src python omniplanner/tests/test_pddl_goal_relevant.py
"""
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples")
)
from utils import build_test_dsg  # noqa: E402

from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal  # noqa: E402
from dsg_pddl.dsg_pddl_grounding import ground_problem  # noqa: E402
from dsg_pddl.dsg_pddl_planning import make_plan  # noqa: E402
from importlib.resources import as_file, files  # noqa: E402
import dsg_pddl.domains  # noqa: E402


def _load_domain(name, scene_scope):
    with as_file(files(dsg_pddl.domains).joinpath(name)) as p:
        domain = PddlDomain(open(p).read())
    domain.scene_scope = scene_scope
    return domain


def _ground(domain, goal_str):
    G = build_test_dsg()
    goal = PddlGoal(robot_id="euclid", pddl_goal=goal_str)
    grounded = ground_problem(domain, G, {"euclid": np.array([0.0, 0.0])}, goal)
    return G, grounded.value


def _plan_action_names(G, gp):
    plan = make_plan(gp, G)
    return [a[0] for a in plan.symbolic_actions]


def test_goal_relevant_is_smaller_than_full():
    goal = "(and (visited-object o0))"
    _, gp_rel = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"), goal
    )
    _, gp_full = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "full"), goal
    )
    # The pruned problem references only a couple of symbols; full encodes them all.
    assert len(gp_rel.symbols) < len(gp_full.symbols)
    assert "o0" in gp_rel.symbols
    assert "pstart" in gp_rel.symbols


def test_goal_relevant_goto_plan_visits_object():
    G, gp = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"),
        "(and (visited-object o0))",
    )
    plan = make_plan(gp, G)
    gotos = [a for a in plan.symbolic_actions if a[0] == "goto-poi"]
    assert gotos, "expected at least one goto-poi action"
    # the object should be the final POI we navigate to
    assert gotos[-1][-1] == "o0"


def test_goal_relevant_pick_place_plan():
    G, gp = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant"),
        "(and (object-in-place o0 p1))",
    )
    actions = _plan_action_names(G, gp)
    assert "pick-object" in actions
    assert "place-object" in actions


def test_offline_targeted_robot_raises_clear_error():
    # A None pose is what a lazy/offline robot lookup yields; planning *for* such
    # a robot should fail with a clear message rather than a cryptic TypeError.
    domain = _load_domain("RegionObjectRearrangementDomain.pddl", "goal_relevant")
    G = build_test_dsg()
    goal = PddlGoal(robot_id="euclid", pddl_goal="(and (visited-object o0))")
    try:
        ground_problem(domain, G, {"euclid": None}, goal)
        assert False, "expected RuntimeError for offline targeted robot"
    except RuntimeError as e:
        assert "no transform" in str(e)


def test_full_scope_still_plans():
    G, gp = _ground(
        _load_domain("RegionObjectRearrangementDomain.pddl", "full"),
        "(and (visited-object o0))",
    )
    actions = _plan_action_names(G, gp)
    assert "goto-poi" in actions


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}: {e!r}")
    sys.exit(1 if failures else 0)
