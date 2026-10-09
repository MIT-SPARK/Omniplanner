"""Language goals in the ``goto_points`` domain take node symbols, not prose.

Previously a natural-language command sent with domain_type "goto_points"
crashed deep in grounding with ``'NoneType' object has no attribute 'fixed'``.
"""

import pytest
import spark_dsg

from omniplanner.language_planner import (
    LanguageDomain,
    LanguageGoal,
    _assign_to_addressed_robot,
    ground_problem,
)
from omniplanner.utils import str_to_ns_value


def test_str_to_ns_value_rejects_non_symbols():
    assert str_to_ns_value("O(12)") == spark_dsg.NodeSymbol("O", 12).value
    with pytest.raises(ValueError, match="not a scene-graph node symbol"):
        str_to_ns_value("go")


@pytest.mark.parametrize("command", ["go look at the closest sign", "O(2) and", ""])
def test_goto_points_language_goal_needs_symbols(command):
    domain = LanguageDomain(domain_type="goto_points")
    goal = LanguageGoal(robot_id="euclid", command=command)
    with pytest.raises(ValueError, match="expects node symbols"):
        ground_problem(domain, None, {}, goal)


def test_llm_goal_reassigned_to_addressed_robot():
    goal = LanguageGoal(robot_id="euclid", command="inspect the sign")
    out = _assign_to_addressed_robot({"apollo": "(visited-object o1)"}, goal)
    assert out == {"euclid": "(visited-object o1)"}
    # an explicitly named robot, or a multi-robot split, is respected
    named = LanguageGoal(robot_id="euclid", command="apollo, inspect the sign")
    assert _assign_to_addressed_robot({"apollo": "g"}, named) == {"apollo": "g"}
    split = {"apollo": "a", "hamilton": "b"}
    assert _assign_to_addressed_robot(split, goal) == split
