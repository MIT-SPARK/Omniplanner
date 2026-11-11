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


@pytest.mark.parametrize(
    "command, named",
    [
        ("apollo, inspect the sign", True),
        ("Have APOLLO inspect the sign", True),
        ("inspect the sign (apollo).", True),
        ("inspect the apollo-11 poster", True),  # hyphen is a word boundary
        ("inspect the apollonian sign", False),
        ("inspect the sign near neoapollo", False),
    ],
)
def test_named_robot_is_matched_as_whole_word(command, named):
    goal = LanguageGoal(robot_id="euclid", command=command)
    out = _assign_to_addressed_robot({"apollo": "g"}, goal)
    assert out == ({"apollo": "g"} if named else {"euclid": "g"})


class _FailingLLM:
    def request_plan_specification(self, instruction, dsg):
        raise AssertionError("the LLM must not be called")


@pytest.mark.parametrize("domain_type", ["Pddl", "goto_points"])
def test_unknown_addressed_robot_fails_before_planning(domain_type):
    domain = LanguageDomain(domain_type=domain_type, llm_interface=_FailingLLM())
    goal = LanguageGoal(robot_id="eucld", command="O(1)")
    with pytest.raises(ValueError, match="unknown robot 'eucld'.*euclid, hamilton"):
        ground_problem(domain, None, {"euclid": None, "hamilton": None}, goal)


def test_goto_points_needs_a_robot():
    domain = LanguageDomain(domain_type="goto_points")
    goal = LanguageGoal(robot_id="", command="O(1)")
    with pytest.raises(ValueError, match="needs a robot_id"):
        ground_problem(domain, None, {"euclid": None}, goal)


def test_team_goal_without_robot_id_skips_the_hint():
    """An empty robot_id (e.g. the panel's "Any robot") is not checked or
    hinted; the LLM's split is used as-is."""
    seen = []

    class _LLM:
        response_history = []

        def request_plan_specification(self, instruction, dsg):
            seen.append(instruction)
            return "{}"

    domain = LanguageDomain(domain_type="Pddl", llm_interface=_LLM())
    goal = LanguageGoal(robot_id="", command="inspect every sign")
    assert ground_problem(domain, None, {"euclid": None}, goal) == []
    assert seen == ["inspect every sign"]
