"""A fast-downward failure says why, in words the operator can act on.

The end-to-end cases need the plan_repair_graph prior map (OMNIPLANNER_TEST_DSG)
and fast-downward; they skip without them.
"""

from types import SimpleNamespace as NS

import pytest
from dsg_pddl.pddl_planning import failure_reason
from test_world_state import _fd, _map, needs_map


def test_reasons_by_exit_code():
    plain = NS(failure_hint="")
    assert "Cannot satisfy the goal and constraints" in failure_reason(12, plain)
    assert "ran out of time" in failure_reason(23, plain)
    assert "could not be read" in failure_reason(31, plain)
    assert "exit code 99" in failure_reason(99, plain)
    hinted = NS(failure_hint="o5 cannot be reached")
    assert failure_reason(12, hinted).endswith("o5 cannot be reached")


@needs_map
def test_forbidden_goal_names_the_unreachable_object():
    G = _map()
    forbid = NS(predicate="forbidden-poi", symbols=["o4"])
    with pytest.raises(Exception) as err:
        _fd(G, "(visited-object o4)", (0.0, 0.0), constraints=[forbid])
    text = str(err.value)
    assert text.startswith("Cannot satisfy the goal and constraints: o4")
    assert "forbidden-poi" in text


@needs_map
def test_unknown_object_says_the_goal_could_not_be_read():
    with pytest.raises(Exception) as err:
        _fd(_map(), "(visited-object o999)", (0.0, 0.0))
    assert "could not be read" in str(err.value)


@needs_map
def test_contradictory_goal_cannot_be_satisfied():
    with pytest.raises(Exception) as err:
        _fd(_map(), "(and (visited-object o4) (not (visited-object o4)))", (0.0, 0.0))
    assert str(err.value).endswith("no plan achieves the goal")
