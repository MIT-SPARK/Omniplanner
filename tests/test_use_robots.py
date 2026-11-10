"""A use-robots constraint keeps a goal to the robots the user named."""

import numpy as np
import pytest
from dsg_pddl.dsg_pddl_grounding_multirobot import planning_robot_states
from dsg_pddl.pddl_grounding import ConstraintFact

STATES = {"hilbert": np.zeros(3), "euclid": np.ones(3), "kepler": None}


def test_without_it_every_robot_with_a_pose_may_act():
    assert planning_robot_states(STATES, []) is STATES


def test_it_keeps_only_the_named_robots():
    use = [ConstraintFact(predicate="use-robots", symbols=["Hilbert"])]
    assert list(planning_robot_states(STATES, use)) == ["hilbert"]


def test_other_constraints_still_apply_alongside():
    use = [
        ConstraintFact(predicate="forbidden-poi", symbols=["t4"]),
        ConstraintFact(predicate="use-robots", symbols=["euclid", "hilbert"]),
    ]
    assert sorted(planning_robot_states(STATES, use)) == ["euclid", "hilbert"]


def test_a_named_robot_without_a_pose_is_an_error_not_any_robot():
    use = [ConstraintFact(predicate="use-robots", symbols=["kepler"])]
    with pytest.raises(ValueError, match="kepler"):
        planning_robot_states(STATES, use)
