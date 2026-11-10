"""GRSTAPS-X in the willow field test (Nov 10): a robot holding an object goes
offline, every robot is offline while the goal already holds, the robot-bound
placing goal, and goals that add work while robots are busy.

Grounding only: nothing here runs the solver.
"""

import numpy as np
import pytest
from test_grstaps_regions import G, _domain, _itags_input  # noqa: F401  (fixture)

from omniplanner.grstaps_planner import (
    MANIP,
    GrstapsGoal,
    _fleet_names,
    _identity_dims,
    enumerate_tasks,
    ground_problem,
    parse_goal_targets,
    residual_goal,
)
from omniplanner.world_state import BUSY_ROBOTS, WorldState

DURATIONS = {
    "visit-location": 5.0,
    "inspect-object": 10.0,
    "pick-object": 10.0,
    "place-object": 10.0,
}


class _C:
    def __init__(self, predicate, symbols):
        self.predicate, self.symbols = predicate, symbols


# ---------- P6: a robot holding an object has no pose ----------


def test_an_offline_robot_holding_an_object_does_not_block_the_rest(G, tmp_path):  # noqa: F811
    goal = GrstapsGoal(
        goal_points=["o0"],
        manipulate_points=["o0"],
        manipulate_destinations={"o0": "p0"},
        robot_id="willow",
        world_state=WorldState(holding={"testbot_b": ["o0"]}),
    )
    poses = {"willow": None, "testbot_a": np.zeros(3), "testbot_b": None}
    # o0 is in testbot_b's hand; testbot_a still visits it -- the plan goes on.
    ground_problem(_domain(tmp_path), G, poses, goal)
    names = [t["name"] for t in _itags_input(tmp_path, robot="willow")["tasks"]]
    assert names == ["visit-location o0 :: visit"]


# ---------- P7: nobody online and the goal already holds ----------


def test_no_robot_online_and_nothing_to_do_says_so(G, tmp_path):  # noqa: F811
    goal = GrstapsGoal(
        goal_points=["o0"], robot_id="willow", world_state=WorldState(visited={"o0"})
    )
    with pytest.raises(ValueError, match="goal already satisfied.*no robot is online"):
        ground_problem(_domain(tmp_path), G, {"willow": None, "testbot_a": None}, goal)


# ---------- P15: object-placed-by ----------


def test_object_placed_by_is_a_relocation_pinned_to_its_robot():
    found = parse_goal_targets("(object-placed-by testbot_b o58 t2260)")
    assert found["manipulate"] == ["o58"]
    assert found["destinations"] == {"o58": "t2260"}
    assert found["bound"] == {"manipulate": {"o58": ["testbot_b"]}}


def test_object_placed_by_rejects_negation_and_bad_arity():
    with pytest.raises(ValueError, match="negative"):
        parse_goal_targets("(not (object-placed-by testbot_b o58 t2260))")
    with pytest.raises(ValueError, match="Invalid arguments"):
        parse_goal_targets("(object-placed-by o58 t2260)")


def test_the_relocate_task_demands_the_named_robot():
    identity = _identity_dims(["testbot_a", "testbot_b"])
    goal = GrstapsGoal(
        goal_points=[],
        robot_id="willow",
        manipulate_points=["o58"],
        manipulate_destinations={"o58": "t1"},
        robot_bound={"manipulate": {"o58": ["testbot_b"]}},
    )
    vertex = {"o58": {"id": 0, "x": 0.0, "y": 0.0}, "t1": {"id": 1, "x": 1.0, "y": 0.0}}
    tasks, _, _ = enumerate_tasks(goal, DURATIONS, vertex, identity=identity)
    (task,) = tasks
    assert task["name"] == "relocate-object o58 :: relocate by testbot_b"
    traits = task["desired_traits"]
    assert traits[MANIP] == 1 and traits[identity["testbot_b"]] == 1
    assert traits[identity["testbot_a"]] == 0


def test_a_pinned_relocation_is_done_only_when_that_robot_placed_it(monkeypatch):
    import omniplanner.grstaps_planner as gp

    monkeypatch.setattr(gp, "_nearest_place", lambda dsg, obj: "t1")  # it is there
    goal = GrstapsGoal(
        goal_points=[],
        robot_id="willow",
        manipulate_points=["o58"],
        manipulate_destinations={"o58": "t1"},
        robot_bound={"manipulate": {"o58": ["testbot_b"]}},
        world_state=WorldState(placed_by={"testbot_a": {"o58"}}),
    )
    left, _ = residual_goal(goal, None)
    assert left.manipulate_points == ["o58"]  # testbot_a put it there, not testbot_b
    assert left.robot_bound == {"manipulate": {"o58": ["testbot_b"]}}
    goal.world_state = WorldState(placed_by={"testbot_b": {"o58"}})
    left, _ = residual_goal(goal, None)
    assert left.manipulate_points == [] and left.robot_bound == {}


# ---------- P14: busy robots ----------


def test_an_additional_task_leaves_busy_and_carrying_robots_out_of_the_fleet():
    poses = {
        "testbot_a": np.zeros(3),
        "testbot_b": np.zeros(3),
        "testbot_c": np.zeros(3),
    }
    domain = type("D", (), {"robot_species": {}})()
    goal = GrstapsGoal(
        goal_points=["o0"],
        robot_id="willow",
        constraints=[_C(BUSY_ROBOTS, ["testbot_a"])],
        world_state=WorldState(holding={"testbot_c": ["o5"]}),
    )
    assert _fleet_names(poses, goal, domain) == ["testbot_b"]
    # a repair (no busy-robots) may replan everybody
    goal.constraints = []
    assert _fleet_names(poses, goal, domain) == ["testbot_a", "testbot_b", "testbot_c"]
    # naming a busy robot makes it available
    goal.constraints = [_C(BUSY_ROBOTS, ["testbot_a"])]
    goal.robot_bound = {"visit": {"o0": ["testbot_a"]}}
    assert "testbot_a" in _fleet_names(poses, goal, domain)


def test_an_additional_task_with_everyone_busy_names_them(G, tmp_path):  # noqa: F811
    goal = GrstapsGoal(
        goal_points=["o0"],
        robot_id="willow",
        constraints=[_C(BUSY_ROBOTS, ["testbot_a"])],
    )
    with pytest.raises(ValueError, match=r"busy \(testbot_a\)"):
        ground_problem(_domain(tmp_path), G, {"testbot_a": np.zeros(3)}, goal)
