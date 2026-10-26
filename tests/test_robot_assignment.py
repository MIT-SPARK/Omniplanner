"""A goal can say which robot does which task: (visited-object-by r o).

Without it every visit is "some robot", and fast-downward hands all the work to
whichever robot is cheapest, whatever the operator assigned.

Needs the plan_repair_graph prior map (OMNIPLANNER_TEST_DSG) and fast-downward;
skips without them.
"""

from importlib.resources import files

import dsg_pddl.domains
import numpy as np
from dsg_pddl.dsg_pddl_grounding_multirobot import generate_multirobot_region_pddl
from dsg_pddl.pddl_grounding import MultiRobotPddlDomain, PddlGoal
from test_world_state import _map, _pos, needs_map

from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner.world_state import WorldState, WorldStateTracker


def _fd(G, goal, robot_states, world_state=None):
    text = (
        files(dsg_pddl.domains)
        .joinpath("RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl")
        .read_text()
    )
    req = PlanRequest(
        domain=MultiRobotPddlDomain(text),
        goal=PddlGoal(pddl_goal=goal, robot_id="willow", world_state=world_state),
        robot_states=robot_states,
    )
    plan = full_planning_pipeline(req, G)
    while hasattr(plan, "value"):
        plan = plan.value
    return plan.symbolic_actions


def _targets_by_robot(actions):
    out = {}
    for a in actions:
        out.setdefault(a[1], set()).add(a[-1])
    return out


@needs_map
def test_each_robot_visits_what_it_was_assigned():
    G = _map()
    # Both start beside o2, so unassigned, one robot would take both visits.
    start = np.array(_pos(G, "t3")[:2])
    robots = {"hamilton": start, "gauss": start + 0.1}
    goal = "(and (visited-object-by hamilton o16) (visited-object-by gauss o2))"
    by_robot = _targets_by_robot(_fd(G, goal, robots))
    assert "o16" in by_robot.get("hamilton", set())
    assert "o2" in by_robot.get("gauss", set())


@needs_map
def test_inspection_by_a_named_robot():
    G = _map()
    start = np.array(_pos(G, "t3")[:2])
    robots = {"hamilton": start, "gauss": start + 0.1}
    actions = _fd(G, "(inspected-by gauss o2)", robots)
    inspections = [a for a in actions if a[0] == "inspect"]
    assert [a[1] for a in inspections] == ["gauss"]


@needs_map
def test_a_robot_s_own_visits_are_not_redone():
    G = _map()
    start = np.array(_pos(G, "t3")[:2])
    robots = {"hamilton": start, "gauss": start + 0.1}
    done = WorldState(visited={"o16"}, visited_by={"hamilton": {"o16"}})
    goal = "(and (visited-object-by hamilton o16) (visited-object-by gauss o2))"
    by_robot = _targets_by_robot(_fd(G, goal, robots, done))
    assert "o16" not in by_robot.get("hamilton", set())


@needs_map
def test_assigning_a_robot_without_a_pose_says_so():
    G = _map()
    robots = {"hamilton": np.array(_pos(G, "t3")[:2]), "lewis": None}
    _, _, hint = generate_multirobot_region_pddl(
        G, "(visited-object-by lewis o2)", robots
    )
    assert "lewis" in hint and "no pose" in hint


def test_tracker_records_who_did_what():
    t = WorldStateTracker()
    t.record("GAZE", True, "Gauss", object_id="O2")
    t.record("GAZE", False, "hamilton", object_id="o5")
    s = t.snapshot()
    assert s.inspected_by == {"gauss": {"o2"}}
    t.reset()
    assert t.snapshot().inspected_by == {}
