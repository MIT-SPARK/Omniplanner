"""The goal manager stops and tells the operator when the world breaks the plan."""

import json
from types import SimpleNamespace as NS

from std_msgs.msg import String

from omniplanner_msgs.msg import ConstrainedPddlGoalMsg
from omniplanner_ros.goal_manager_node import GoalManager


def _manager():
    sent = {"pause": 0, "resume": 0, "stop": 0, "goals": [], "display": [], "alert": []}
    gm = NS(
        _plan_constraints=frozenset(),
        _running_constraints=frozenset(),
        _running_goal=None,
        _pending_goal=None,
        _replan_reason=None,
        _plan_visited_pois=set(),
        _cache_valid=False,
        _plan_robots={"hilbert"},
        _paused={"hilbert"},
        _executor_cmd=lambda robot, verb: sent.__setitem__(verb, sent[verb] + 1),
        _goal_pub=NS(publish=sent["goals"].append),
        _display_pub=NS(publish=lambda m: sent["display"].append(m.data)),
        _alert_pub=NS(publish=lambda m: sent["alert"].append(m.data)),
        get_logger=lambda: NS(info=lambda *_: None, warning=lambda *_: None),
    )
    for name in (
        "_planner_failed_cb",
        "_replan_request_cb",
        "_notify_operator",
        "_pause_executor",
        "_resume_executor",
        "_stop_executor",
        "_visited_pois_cb",
    ):
        setattr(gm, name, getattr(GoalManager, name).__get__(gm))
    goal = ConstrainedPddlGoalMsg()
    goal.goal.robot_id, goal.goal.pddl_goal = "hilbert", "(visited-object o20)"
    gm._pending_goal = gm._running_goal = goal
    return gm, sent


def test_failed_operator_goal_stops_and_says_why():
    gm, sent = _manager()
    gm._planner_failed_cb(
        String(
            data="multi_robot_pddl_constrained: Cannot satisfy the goal and "
            "constraints: o20 cannot be reached"
        )
    )
    assert sent["stop"] == 1 and sent["resume"] == 0
    (text,) = sent["display"]
    assert sent["alert"] == [text]
    assert "(visited-object o20)" in text and "o20 cannot be reached" in text
    assert "Stopped; waiting for a new goal" in text
    assert "multi_robot_pddl_constrained" not in text


def test_failed_scene_change_replan_stops_and_tells_the_operator():
    gm, sent = _manager()
    gm._replan_request_cb(String(data="scene changed: o20 removed before the visit"))
    gm._planner_failed_cb(
        String(data="grstaps_planner: Goal symbol(s) not in the DSG: o20")
    )
    assert sent["stop"] == 1 and sent["resume"] == 0  # the old plan drives to o20
    (text,) = sent["display"]
    assert sent["alert"] == [text]
    assert "o20 removed" in text and "not in the DSG: o20" in text
    assert "hilbert stopped" in text and "grstaps_planner" not in text


def test_a_scene_change_stops_the_robots_even_with_resume_on_failure():
    gm, sent = _manager()
    gm._resume_on_failure = True
    gm._replan_request_cb(String(data="scene changed: o20 removed before the visit"))
    gm._planner_failed_cb(String(data="p: no plan"))
    assert sent["stop"] == 1 and sent["resume"] == 0


def test_a_replan_that_succeeded_is_not_held_against_a_later_failure():
    gm, sent = _manager()
    gm._replan_request_cb(String(data="scene changed: o4 moved"))
    gm._visited_pois_cb(String(data=json.dumps(["o4"])))  # the new plan arrived
    gm._planner_failed_cb(String(data="p: no plan"))
    assert "Cannot repair" not in sent["display"][0]
