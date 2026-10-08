"""The goal manager tells the operator why a goal could not be planned."""

from types import SimpleNamespace as NS

from std_msgs.msg import String

from omniplanner_msgs.msg import ConstrainedPddlGoalMsg
from omniplanner_ros.goal_manager_node import GoalManager


def _manager():
    sent = {"pause": 0, "resume": 0, "goals": [], "display": [], "alert": []}
    gm = NS(
        _plan_constraints=frozenset(),
        _running_constraints=frozenset(),
        _pending_goal=None,
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
        "_notify_operator",
        "_pause_executor",
        "_resume_executor",
    ):
        setattr(gm, name, getattr(GoalManager, name).__get__(gm))
    goal = ConstrainedPddlGoalMsg()
    goal.goal.robot_id, goal.goal.pddl_goal = "hilbert", "(visited-object o20)"
    gm._pending_goal = goal
    return gm, sent


def test_failed_operator_goal_resumes_and_says_why():
    gm, sent = _manager()
    gm._planner_failed_cb(
        String(
            data="multi_robot_pddl_constrained: Cannot satisfy the goal and "
            "constraints: o20 cannot be reached"
        )
    )
    assert sent["resume"] == 1
    (text,) = sent["display"]
    assert sent["alert"] == [text]
    assert "(visited-object o20)" in text and "o20 cannot be reached" in text
    assert "Continuing the current plan" in text
    assert "multi_robot_pddl_constrained" not in text
