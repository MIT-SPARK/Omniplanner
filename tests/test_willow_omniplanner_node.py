"""The omniplanner node in the willow field test (Nov 10): offline robots whose
last transforms fleetlink keeps for hours, and goals that already hold.

No ROS graph: the tf buffer, clock and publishers are faked.
"""

import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
import tf2_ros
from nav_msgs.msg import Path
from robot_executor_interface.action_descriptions import ActionSequence, Follow

node_mod = pytest.importorskip("omniplanner_ros.omniplanner_node")
N = node_mod.OmniPlannerRos


class _Tf:
    """A tf buffer that has a transform for some robots, stamped at stamp_s."""

    def __init__(self, stamps):
        self.stamps = stamps
        self.timeouts = []

    def can_transform(self, parent, child, time, timeout=None):
        self.timeouts.append(timeout.nanoseconds if timeout is not None else None)
        return child in self.stamps

    def lookup_transform(self, parent, child, time, timeout=None):
        if child not in self.stamps:
            raise tf2_ros.LookupException(child)
        t = NS(
            header=NS(stamp=NS(sec=int(self.stamps[child]), nanosec=0)),
            transform=NS(
                translation=NS(x=1.0, y=2.0, z=0.0),
                rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
        return t


def _adaptor(name, tf):
    a = node_mod.RobotPlanningAdaptor.__new__(node_mod.RobotPlanningAdaptor)
    a.tf_buffer, a.name, a.child_frame = tf, name, f"{name}/base_link"
    a.robot_type, a.ros_logger = "spot", None
    return a


def _node(stamps, now_s=100.0, max_age_s=5.0):
    node = N.__new__(N)
    tf = _Tf({f"{r}/base_link": s for r, s in stamps.items()})
    logs = []
    node.__dict__.update(
        robot_adaptors={
            r: _adaptor(r, tf) for r in ("testbot_a", "testbot_b", "testbot_c")
        },
        dsg_frame="map",
        _max_pose_age_s=max_age_s,
        _pose_status={},
        f_logs=logs,
        f_tf=tf,
        f_now={"t": now_s},
    )
    node.get_logger = lambda: NS(
        info=lambda m: logs.append(("info", m)),
        warning=lambda m: logs.append(("warning", m)),
        error=lambda m: logs.append(("error", m)),
    )
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=int(node.f_now["t"] * 1e9)))
    return node


# ---------- P3 / P10: stale and missing poses ----------


def test_a_stale_pose_is_no_pose_and_a_fresh_one_is():
    # testbot_a updated a second ago; testbot_b's last transform is an hour old
    # (fleetlink keeps it); testbot_c never had one.
    node = _node({"testbot_a": 9999, "testbot_b": 10000 - 3600}, now_s=10000.0)
    poses = node.get_robot_poses("map")
    np.testing.assert_allclose(poses["testbot_a"], [1.0, 2.0, 0.0, 0.0])
    assert poses["testbot_b"] is None and poses["testbot_c"] is None


def test_lookups_never_wait():
    node = _node({"testbot_a": 99})
    node.get_robot_poses("map")
    assert node.f_tf.timeouts and all(t == 0 for t in node.f_tf.timeouts)


def test_a_robot_going_stale_and_back_is_logged_once_each_way():
    node = _node({"testbot_a": 99, "testbot_b": 98, "testbot_c": 99})
    node.get_robot_poses("map")
    assert node.f_logs == []  # all fresh: nothing to say
    node.f_now["t"] = 110.0  # nobody updated for 11 s
    node.f_tf.stamps["testbot_a/base_link"] = 109
    for _ in range(3):
        node.get_robot_poses("map")
    warned = [m for level, m in node.f_logs if level == "warning"]
    assert len(warned) == 2  # testbot_b and testbot_c, once each
    assert all("old" in m for m in warned)
    node.f_tf.stamps["testbot_b/base_link"] = 110
    node.get_robot_poses("map")
    node.get_robot_poses("map")
    assert [m for level, m in node.f_logs if level == "info"] == [
        "testbot_b: pose is fresh again"
    ]


def test_sim_time_compares_on_the_node_clock():
    # Sim time: the clock and the stamps are both small numbers.
    node = _node({"testbot_a": 12}, now_s=14.0)
    assert node.robot_pose("testbot_a") is not None
    node.f_now["t"] = 30.0
    assert node.robot_pose("testbot_a") is None


def test_a_static_transform_is_fresh():
    node = _node({"testbot_a": 0}, now_s=5000.0)
    assert node.robot_pose("testbot_a") is not None


def test_no_age_limit_takes_any_pose():
    node = _node({"testbot_a": 1}, now_s=5000.0, max_age_s=0.0)
    assert node.robot_pose("testbot_a") is not None


# ---------- P8: a goal that already holds ----------


def _planning_node(monkeypatch, plan_dict):
    node = N.__new__(N)
    sent = {"plans": [], "failed": [], "robots": []}
    handlers = []
    node.__dict__.update(
        goal_count=0,
        goal_count_lock=threading.Lock(),
        planning_lock=threading.Lock(),
        current_planner_lock=threading.Lock(),
        plan_time_start_lock=threading.Lock(),
        dsg_lock=threading.Lock(),
        current_planner=None,
        plan_time_start=None,
        dsg_last=object(),
        dsg_frame="map",
        goal_group=None,
        feedback=NS(plugin_feedback_collectors={}),
        robot_adaptors={
            r: NS(publish_plan=lambda m, r=r: sent["plans"].append(r))
            for r in plan_dict
        },
        plan_failed_pub=NS(publish=lambda m: sent["failed"].append(m.data)),
        plan_robots_pub=NS(publish=lambda m: sent["robots"].append(m.data)),
        compiled_plan_viz_pub=NS(publish=lambda m: None),
    )
    node.get_logger = lambda: NS(info=print, warning=print, error=print)
    node.get_robot_poses = lambda frame: {r: None for r in plan_dict}
    node.create_subscription = (
        lambda t, topic, cb, q, callback_group=None: handlers.append(cb)
    )
    monkeypatch.setattr(node_mod, "full_planning_pipeline", lambda *a: None)
    monkeypatch.setattr(node_mod, "compile_plan", lambda *a: None)
    monkeypatch.setattr(node_mod, "collect_plans", lambda p: dict(plan_dict))
    monkeypatch.setattr(node_mod, "to_msg", lambda p: p)
    monkeypatch.setattr(node_mod, "to_viz_msg", lambda p, r: None)
    plugin = NS(
        get_plan_callback=lambda: (object, "pddl_goal", lambda msg, poses: None),
        get_plugin_feedback=lambda n: None,
    )
    N.register_plugin(node, "multi_robot_pddl_constrained", plugin)
    return handlers[0], sent


def test_an_already_met_goal_dispatches_nothing_and_says_so(monkeypatch):
    # (object-in-place o58 t2260) with o58 already there: every plan empty.
    handle, sent = _planning_node(
        monkeypatch,
        {"testbot_a": ActionSequence("p", "testbot_a", []), "testhusky": Path()},
    )
    handle(NS())
    assert sent["plans"] == [] and sent["robots"] == []
    assert sent["failed"] == [
        "multi_robot_pddl_constrained: goal already satisfied: nothing to do"
    ]


def test_a_plan_with_work_is_dispatched(monkeypatch):
    handle, sent = _planning_node(
        monkeypatch,
        {
            "testbot_a": ActionSequence(
                "p", "testbot_a", [Follow("map", np.zeros((2, 2)))]
            ),
            "testhusky": Path(),
        },
    )
    handle(NS())
    assert sorted(sent["plans"]) == ["testbot_a", "testhusky"]
    assert sent["failed"] == []


def test_no_robot_at_all_is_a_failure_not_a_published_plan(monkeypatch):
    handle, sent = _planning_node(monkeypatch, {})
    handle(NS())
    assert sent["failed"] == [
        "multi_robot_pddl_constrained: no robot is online to plan for"
    ]
