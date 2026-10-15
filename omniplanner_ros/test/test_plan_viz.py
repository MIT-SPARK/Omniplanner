"""Combined, latched multi-robot plan markers (omniplanner_ros.plan_viz)."""

import os
import time

import numpy as np
import pytest
from robot_executor_interface.action_descriptions import ActionSequence, Follow
from robot_executor_interface_ros.action_descriptions_ros import to_viz_msg
from visualization_msgs.msg import Marker, MarkerArray

from omniplanner_ros.plan_viz import (
    combine_plan_markers,
    hex_to_rgba,
    parse_robot_colors,
)


def _plan(robot, x):
    path = np.array([[x, 0.0], [x, 1.0]])
    return ActionSequence(
        plan_id=f"{robot}-plan", robot_name=robot, actions=[Follow("map", path)]
    )


def _robot_markers(robot, x, colors):
    return to_viz_msg(_plan(robot, x), robot, color=colors.get(robot)).markers


def test_parse_robot_colors():
    expected = {"a": (1.0, 0.0, 0.0, 1.0), "b": (0.0, 0.0, 1.0, 1.0)}
    assert parse_robot_colors("a:#ff0000,b:#0000ff") == expected
    assert parse_robot_colors(["a:#ff0000", "b:0000ff"]) == expected
    assert parse_robot_colors("none") == {}
    assert parse_robot_colors("a:#ff0000,bad,c:#12") == {"a": expected["a"]}
    assert hex_to_rgba("#zzzzzz") is None


def test_combine_keeps_every_robot_after_deleteall():
    colors = parse_robot_colors("a:#ff0000,b:#0000ff")
    by_robot = {"b": _robot_markers("b", 1.0, colors)}
    by_robot["a"] = _robot_markers("a", 0.0, colors)
    ma = combine_plan_markers(by_robot)
    assert ma.markers[0].action == Marker.DELETEALL
    namespaces = {m.ns.split("/")[0] for m in ma.markers[1:]}
    assert namespaces == {"a", "b"}
    line = {m.ns.split("/")[0]: m for m in ma.markers if m.type == Marker.LINE_STRIP}
    assert (line["a"].color.r, line["a"].color.b) == (1.0, 0.0)
    assert (line["b"].color.r, line["b"].color.b) == (0.0, 1.0)

    # A replanned robot replaces only its own markers.
    by_robot["a"] = _robot_markers("a", 5.0, colors)
    ma = combine_plan_markers(by_robot)
    xs = [m.points[0].x for m in ma.markers if m.type == Marker.LINE_STRIP]
    assert sorted(xs) == [1.0, 5.0]


def test_late_transient_local_subscriber_sees_both_robots():
    """Publish two robots' plans in turn; a subscriber joining afterwards gets both."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    # Same profile omniplanner_node uses for ~/compiled_plan_viz_out.
    qos = QoSProfile(
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
    )
    topic = f"test_plan_viz_{os.getpid()}"
    context = rclpy.Context()
    rclpy.init(context=context)
    try:
        node = rclpy.create_node("plan_viz_test", context=context)
        pub = node.create_publisher(MarkerArray, topic, qos)
        colors = parse_robot_colors("a:#ff0000,b:#0000ff")
        by_robot = {}
        for robot, x in (("a", 0.0), ("b", 1.0)):
            by_robot[robot] = _robot_markers(robot, x, colors)
            pub.publish(combine_plan_markers(by_robot))

        received = []
        node.create_subscription(MarkerArray, topic, received.append, qos)
        executor = rclpy.executors.SingleThreadedExecutor(context=context)
        executor.add_node(node)
        deadline = time.time() + 10.0
        while not received and time.time() < deadline:
            executor.spin_once(timeout_sec=0.1)
        assert received, "late subscriber got no latched plan markers"
        last = received[-1]
        assert last.markers[0].action == Marker.DELETEALL
        assert {m.ns.split("/")[0] for m in last.markers[1:]} == {"a", "b"}
        node.destroy_node()
    finally:
        rclpy.shutdown(context=context)
