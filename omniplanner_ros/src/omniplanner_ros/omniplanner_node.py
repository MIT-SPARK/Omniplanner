import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import rclpy
import rclpy.duration
import tf2_ros
import tf_transformations
from hydra_ros import DsgSubscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_executor_interface_ros.action_descriptions_ros import to_msg, to_viz_msg
from robot_executor_msgs.msg import ActionSequenceMsg
from ros_system_monitor_msgs.msg import NodeInfoMsg
from spark_config import Config, config_field
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import MarkerArray

from omniplanner.omniplanner import compile_plan, full_planning_pipeline

# TODO: get this import either through __init__.py or autodiscovery
from omniplanner_ros.goto_points_ros import GotoPointsConfig  # NOQA
from omniplanner_ros.language_planner_ros import LanguagePlannerConfig  # NOQA
from omniplanner_ros.ros_logging import setup_ros_log_forwarding

Logger = logging.getLogger("omniplanner")


@dataclass
class PlannerConfig(Config):
    plugin: Any = config_field("omniplanner_pipeline", required=False)


@dataclass
class OmniplannerNodeConfig(Config):
    planners: Dict[str, PlannerConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str):
        return Config.load(OmniplannerNodeConfig, path)


def get_robot_pose(
    tf_buffer, target_frame: str = "map", source_frame: str = "spot/base_link"
) -> np.ndarray:
    """
    Looks up the transform from target_frame to source_frame and returns [x, y, z, yaw].

    """
    try:
        now = rclpy.time.Time()
        tf_buffer.can_transform(
            target_frame,
            source_frame,
            now,
            timeout=rclpy.duration.Duration(seconds=1.0),
        )
        transform = tf_buffer.lookup_transform(target_frame, source_frame, now)

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert quaternion to Euler angles
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(quat)

        return np.array([translation.x, translation.y, translation.z, yaw])

    except tf2_ros.TransformException as e:
        print(f"Transform error: {e}")
        raise


# NOTE: What's the best way to deal with multiple robots / robot discovery?
# Probably tie into the general robot discovery mechanism we were thinking
# about for multi-robot SLAM. Listen for messages broadcast from each robot on
# shared "bus" topic that identifies the robot id and its transform names. In
# this node, we can store a mapping from "robot type" to supported planners (I
# guess, each plugin specifies what robots it can run on). For each robot that
# appears, we create a publisher that can publish compiled plans.  If there are
# multiple robots, goal messages may need to specify which robot they are for.
# Some, such as NaturalLanguage, don't need to specify which robot they are for
# because the robot allocation is explicitly part of the planning process.


class OmniPlannerRos(Node):
    def __init__(self):
        super().__init__("omniplanner_ros")
        self.get_logger().info("Setting up omniplanner")

        # forward python logging to ROS
        setup_ros_log_forwarding(self, Logger)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_dsg_time = 0
        self.current_planner = None
        self.plan_time_start = None
        self.dsg_last = None

        self.last_dsg_time_lock = threading.Lock()
        self.current_planner_lock = threading.Lock()
        self.plan_time_start_lock = threading.Lock()

        self.dsg_lock = threading.Lock()
        DsgSubscriber(self, "~/dsg_in", self.dsg_callback)

        # TODO: need to generalize this cross robots.
        # When we discover a new robot, we should create a new
        # publisher based on the information that the robot provides.
        # Then we can look up the relevant publisher in the {name: publishers} map
        self.compiled_plan_pub = self.create_publisher(
            ActionSequenceMsg, "~/compiled_plan_out", 1
        )

        latching_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.compiled_plan_viz_pub = self.create_publisher(
            MarkerArray, "~/compiled_plan_viz_out", qos_profile=latching_qos
        )

        self.heartbeat_pub = self.create_publisher(NodeInfoMsg, "~/node_status", 1)
        heartbeat_timer_group = MutuallyExclusiveCallbackGroup()
        timer_period_s = 0.1
        self.timer = self.create_timer(
            timer_period_s, self.hb_callback, callback_group=heartbeat_timer_group
        )

        # process virtual config
        self.declare_parameter("plugin_config_path", "")
        config_path = self.get_parameter("plugin_config_path").value
        assert config_path != "", "plugin_config_path cannot be empty"

        self.config = OmniplannerNodeConfig.load(config_path)
        for name, planner in self.config.planners.items():
            plugin = planner.plugin.create()
            self.register_plugin(name, plugin)

    def dsg_callback(self, header, dsg):
        self.get_logger().warning("Setting DSG!")

        with self.last_dsg_time_lock and self.dsg_lock:
            self.dsg_last = dsg
            self.last_dsg_time = time.time()

    def hb_callback(self):
        with (
            self.last_dsg_time_lock
            and self.current_planner_lock
            and self.plan_time_start_lock
        ):
            dsg_age = time.time() - self.last_dsg_time
            have_recent_dsg = dsg_age < 15

            notes = ""
            if not have_recent_dsg:
                notes += f"No recent dsg ({dsg_age} (s) old)"

            if self.current_planner is None:
                if have_recent_dsg:
                    notes += "Ready to plan!"
            else:
                elapsed_planning_time = time.time() - self.plan_time_start
                notes += f"Running {self.current_planner} ({elapsed_planning_time} s)"

        status = NodeInfoMsg.NOMINAL
        if not have_recent_dsg:
            status = NodeInfoMsg.WARNING

        msg = NodeInfoMsg()
        msg.nickname = "omniplanner"
        msg.node_name = self.get_fully_qualified_name()
        msg.status = status
        msg.notes = notes
        self.heartbeat_pub.publish(msg)

    def get_spot_pose(self):
        # TODO: parameters
        return get_robot_pose(
            self.tf_buffer, target_frame="map", source_frame="spot/base_link"
        )

    def register_plugin(self, name, plugin):
        self.get_logger().info(f"Registering subscription plugin {name}")
        msg_type, topic, callback = plugin.get_plan_callback()

        def plan_handler(msg):
            self.get_logger().info(f"Handling plan for plugin {name}")

            if self.dsg_last is None:
                self.get_logger().error("Got plan request, but no DSG!")
                return

            with self.current_planner_lock and self.plan_time_start_lock:
                self.current_planner = name
                self.plan_time_start = time.time()

            robot_poses = {"spot": self.get_spot_pose()}

            plan_request = callback(msg, robot_poses)
            with self.dsg_lock:
                plan = full_planning_pipeline(plan_request, self.dsg_last)

            spot_path_frame = "map"  # TODO: parameter
            compiled_plan = compile_plan(
                plan, str(uuid.uuid4()), "spot", spot_path_frame
            )

            self.compiled_plan_pub.publish(to_msg(compiled_plan))
            self.compiled_plan_viz_pub.publish(to_viz_msg(compiled_plan, name))

            with self.current_planner_lock and self.plan_time_start_lock:
                self.current_planner = None
                self.plan_time_start = None
            self.get_logger().info("Published Plan")

        resolved_topic_name = name + "/" + topic
        self.get_logger().info(
            f"Registering subscription for {resolved_topic_name} (type {str(msg_type)})"
        )
        self.create_subscription(
            msg_type,
            f"~/{resolved_topic_name}",
            plan_handler,
            1,
        )


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OmniPlannerRos()
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
