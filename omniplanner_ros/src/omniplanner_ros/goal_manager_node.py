"""Goal manager node for plan repair: skip replanning when current plan already achieves the new goal."""

import json

import rclpy
from omniplanner_msgs.msg import PddlGoalMsg
from rclpy.node import Node
from std_msgs.msg import Bool, String

from omniplanner_ros.plan_utils import (
    extract_forbidden_pois,
    extract_visited_places,
    plan_visits_all,
)


class GoalManager(Node):
    def __init__(self):
        super().__init__("goal_manager")

        self.declare_parameter("commanded_goal_topic", "/hilbert/commanded_pddl_goal")
        self.declare_parameter(
            "omniplanner_goal_topic",
            "/hilbert/omniplanner_node/multi_robot_pddl/pddl_goal",
        )
        self.declare_parameter("robot_names", ["hilbert"])

        commanded_topic = self.get_parameter("commanded_goal_topic").value
        omniplanner_topic = self.get_parameter("omniplanner_goal_topic").value
        robot_names = self.get_parameter("robot_names").value

        self.current_goal_formula = {}  # robot_id -> PDDL string
        self.stop_publishers = {}  # robot_id -> Publisher(Bool) for executor stop topic
        self.plan_visited_pois = {}  # robot_id -> Set[str] of POIs in current plan

        self.pub = self.create_publisher(PddlGoalMsg, omniplanner_topic, 10)
        self.sub = self.create_subscription(
            PddlGoalMsg,
            commanded_topic,
            self.commanded_goal_cb,
            10,
        )

        # Subscribe to visited-POI updates published by omniplanner_node
        for rn in robot_names:
            topic = f"/{rn}/omniplanner_node/plan_visited_pois"
            self.create_subscription(
                String,
                topic,
                lambda msg, robot=rn: self._visited_pois_cb(robot, msg),
                10,
            )
            self.get_logger().info(f"Subscribing to {topic} for plan-repair cache")

    def _visited_pois_cb(self, robot_id: str, msg: String):
        pois = set(json.loads(msg.data))
        self.plan_visited_pois[robot_id] = pois
        self.get_logger().info(
            f"[{robot_id}] updated plan cache: {len(pois)} visited POIs"
        )

    def commanded_goal_cb(self, msg: PddlGoalMsg):
        robot_id = msg.robot_id
        requested_pois = extract_visited_places(msg.pddl_goal)

        # First, always request the executor to pause the current plan so the
        # robot visibly pauses before we decide whether to replan.
        if robot_id not in self.stop_publishers:
            pause_topic = f"/{robot_id}/spot_executor_node/pause"
            self.stop_publishers[robot_id] = self.create_publisher(
                Bool, pause_topic, 10
            )
        stop_msg = Bool()
        stop_msg.data = True
        self.stop_publishers[robot_id].publish(stop_msg)

        # Plan-repair logic: if the current plan already visits all requested
        # POIs and doesn't violate any constraints, resume without replanning.
        cached_pois = self.plan_visited_pois.get(robot_id, set())
        self.get_logger().info(f"[{robot_id}] msg.constraints raw: '{msg.constraints}'")
        forbidden = extract_forbidden_pois(msg.constraints)
        if cached_pois and plan_visits_all(cached_pois, requested_pois, forbidden):
            resume_topic = f"/{robot_id}/spot_executor_node/resume"
            resume_pub = self.create_publisher(Bool, resume_topic, 10)
            resume_msg = Bool()
            resume_msg.data = True
            resume_pub.publish(resume_msg)
            self.get_logger().info(
                f"[{robot_id}] current plan already visits all requested POIs "
                f"{requested_pois} without forbidden {forbidden}, resuming"
            )
            return

        # Not satisfied: forward to omniplanner for replanning.
        self.current_goal_formula[robot_id] = msg.pddl_goal

        out = PddlGoalMsg()
        out.robot_id = robot_id
        out.pddl_goal = msg.pddl_goal
        out.constraints = msg.constraints  # forward constraints to omniplanner
        self.pub.publish(out)

        # Log why we're replanning
        if forbidden and forbidden & cached_pois:
            reason = f"plan visits forbidden POIs {forbidden & cached_pois}"
        elif not requested_pois.issubset(cached_pois):
            reason = f"plan missing POIs {requested_pois - cached_pois}"
        else:
            reason = "no cached plan"
        self.get_logger().info(
            f"[{robot_id}] replanning: {reason}. requested={requested_pois} forbidden={forbidden} "
            f"(cached: {cached_pois}), sent goal to omniplanner"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
