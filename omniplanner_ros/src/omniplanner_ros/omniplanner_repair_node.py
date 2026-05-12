"""Omniplanner node with execution-time plan repair and runtime constraints.

Extends OmniPlannerRos with:
- Runtime constraints (forbidden-poi, forbidden-edge) via ~/constraints topic
  and per-message constraints in PddlGoalMsg.constraints
- Visited POI publishing for goal_manager viability checks
- Better error handling with traceback logging
"""

import json
import logging
import threading
import time
import traceback

import rclpy
from omniplanner.compile_plan import collect_plans, compile_plan
from omniplanner.omniplanner import full_planning_pipeline
from rclpy.executors import MultiThreadedExecutor
from robot_executor_interface_ros.action_descriptions_ros import to_msg, to_viz_msg
from std_msgs.msg import String

from omniplanner_ros.last_pddl_plans import get_last_plan
from omniplanner_ros.omniplanner_node import OmniPlannerRos
from omniplanner_ros.plan_utils import extract_visited_pois_from_plan

logger = logging.getLogger(__name__)


class OmniPlannerRepairRos(OmniPlannerRos):
    """OmniPlannerRos with plan repair, constraints, and POI publishing."""

    def __init__(self):
        # Add constraints state BEFORE super().__init__() because
        # register_plugin is called during super().__init__()
        self._repair_pre_init()
        super().__init__()
        self._repair_post_init()

    def _repair_pre_init(self):
        """Initialize repair state before parent __init__ (which calls register_plugin)."""
        self.constraints_lock = threading.Lock()
        self.active_constraints = []

    def _repair_post_init(self):
        """Set up repair-specific subscriptions and publishers after parent __init__."""
        # Subscribe to constraints topic
        self.create_subscription(
            String,
            "~/constraints",
            self._constraints_callback,
            10,
        )
        self.get_logger().info("Listening for constraints on ~/constraints")

        # Add visited POI publishers to each robot adaptor
        for name, adaptor in self.robot_adaptors.items():
            adaptor.visited_pois_pub = self.create_publisher(
                String, f"/{name}/omniplanner_node/plan_visited_pois", 1
            )

    def _constraints_callback(self, msg: String):
        """Accept a JSON-encoded list of constraint facts.

        Each fact is a list like ["forbidden-poi", "p123"] or
        ["forbidden-edge", "p1", "p2"]. Replaces all active constraints.
        Send [] to clear.
        """
        try:
            raw = json.loads(msg.data)
            constraints = [tuple(c) for c in raw]
            with self.constraints_lock:
                self.active_constraints = constraints
            self.get_logger().info(f"Updated constraints: {constraints}")
        except Exception as e:
            self.get_logger().error(f"Bad constraints message: {e}")

    def register_plugin(self, name, plugin):
        """Override to add constraints injection, POI publishing, and error handling."""
        self.get_logger().info(f"Registering subscription plugin {name}")
        msg_type, topic, callback = plugin.get_plan_callback()
        self.feedback.plugin_feedback_collectors[name] = plugin.get_plugin_feedback(
            self
        )

        def plan_handler(msg):
            self.get_logger().info(f"Handling plan for plugin {name}")

            if self.dsg_last is None:
                self.get_logger().error("Got plan request, but no DSG!")
                return

            with self.current_planner_lock and self.plan_time_start_lock:
                self.current_planner = name
                self.plan_time_start = time.time()

            try:
                # Log the received message
                goal_str = getattr(msg, "pddl_goal", str(msg))
                constraints_str = getattr(msg, "constraints", "")
                self.get_logger().info(
                    f"=== Received goal: {goal_str}"
                    + (f" | constraints: {constraints_str}" if constraints_str else "")
                    + " ==="
                )

                robot_poses = self.get_robot_poses(self.dsg_frame)
                self.get_logger().info(f"Planning with robot poses {robot_poses}")

                plan_request = callback(msg, robot_poses)

                # Merge constraints: persistent (from ~/constraints topic)
                # + per-message (from goal msg .constraints field)
                with self.constraints_lock:
                    combined = list(self.active_constraints)
                msg_constraints = getattr(msg, "constraints", "")
                if msg_constraints:
                    try:
                        extra = [tuple(c) for c in json.loads(msg_constraints)]
                        combined.extend(extra)
                        with self.constraints_lock:
                            for c in extra:
                                if c not in self.active_constraints:
                                    self.active_constraints.append(c)
                        self.get_logger().info(
                            f"Added {len(extra)} constraints from goal message"
                        )
                    except Exception as e:
                        self.get_logger().error(f"Bad constraints in goal msg: {e}")
                plan_request.constraints = combined

                with self.dsg_lock:
                    plans = full_planning_pipeline(
                        plan_request, self.dsg_last, self.feedback
                    )

                compiled_plans = compile_plan(
                    self.robot_adaptors, self.dsg_frame, plans
                )
                plan_dict = collect_plans(compiled_plans)
                for robot_name, compiled_plan in plan_dict.items():
                    self.robot_adaptors[robot_name].publish_plan(to_msg(compiled_plan))
                    self.compiled_plan_viz_pub.publish(
                        to_viz_msg(compiled_plan, robot_name)
                    )

                # Log the symbolic plan clearly
                for robot_name in plan_dict:
                    cached = get_last_plan(robot_name)
                    if cached is not None:
                        self.get_logger().info(
                            f"=== Symbolic plan for {robot_name} ==="
                        )
                        for i, action in enumerate(cached.symbolic_actions):
                            self.get_logger().info(f"  [{i}] {action}")

                # Publish visited POIs per robot for goal_manager viability check
                for robot_name in plan_dict:
                    cached = get_last_plan(robot_name)
                    if cached is not None and hasattr(
                        self.robot_adaptors[robot_name], "visited_pois_pub"
                    ):
                        pois = extract_visited_pois_from_plan(cached)
                        poi_msg = String()
                        poi_msg.data = json.dumps(sorted(pois))
                        self.robot_adaptors[robot_name].visited_pois_pub.publish(
                            poi_msg
                        )

                self.get_logger().info("Published Plan")
            except Exception as e:
                self.get_logger().error(
                    f"Planning failed for plugin {name}: {e}\n{traceback.format_exc()}"
                )
            finally:
                with self.current_planner_lock and self.plan_time_start_lock:
                    self.current_planner = None
                    self.plan_time_start = None

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
        node = OmniPlannerRepairRos()
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
