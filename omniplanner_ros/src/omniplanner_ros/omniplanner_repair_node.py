"""Omniplanner node with runtime constraint plumbing.

Thin subclass of :class:`omniplanner_ros.omniplanner_node.OmniPlannerRos` that
adds support for a persistent set of runtime constraints (e.g.
``forbidden-poi``/``forbidden-edge``).

Other plugins that want to consume the persistent constraints look at
``node.active_constraints``; the ``MultiRobotPddlConstrainedPlannerRos``
plugin in :mod:`omniplanner_ros.multirobot_ros` merges them with
per-message constraints before grounding.

This module does **not** override ``register_plugin`` or duplicate the
plan-handling logic in ``OmniPlannerRos``; the plugin-side ``pddl_callback``
and the upstream ``plugin.on_plan_compiled`` hook handle the per-call work.
"""

from __future__ import annotations

import threading

import rclpy
from dsg_pddl.pddl_grounding import ConstraintFact
from omniplanner_msgs.msg import ConstraintList
from rclpy.executors import MultiThreadedExecutor

from omniplanner_ros.omniplanner_node import OmniPlannerRos


class OmniPlannerRepairRos(OmniPlannerRos):
    """OmniPlannerRos plus a ``~/constraints`` topic for persistent constraints."""

    def __init__(self):
        # Initialize constraint state before super().__init__() so plugins
        # registered during the parent constructor can read it safely if they
        # need to.
        self._constraints_lock = threading.Lock()
        self.active_constraints: list[ConstraintFact] = []

        super().__init__()

        self.create_subscription(
            ConstraintList,
            "~/constraints",
            self._constraints_callback,
            10,
        )
        self.get_logger().info(
            "OmniPlannerRepairRos listening for persistent constraints on ~/constraints"
        )

    def _constraints_callback(self, msg: ConstraintList) -> None:
        new_constraints = [
            ConstraintFact(predicate=c.predicate, symbols=list(c.symbols))
            for c in msg.constraints
        ]
        with self._constraints_lock:
            self.active_constraints = new_constraints
        self.get_logger().info(
            f"Updated persistent constraints: {len(new_constraints)} fact(s)"
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
