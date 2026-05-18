"""Goal manager: skip replanning when the current plan already satisfies a new goal.

The node sits between the user (or upstream agent) and the omniplanner. It owns
the "should we even replan?" decision so we can avoid restarting the executor
on incremental goal updates that are already covered.

Topics are intentionally private (``~/...``); connect them to robot-specific
paths via roslaunch ``remap``::

  remap:
    - {from: ~/commanded_goal,    to: /<robot>/commanded_goal}
    - {from: ~/planner_goal,      to: /<robot>/omniplanner_node/multi_robot_pddl_constrained/pddl_goal}
    - {from: ~/plan_visited_pois, to: /<robot>/omniplanner_node/plan_visited_pois}
    - {from: ~/executor_pause,   to: /<robot>/spot_executor_node/pause}
    - {from: ~/executor_resume,  to: /<robot>/spot_executor_node/resume}

The current implementation is specific to the multi-robot PDDL "visit POIs"
domain. The decision is broken out into ``_goal_already_satisfied`` so it's
clear what would need to change to support other domains in the future.
"""

from __future__ import annotations

import json
import re
from typing import Set

import rclpy
from omniplanner_msgs.msg import ConstrainedPddlGoalMsg
from rclpy.node import Node
from std_msgs.msg import Bool, String

# Goal-side predicates we treat as "must visit"
_VISIT_PREDICATE_RE = re.compile(r"\(visited-(?:place|object|poi)\s+([^\s\)]+)\)")


def extract_visit_targets(pddl_goal: str) -> Set[str]:
    """Return the set of POI ids requested by visited-{place,object,poi} predicates."""
    return set(_VISIT_PREDICATE_RE.findall(pddl_goal))


def extract_forbidden_pois(constraint_facts) -> Set[str]:
    """Return the set of POI ids appearing in any forbidden-poi constraint."""
    out: Set[str] = set()
    for c in constraint_facts:
        if c.predicate == "forbidden-poi" and len(c.symbols) >= 1:
            out.add(c.symbols[0])
    return out


class GoalManager(Node):
    """Pause executor, decide if replan is needed, then either resume or forward."""

    def __init__(self):
        super().__init__("goal_manager")

        # Per-robot caches: latest set of POIs visited by the active plan.
        # Updated by the omniplanner's `on_plan_compiled` hook publishing on
        # ~/plan_visited_pois (which is itself a remap of a per-robot topic).
        self._plan_visited_pois: Set[str] = set()
        self._cache_valid: bool = False

        # Subscriptions (private; remap at launch time).
        self._goal_sub = self.create_subscription(
            ConstrainedPddlGoalMsg, "~/commanded_goal", self._commanded_goal_cb, 10
        )
        self._visited_sub = self.create_subscription(
            String, "~/plan_visited_pois", self._visited_pois_cb, 10
        )

        # Publishers (private; remap at launch time).
        self._goal_pub = self.create_publisher(
            ConstrainedPddlGoalMsg, "~/planner_goal", 10
        )
        self._pause_pub = self.create_publisher(Bool, "~/executor_pause", 10)
        self._resume_pub = self.create_publisher(Bool, "~/executor_resume", 10)

        self.get_logger().info("goal_manager up; awaiting goals on ~/commanded_goal")

    # ------------- callbacks -------------

    def _visited_pois_cb(self, msg: String) -> None:
        try:
            pois = set(json.loads(msg.data))
        except Exception as exc:
            self.get_logger().warning(f"Bad plan_visited_pois payload: {exc}")
            return
        self._plan_visited_pois = pois
        self._cache_valid = True
        self.get_logger().info(f"Updated plan cache: {len(pois)} visited POIs")

    def _commanded_goal_cb(self, msg: ConstrainedPddlGoalMsg) -> None:
        # Flow: pause → decide → resume_or_forward → log.
        self._pause_executor()
        if self._goal_already_satisfied(msg):
            self._resume_executor()
            self._log_skip(msg)
            return
        self._forward_for_replanning(msg)
        self._log_replan(msg)

    # ------------- decision (domain-specific) -------------

    def _goal_already_satisfied(self, msg: ConstrainedPddlGoalMsg) -> bool:
        """Return True iff the cached plan already satisfies the new goal.

        Specific to multi-robot PDDL with visited-{place,object,poi} predicates
        and forbidden-poi constraints. Returns False whenever we don't have a
        cached plan yet (forces an initial planning call).
        """
        if not self._cache_valid:
            return False
        requested = extract_visit_targets(msg.goal.pddl_goal)
        if not requested:
            return False
        forbidden = extract_forbidden_pois(msg.constraints)
        if forbidden & self._plan_visited_pois:
            return False  # current plan would step on a now-forbidden POI
        return requested.issubset(self._plan_visited_pois)

    # ------------- side effects -------------

    def _pause_executor(self) -> None:
        self._pause_pub.publish(Bool(data=True))

    def _resume_executor(self) -> None:
        self._resume_pub.publish(Bool(data=True))

    def _forward_for_replanning(self, msg: ConstrainedPddlGoalMsg) -> None:
        self._goal_pub.publish(msg)

    # ------------- logging -------------

    def _log_skip(self, msg: ConstrainedPddlGoalMsg) -> None:
        requested = extract_visit_targets(msg.goal.pddl_goal)
        self.get_logger().info(
            f"current plan already visits {requested}; resuming without replan"
        )

    def _log_replan(self, msg: ConstrainedPddlGoalMsg) -> None:
        requested = extract_visit_targets(msg.goal.pddl_goal)
        forbidden = extract_forbidden_pois(msg.constraints)
        if not self._cache_valid:
            reason = "no cached plan yet"
        elif forbidden & self._plan_visited_pois:
            reason = f"plan visits forbidden POIs {forbidden & self._plan_visited_pois}"
        else:
            missing = requested - self._plan_visited_pois
            reason = f"plan missing POIs {missing}"
        self.get_logger().info(
            f"replanning: {reason}. requested={requested} forbidden={forbidden} "
            f"cached={self._plan_visited_pois}; forwarded to planner"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
