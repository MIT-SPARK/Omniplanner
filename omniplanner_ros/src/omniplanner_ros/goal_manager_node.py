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
from typing import Optional, Set

import rclpy
from dsg_pddl.pddl_utils import lisp_string_to_ast
from omniplanner_msgs.msg import ConstrainedPddlGoalMsg
from rclpy.node import Node
from std_msgs.msg import Bool, String

# Goal-side predicates we treat as "must visit".
_VISIT_PREDS = {"visited-place", "visited-object", "visited-poi"}


def _eval_goal_against_visited(ast, visited: Set[str]) -> Optional[bool]:
    """Three-valued evaluation of a parsed PDDL goal against a known visited set.

    Returns True if the goal is definitely satisfied by the cached plan,
    False if definitely not, None ("unknown") if the goal mentions
    predicates we don't track (e.g. ``(have ?o)``) or structure we don't
    handle (quantifiers, ``imply``, ...). Callers should treat ``None``
    as "must replan" -- replanning when we could have skipped is mere
    overhead; skipping when we should have replanned is a bug.
    """
    if isinstance(ast, str) or not ast:
        return None
    head = ast[0]
    if head == "and":
        result: Optional[bool] = True
        for sub in ast[1:]:
            v = _eval_goal_against_visited(sub, visited)
            if v is False:
                return False
            if v is None:
                result = None
        return result
    if head == "or":
        result = False
        for sub in ast[1:]:
            v = _eval_goal_against_visited(sub, visited)
            if v is True:
                return True
            if v is None:
                result = None
        return result
    if head == "not" and len(ast) == 2:
        v = _eval_goal_against_visited(ast[1], visited)
        return None if v is None else (not v)
    if head in _VISIT_PREDS and len(ast) >= 2:
        return ast[1] in visited
    # Any other predicate, quantifier, or imply: undetermined.
    return None


def goal_satisfied_by(pddl_goal: str, visited: Set[str]) -> Optional[bool]:
    """Wrap parsing + evaluation. Returns None on parse failure.

    ``lisp_string_to_ast`` does not split adjacent parens: "(a)(b)" parses as a
    single clause containing a ")(" token rather than two clauses. Goals in that
    form -- which is what the heracles in-context examples emit -- then had only
    their first conjunct evaluated, so a partly-satisfied multi-target goal could
    report True and skip a replan that was actually needed. Normalise first.
    """
    try:
        ast = lisp_string_to_ast((pddl_goal or "").replace(")(", ") ("))
    except Exception:
        return None
    return _eval_goal_against_visited(ast, visited)


def constraint_signature(constraint_facts) -> frozenset:
    """Hashable identity of a constraint set, for "did the constraints change?".

    Skipping a replan is only sound if the cached plan was built under the same
    constraints. Comparing against the plan's visited POIs is not enough: a new
    forbidden POI the plan merely passes close to would not intersect, and we
    would resume into it.
    """
    return frozenset((c.predicate, tuple(c.symbols)) for c in constraint_facts or [])


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
        # Constraints the cached plan was grounded under, so we can tell when a
        # new goal changes them and a skip would be unsound.
        self._plan_constraints: frozenset = frozenset()
        # Constraints of the plan that is actually running, kept while a
        # forwarded goal is being planned: if that planning fails, the running
        # plan is still the one built under these.
        self._running_constraints: frozenset = frozenset()
        # The goal being planned, so a planning failure can say which one.
        self._pending_goal = None

        # Subscriptions (private; remap at launch time).
        self._goal_sub = self.create_subscription(
            ConstrainedPddlGoalMsg, "~/commanded_goal", self._commanded_goal_cb, 10
        )
        self._visited_sub = self.create_subscription(
            String, "~/plan_visited_pois", self._visited_pois_cb, 10
        )
        self._failed_sub = self.create_subscription(
            String, "~/planner_failed", self._planner_failed_cb, 10
        )

        # Publishers (private; remap at launch time).
        self._goal_pub = self.create_publisher(
            ConstrainedPddlGoalMsg, "~/planner_goal", 10
        )
        self._pause_pub = self.create_publisher(Bool, "~/executor_pause", 10)
        self._resume_pub = self.create_publisher(Bool, "~/executor_resume", 10)
        # Messages for the operator: shown in the rviz instruction panel
        # (~/operator_display) and published for any other UI (~/operator_alert).
        self._display_pub = self.create_publisher(String, "~/operator_display", 10)
        self._alert_pub = self.create_publisher(String, "~/operator_alert", 10)

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
        self._running_constraints = self._plan_constraints
        self.get_logger().info(f"Updated plan cache: {len(pois)} visited POIs")

    def _planner_failed_cb(self, msg: String) -> None:
        """The forwarded goal produced no plan: carry on with the current one.

        The executor was paused when the goal arrived and only a new plan would
        have released it, so without this the robot waits for ever -- possibly
        holding an object. The goal is dropped; the operator has to restate it.
        """
        self._plan_constraints = self._running_constraints
        why = msg.data.split(": ", 1)[-1]  # drop the plugin name
        goal = self._pending_goal.goal if self._pending_goal is not None else None
        target = f"'{goal.pddl_goal}'" if goal else "the goal"
        self._resume_executor()
        self._notify_operator(
            f"Could not plan {target} ({why}). Continuing the current plan."
        )

    def _notify_operator(self, text: str) -> None:
        self.get_logger().warning(f"operator: {text}")
        self._display_pub.publish(String(data=text))
        self._alert_pub.publish(String(data=text))

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
        cached plan yet (forces an initial planning call) or whenever the goal
        evaluator returns ``None`` (unknown -- replan to be safe).
        """
        if not self._cache_valid:
            return False
        if constraint_signature(msg.constraints) != self._plan_constraints:
            return False  # constraints changed; the cached plan predates them
        forbidden = extract_forbidden_pois(msg.constraints)
        if forbidden & self._plan_visited_pois:
            return False  # current plan would step on a now-forbidden POI
        return goal_satisfied_by(msg.goal.pddl_goal, self._plan_visited_pois) is True

    # ------------- side effects -------------

    def _pause_executor(self) -> None:
        self._pause_pub.publish(Bool(data=True))

    def _resume_executor(self) -> None:
        self._resume_pub.publish(Bool(data=True))

    def _forward_for_replanning(self, msg: ConstrainedPddlGoalMsg) -> None:
        # Record what the incoming plan will be grounded under, so the next goal
        # can tell whether the constraints have since changed.
        self._plan_constraints = constraint_signature(msg.constraints)
        self._pending_goal = msg
        self._goal_pub.publish(msg)

    # ------------- logging -------------

    def _log_skip(self, msg: ConstrainedPddlGoalMsg) -> None:
        self.get_logger().info(
            f"cached plan satisfies goal; resuming without replan. "
            f"goal={msg.goal.pddl_goal!r} cached={sorted(self._plan_visited_pois)}"
        )

    def _log_replan(self, msg: ConstrainedPddlGoalMsg) -> None:
        forbidden = extract_forbidden_pois(msg.constraints)
        sat = goal_satisfied_by(msg.goal.pddl_goal, self._plan_visited_pois)
        if not self._cache_valid:
            reason = "no cached plan yet"
        elif forbidden & self._plan_visited_pois:
            reason = f"plan visits forbidden POIs {forbidden & self._plan_visited_pois}"
        elif sat is False:
            reason = "cached plan violates new goal"
        else:
            reason = "goal satisfaction unknown for cached plan"
        self.get_logger().info(
            f"replanning: {reason}. goal={msg.goal.pddl_goal!r} "
            f"forbidden={forbidden} cached={sorted(self._plan_visited_pois)}; "
            f"forwarded to planner"
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
