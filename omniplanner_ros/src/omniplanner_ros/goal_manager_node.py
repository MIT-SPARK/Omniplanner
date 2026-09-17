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
from omniplanner_msgs.msg import ConstrainedPddlGoalMsg, TaskScheduleMsg
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


def _parse_goal(pddl_goal: str):
    """Parse a PDDL goal string to an AST, or None if it does not parse.

    ``lisp_string_to_ast`` does not split adjacent parens: "(a)(b)" parses as a
    single clause containing a ")(" token rather than two clauses. Goals in that
    form -- which is what the heracles in-context examples emit -- then had only
    their first conjunct evaluated, so a partly-satisfied multi-target goal could
    report True and skip a replan that was actually needed. Normalise first.
    """
    try:
        return lisp_string_to_ast((pddl_goal or "").replace(")(", ") ("))
    except Exception:
        return None


def goal_satisfied_by(pddl_goal: str, visited: Set[str]) -> Optional[bool]:
    """Wrap parsing + evaluation. Returns None on parse failure."""
    ast = _parse_goal(pddl_goal)
    if ast is None:
        return None
    return _eval_goal_against_visited(ast, visited)


def _goal_symbols(pddl_goal: str) -> Set[str]:
    """Every POI a goal names under a visit predicate.

    Used to look each one up in the schedule. Deliberately ignores and/or/not
    structure: for a horizon check we want the symbols mentioned, not whether
    the goal as a whole holds -- ``goal_satisfied_by`` already answers that.
    """
    out: Set[str] = set()

    def walk(node):
        if isinstance(node, str) or not node:
            return
        if node[0] in ("and", "or", "not"):
            for sub in node[1:]:
                walk(sub)
        elif node[0] in _VISIT_PREDS and len(node) >= 2:
            # Same argument slot _eval_goal_against_visited matches on, so the
            # horizon check and the coverage check always mean the same symbol.
            out.add(node[1])

    walk(_parse_goal(pddl_goal))
    return out


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
        # What the planner says the whole plan covers, when it reports that.
        self._plan_covered_pois: Set[str] = set()
        self._cache_valid: bool = False
        # Constraints the cached plan was grounded under, so we can tell when a
        # new goal changes them and a skip would be unsound.
        self._plan_constraints: frozenset = frozenset()
        # Constraints of the plan that is actually running, kept while a
        # forwarded goal is being planned: if that planning fails, the running
        # plan is still the one built under these.
        self._running_constraints: frozenset = frozenset()

        # When the active plan's schedule is known, {poi -> finish time in
        # seconds}. Empty for planners that publish no schedule, which leaves
        # the horizon check inert rather than wrong.
        self._plan_finish_times: dict = {}

        # A goal already covered by the cached plan is normally skipped. If the
        # plan does not get to those POIs until after this many seconds, replan
        # anyway and see whether a fresh plan does better -- coverage alone says
        # nothing about *when*. Zero disables the check, which is the default so
        # behaviour is unchanged unless a deployment opts in.
        self.declare_parameter("replan_horizon_s", 0.0)
        self._replan_horizon_s = float(
            self.get_parameter("replan_horizon_s").value or 0.0
        )

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
        # Fleet-wide coverage, from a planner that reports it. The topic above
        # carries only this robot's share of a multi-robot plan.
        self._coverage_sub = self.create_subscription(
            String, "~/plan_coverage", self._coverage_cb, 10
        )
        self._schedule_sub = self.create_subscription(
            TaskScheduleMsg, "~/plan_schedule", self._schedule_cb, 10
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
        self._running_constraints = self._plan_constraints
        self.get_logger().info(f"Updated plan cache: {len(pois)} visited POIs")

    def _planner_failed_cb(self, msg: String) -> None:
        """The forwarded goal produced no plan: carry on with the current one.

        The executor was paused when the goal arrived and only a new plan would
        have released it, so without this the robot waits for ever -- possibly
        holding an object. The goal is dropped; the operator has to restate it.
        """
        self._plan_constraints = self._running_constraints
        self._resume_executor()
        self.get_logger().warning(
            f"planner produced no plan ({msg.data}); resuming the current plan"
        )

    def _coverage_cb(self, msg: String) -> None:
        """Every POI the active plan visits, whichever robot visits it."""
        try:
            pois = set(json.loads(msg.data))
        except Exception as exc:
            self.get_logger().warning(f"Bad plan_coverage payload: {exc}")
            return
        self._plan_covered_pois = pois
        self._cache_valid = True
        self._running_constraints = self._plan_constraints
        self.get_logger().info(f"Updated fleet coverage: {len(pois)} POIs")

    @property
    def _coverage(self) -> Set[str]:
        """This robot's POIs plus whatever the fleet plan covers elsewhere."""
        return self._plan_visited_pois | self._plan_covered_pois

    def _schedule_cb(self, msg: TaskScheduleMsg) -> None:
        """Record when the active plan reaches each POI.

        Coverage tells us a POI is in the plan; only the schedule says whether
        it is reached in ten seconds or ten minutes. Kept separate from the
        visited-POI cache because a planner may publish one and not the other.
        """
        finish = {}
        for t in msg.tasks:
            if not t.target:
                continue
            # A POI may be touched by several tasks (inspect, pick, place);
            # the plan is only done with it at the last one.
            finish[t.target] = max(finish.get(t.target, 0.0), float(t.finish_time))
        self._plan_finish_times = finish
        self.get_logger().info(
            f"Updated plan schedule: makespan={msg.makespan:.1f}s, "
            f"{len(msg.tasks)} task(s)"
        )

    def _late_pois(self, pddl_goal: str) -> Set[str]:
        """POIs this goal needs that the cached plan reaches after the horizon."""
        if self._replan_horizon_s <= 0 or not self._plan_finish_times:
            return set()
        wanted = _goal_symbols(pddl_goal)
        return {
            p
            for p in wanted
            if self._plan_finish_times.get(p, 0.0) > self._replan_horizon_s
        }

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
        if forbidden & self._coverage:
            return False  # current plan would step on a now-forbidden POI
        if goal_satisfied_by(msg.goal.pddl_goal, self._coverage) is not True:
            return False
        # Covered, but possibly not soon enough to be worth keeping.
        return not self._late_pois(msg.goal.pddl_goal)

    # ------------- side effects -------------

    def _pause_executor(self) -> None:
        self._pause_pub.publish(Bool(data=True))

    def _resume_executor(self) -> None:
        self._resume_pub.publish(Bool(data=True))

    def _forward_for_replanning(self, msg: ConstrainedPddlGoalMsg) -> None:
        # Record what the incoming plan will be grounded under, so the next goal
        # can tell whether the constraints have since changed.
        self._plan_constraints = constraint_signature(msg.constraints)
        self._goal_pub.publish(msg)

    # ------------- logging -------------

    def _log_skip(self, msg: ConstrainedPddlGoalMsg) -> None:
        self.get_logger().info(
            f"cached plan satisfies goal; resuming without replan. "
            f"goal={msg.goal.pddl_goal!r} cached={sorted(self._coverage)}"
        )

    def _log_replan(self, msg: ConstrainedPddlGoalMsg) -> None:
        forbidden = extract_forbidden_pois(msg.constraints)
        sat = goal_satisfied_by(msg.goal.pddl_goal, self._coverage)
        if not self._cache_valid:
            reason = "no cached plan yet"
        elif forbidden & self._coverage:
            reason = f"plan visits forbidden POIs {forbidden & self._coverage}"
        elif sat is False:
            reason = "cached plan violates new goal"
        elif late := self._late_pois(msg.goal.pddl_goal):
            reason = (
                f"covered, but {sorted(late)} not reached until after "
                f"{self._replan_horizon_s:.0f}s"
            )
        else:
            reason = "goal satisfaction unknown for cached plan"
        self.get_logger().info(
            f"replanning: {reason}. goal={msg.goal.pddl_goal!r} "
            f"forbidden={forbidden} cached={sorted(self._coverage)}; "
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
