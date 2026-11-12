"""Goal manager: decide, per goal, what the planner still has to do.

The node sits between the user (or upstream agent) and the omniplanner. It owns
the "should we even replan?" decision so we can avoid restarting the executor
on incremental goal updates that are already covered.

Topics are intentionally private (``~/...``); connect them to robot-specific
paths via roslaunch ``remap``::

  remap:
    - {from: ~/commanded_goal,    to: /<robot>/commanded_goal}
    - {from: ~/planner_goal,      to: /<robot>/omniplanner_node/multi_robot_pddl_constrained/pddl_goal}
    - {from: ~/plan_robots,       to: /<robot>/omniplanner_node/plan_robots}
    - {from: ~/map_version,       to: /<robot>/omniplanner_node/map_version}

What each robot is doing is kept per robot, from the planner's and executors'
own absolute topics: ``/<name>/omniplanner_node/compiled_plan_out`` (its action
sequence: plan id and length), ``/<name>/omniplanner_node/plan_visited_pois``
(the POIs its plan visits) and ``/action_done`` (each finished action). A Husky
reports no actions; the planner follows its pose along its Path and says on
``/<name>/omniplanner_node/path_status`` when it has finished or stalled. A
Spot moving is reported there as progress, so a long action is not a stall. A robot
that finishes moves what it visited into a set of achieved facts; one that
fails or makes no progress (``progress_timeout_s``) gives its part of the goal
back. A new goal is split into conjuncts, each done, in progress or new; only a
goal with something new, or a lost robot, goes to the planner, and only a goal
whose every conjunct is achieved is skipped. A goal that does not touch the
running mission while robots are busy goes as an additional task, with a
``busy-robots`` constraint naming the robots the planner should leave alone.
The per-robot state is latched on ``~/robot_plan_states`` (JSON).

A goal with a ``use-robots`` constraint, or whose every open conjunct names
a robot (in a ``*-by``, ``holding``, ``at-*`` or ``in-region`` atom), is
robot-scoped unless it names the whole fleet: it goes to the planner with
``use-robots`` naming only those robots, only they are paused, and every other
robot's plan, standing goal and held objects are left alone. Each robot keeps
the last scoped goal it was given, so goals for different robots compose. A
goal with a conjunct naming no robot (and no ``use-robots``) is the team's,
even if other conjuncts name robots, and replaces them all. If the goal moves or holds an object another
robot holds or its plan acts on, that robot is replanned too, with the rest of
its goal, and the operator is told. Where a robot is or what it holds (at-*,
in-region, holding) is never remembered as done: it stops holding when the
robot moves.

A pause reaches a robot on another topic than its plan (over fleetlink it is
retried for up to its ttl), so one sent before a new plan can arrive after it
and stop the new plan for good. Every pause on a Spot's pause topic is
watched; a robot given a plan within ``stale_pause_window_s`` of one is
resumed until that pause can no longer arrive, unless it is paused again.

Executors are addressed directly: a Spot as ``/<name>/spot_executor_node/{pause,
resume,stop}``; a Husky (phoenix, which has no pause) is stopped with an empty
Path on its ``compiled_plan_out``. The platform comes from the planner's robots
config (``plugin_config_path``, the omniplanner_plugins.yaml the planner loads),
else from the type of that topic. A base station needs no robot names; its own
name (``base_station_name``, default the node's namespace) is the fleet, never a
robot.

The current implementation is specific to the multi-robot PDDL "visit POIs"
domain. The decision is broken out into ``_decide`` so it's clear what would
need to change to support other domains in the future.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import rclpy
import yaml
from dsg_pddl.pddl_utils import lisp_string_to_ast
from nav_msgs.msg import Path
from omniplanner_msgs.msg import ConstrainedPddlGoalMsg, ConstraintFact
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from robot_executor_msgs.msg import ActionDoneMsg, ActionSequenceMsg
from std_msgs.msg import Bool, Int64, String

# Goal-side predicates we treat as "must visit".
# Region goals (visited-region, explored-region and their -by forms) are
# deliberately NOT here: whether a region is done depends on which of its places
# were reached, and this node has no region membership. They evaluate to
# "unknown", which means replan.
_VISIT_PREDS = {"visited-place", "visited-object", "visited-poi"}

# Where a robot is, or what it holds, NOW. Such a fact stops holding as soon as
# the robot moves, picks or places, so "the planner found it holds" is never
# remembered for it: the goal always goes to the planner (unlike a visit, which
# happened once and stays done).
_STATE_PREDS = {"at-poi", "at-place", "at-object", "in-region", "holding", "hand-full"}
# Predicates whose first argument is a robot (besides every ``*-by``).
_ROBOT_FIRST = _STATE_PREDS | {"can-manipulate"}
# The argument slot of the object in each predicate that names one.
_OBJECT_SLOT = {
    "visited-object": 1,
    "visited-object-by": 2,
    "at-object": 2,
    "holding": 2,
    "object-in-place": 1,
    "object-placed-by": 2,
    "inspected-by": 2,
    "safe": 1,
    "suspicious": 1,
}
# Predicates that move or hold their object: two robots' goals conflict over it.
_MOVE_PREDS = {"holding", "object-in-place", "object-placed-by"}
# The constraint that limits a plan to the robots it names.
USE_ROBOTS = "use-robots"


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


def _ast_to_str(ast) -> str:
    if isinstance(ast, str):
        return ast
    return "(" + " ".join(_ast_to_str(a) for a in ast) + ")"


def goal_conjuncts(pddl_goal: str) -> List[str]:
    """The goal's top-level conjuncts, each as PDDL text.

    "(and a b)", "a b" and "(and (and a) b)" all give [a, b]. A goal that does
    not parse is one conjunct, its own text: it is then never judged done.
    """
    text = (pddl_goal or "").strip()
    if not text:
        return []
    ast = _parse_goal(f"(and {text})")
    if ast is None:
        return [text]
    out: List[str] = []

    def flatten(node):
        if not isinstance(node, str) and node and node[0] == "and":
            for sub in node[1:]:
                flatten(sub)
        else:
            out.append(_ast_to_str(node))

    flatten(ast)
    return list(dict.fromkeys(out))


def _named_robots(ast) -> Set[str]:
    """Robots a goal names explicitly: the first argument of a ``*-by``
    predicate (visited-object-by, object-placed-by, explored-region-by...)."""
    if isinstance(ast, str) or not ast:
        return set()
    if ast[0] in ("and", "or", "not"):
        return set().union(*(_named_robots(sub) for sub in ast[1:]))
    if str(ast[0]).endswith("-by") and len(ast) >= 2:
        return {str(ast[1]).lower()}
    return set()


def _atoms(ast):
    """The atomic formulas of a parsed goal, through and/or/not."""
    if isinstance(ast, str) or not ast:
        return
    if ast[0] in ("and", "or", "not"):
        for sub in ast[1:]:
            yield from _atoms(sub)
    else:
        yield ast


def goal_robots(conjuncts) -> Set[str]:
    """Robots the conjuncts name: the robot of a ``*-by``, ``holding``,
    ``at-*`` or ``in-region`` atom. Lowercase."""
    out: Set[str] = set()
    for c in conjuncts:
        for atom in _atoms(_parse_goal(c.lower())):
            head = str(atom[0])
            if (head.endswith("-by") or head in _ROBOT_FIRST) and len(atom) >= 2:
                out.add(str(atom[1]).lower())
    return out


def goal_objects(conjuncts, moved_only: bool = False) -> Set[str]:
    """Objects the conjuncts name, in an object slot (only those they move or
    hold, with moved_only). Lowercase."""
    out: Set[str] = set()
    for c in conjuncts:
        for atom in _atoms(_parse_goal(c.lower())):
            if moved_only and str(atom[0]) not in _MOVE_PREDS:
                continue
            slot = _OBJECT_SLOT.get(str(atom[0]))
            if slot is not None and len(atom) > slot:
                out.add(str(atom[slot]).lower())
    return out


def _states_where_a_robot_is(conjunct: str) -> bool:
    """The conjunct says where a robot is or what it holds now."""
    return any(str(a[0]) in _STATE_PREDS for a in _atoms(_parse_goal(conjunct.lower())))


def use_robots(constraint_facts) -> Set[str]:
    """Robots a use-robots constraint limits the goal to. Lowercase."""
    return {
        str(s).lower()
        for c in constraint_facts or []
        if c.predicate == USE_ROBOTS
        for s in c.symbols
    }


def _conjunct_symbols(ast) -> Set[str]:
    """Every object, place or region a conjunct is about, robots left out."""
    if isinstance(ast, str) or not ast:
        return set()
    if ast[0] in ("and", "or", "not"):
        return set().union(*(_conjunct_symbols(sub) for sub in ast[1:]))
    args = [str(a).lower() for a in ast[1:] if isinstance(a, str)]
    if str(ast[0]).endswith("-by"):
        args = args[1:]
    return set(args)


def _positive(ast) -> bool:
    """No negation anywhere: only such a conjunct can be confirmed done from a
    set of achieved facts (a missing visit is not evidence of no visit)."""
    if isinstance(ast, str) or not ast:
        return True
    if ast[0] == "not":
        return False
    return all(_positive(sub) for sub in ast[1:])


def load_robot_platforms(path: str) -> Dict[str, str]:
    """{robot_name -> platform} from the planner's omniplanner_plugins.yaml.

    The same ``robots`` list ``OmniplannerNodeConfig`` loads; read as plain YAML
    so this node needs none of the planner plugins. ``type: phoenix`` (or
    ``robot_type: husky``) is a Husky; anything else is named by its robot_type.
    """
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    out = {}
    for robot in config.get("robots") or []:
        name = str(robot.get("robot_name") or "")
        if name:
            out[name] = platform_of(robot.get("type"), robot.get("robot_type"))
    return out


HUSKY = "husky"


def platform_of(adaptor_type, robot_type) -> str:
    robot_type = str(robot_type or "").strip().lower()
    if adaptor_type == "phoenix" or robot_type == HUSKY:
        return HUSKY
    return robot_type or "spot"


def plan_topic(robot: str) -> str:
    return f"/{robot}/omniplanner_node/compiled_plan_out"


def _names(robots) -> str:
    robots = sorted(robots)
    if len(robots) <= 2:
        return " and ".join(robots)
    return ", ".join(robots[:-1]) + " and " + robots[-1]


# A robot's plan, as far as this node can tell.
RUNNING, DONE, FAILED, PREEMPTED, STALLED = (
    "running",
    "done",
    "failed",
    "preempted",
    "stalled",
)
# What a conjunct of a new goal is.
CONJ_DONE, CONJ_IN_PROGRESS, CONJ_NEW = "done", "in progress", "new"
# What a new goal does. A robot-scoped goal names some robots (and not the
# whole fleet): only they are planned for, paused or interrupted.
SKIP, WAIT, REPAIR, ADDITIONAL = "skip", "wait", "repair", "additional task"
SCOPED = "robot-scoped"
# An empty Path seen this recently, with the plan that lists the Husky, is
# that plan's Path for it (the two come on different topics).
EMPTY_PATH_WINDOW_S = 5.0
# After a robot's new plan, a pause sent before it may still be on its way
# (over fleetlink a pause is retried for up to its 10 s ttl, on another topic
# than the plan), and would stop the new plan for good. The robot is resumed
# this long after the plan, and again every UNPAUSE_EVERY_S, until the pause
# can no longer arrive.
UNPAUSE_DELAY_S = 0.5
UNPAUSE_EVERY_S = 2.0
# A pause seen on a robot's pause topic this soon after this node sent one is
# that pause, coming back.
OWN_PAUSE_ECHO_S = 1.0

# The constraint an additional task carries: robots the planner leaves alone.
BUSY_ROBOTS = "busy-robots"
# The constraint a goal starting a new mission carries (every robot idle): the
# planner forgets earlier missions' visits, so they do not satisfy this goal.
NEW_MISSION = "new-mission"


@dataclass
class RobotPlan:
    """One robot's share of the running plan."""

    plan_id: str = ""
    # Actions in its sequence, or None when no sequence is seen: a Husky gets a
    # Path and reports no actions, so its progress is unknown.
    total: Optional[int] = None
    done: int = 0
    status: str = RUNNING
    # When it last made progress (got the plan, or finished an action).
    since: float = 0.0
    # (map epoch, map_version) the plan was made on.
    stamp: Tuple[int, int] = (0, 0)
    coverage: Set[str] = field(default_factory=set)
    # Objects its successful actions reached (GAZE, PICK, PLACE).
    reached: Set[str] = field(default_factory=set)
    # The goal conjuncts its plan was given.
    conjuncts: List[str] = field(default_factory=list)

    @property
    def tracked(self) -> bool:
        return self.total is not None

    @property
    def remaining(self) -> Set[str]:
        return self.coverage - self.reached

    def describe(self) -> str:
        if not self.tracked:
            return f"{self.status}, progress unknown (no action reports)"
        return (
            f"{self.status}, {self.done} of {self.total} actions, "
            f"plan {self.plan_id[:8] or '?'}"
        )


@dataclass
class Decision:
    action: str
    reason: str
    # conjunct -> (class, detail)
    classes: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # Conjuncts the planner is sent.
    todo: List[str] = field(default_factory=list)
    busy: List[str] = field(default_factory=list)
    # Every robot was idle: the goal starts a new mission.
    new_mission: bool = False
    # A robot-scoped goal: the robots planned for, those added because the
    # goal needs an object their running plan acts on or they hold ({robot ->
    # objects}), and their conjuncts it supersedes.
    robots: List[str] = field(default_factory=list)
    conflicts: Dict[str, List[str]] = field(default_factory=dict)
    dropped: List[str] = field(default_factory=list)


@dataclass
class Pending:
    """A goal sent to the planner whose plan has not come back yet."""

    kind: str
    msg: object
    conjuncts: List[str]
    stamp: Tuple[int, int]
    at: float
    # A robot-scoped goal: its robots, and the conjuncts of theirs it drops.
    robots: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)


class GoalManager(Node):
    """Decide what a goal still needs; pause, forward, or leave the robots be."""

    def __init__(self, **kwargs):
        super().__init__("goal_manager", **kwargs)

        # The base station's own name: a goal addressed to it is for the fleet,
        # and it is never paused, stopped or listed as a robot.
        self.declare_parameter("base_station_name", "")
        self._base_station = str(
            self.get_parameter("base_station_name").value or ""
        ) or self.get_namespace().strip("/")
        # Which executor each robot runs, from the planner's robots config.
        self.declare_parameter("plugin_config_path", "")
        self._platforms: Dict[str, str] = {}
        config_path = str(self.get_parameter("plugin_config_path").value or "")
        if config_path:
            try:
                self._platforms = load_robot_platforms(config_path)
            except Exception as exc:
                self.get_logger().error(f"Cannot read robots from {config_path}: {exc}")
        if not self._platforms:
            self.get_logger().warning(
                "No robots config (plugin_config_path): each robot's platform is "
                "read from its plan topic's type once the planner advertises it"
            )
        # The frame of the empty Path that stops a Husky.
        self.declare_parameter("plan_frame", "map")
        self._plan_frame = str(self.get_parameter("plan_frame").value or "map")

        # Robots the running plan gave actions to, and those of them paused
        # for a goal that has not been settled yet. A Husky cannot pause, but
        # is held here so a failed goal still stops it.
        self._plan_robots: Set[str] = set()
        self._paused: Set[str] = set()
        self._executor_pubs: Dict[Tuple[str, str], object] = {}
        # A Husky's latest Path: (number of poses, when it came).
        self._paths: Dict[str, Tuple[int, float]] = {}
        # Per robot: what its plan covers and how far it got.
        self._robots: Dict[str, RobotPlan] = {}
        # The latest sequence each robot was sent: (plan_id, actions, when).
        self._sequences: Dict[str, Tuple[str, int, float]] = {}
        self._robot_subs: Dict[str, list] = {}
        # What the fleet has achieved, from finished plans and action reports:
        # {poi -> (map epoch, map_version, robot)} for visits, which outlive
        # map edits; {conjunct -> stamp} for facts the planner said hold, which
        # are only trusted on the map they were judged on.
        self._achieved: Dict[str, Tuple[int, int, str]] = {}
        self._facts: Dict[str, Tuple[int, int]] = {}
        # The running mission's conjuncts, and those of them a lost robot
        # (failed, stalled) gave back: they need a new plan.
        self._mission: List[str] = []
        self._lost: Set[str] = set()
        # {robot -> the conjuncts of the robot-scoped goal it was last given}:
        # a goal naming some robots replaces only theirs, so goals for
        # different robots compose; a goal for the whole team replaces all.
        self._standing: Dict[str, List[str]] = {}
        # {robot -> objects it holds}, from its PICK and PLACE reports.
        self._held: Dict[str, Set[str]] = {}
        # Robots the latest plan was sent to: a coverage message for any other
        # robot is about a plan that left it alone, and is ignored.
        self._last_listed: Optional[Set[str]] = None
        # Pauses: when each robot was last sent one (by anyone), the ones this
        # node sent and has not seen come back, and the robots to resume
        # after their new plan: {robot -> [plan_id, until, next]}.
        self._last_pause: Dict[str, float] = {}
        self._own_pauses: Dict[str, List[float]] = {}
        self._unpause: Dict[str, list] = {}
        # Constraints the pending (or, once it arrives, the running) plan is
        # grounded under, so we can tell when a new goal changes them.
        self._plan_constraints: frozenset = frozenset()
        # Constraints of the plan that is actually running, kept while a
        # forwarded goal is being planned: if that planning fails, the running
        # plan is still the one built under these.
        self._running_constraints: frozenset = frozenset()
        # The goal of the plan being executed, and of the one being planned.
        # A replan request (the scene changed under the plan) re-sends the
        # running mission; a goal the planner rejected never becomes it.
        self._running_goal = None
        self._pending_goal = None
        self._pending: Optional[Pending] = None
        # What a goal that cannot be planned does to the running plan: by
        # default the robots stop and wait for the next goal. The operator
        # asked for something else, so carrying on with the superseded plan
        # is rarely what they want.
        self.declare_parameter("resume_on_plan_failure", False)
        self._resume_on_failure = bool(
            self.get_parameter("resume_on_plan_failure").value
        )
        # Why the planner is replanning on its own (the scene changed, a pick
        # failed), while that replan is outstanding. If it cannot be planned,
        # the running plan is no longer valid either: stop and tell the operator.
        self._replan_reason = None
        # The planner's map. A version that goes down, or a new epoch, is a
        # reload: nothing known about the old map holds, so the epoch counter
        # moves on and every (epoch, version) stamp made before is stale.
        self._map_version: int = 0
        self._map_epoch: int = 0
        self._map_epoch_id: str = ""
        self._running_epoch: int = 0

        # A robot that got a plan and reports no action for this long is
        # stalled: the operator is told, once per plan. Zero disables it.
        self.declare_parameter("progress_timeout_s", 60.0)
        self._progress_timeout_s = float(
            self.get_parameter("progress_timeout_s").value or 0.0
        )
        # How long a pause can take to reach a robot (fleetlink: its ttl, 10 s,
        # plus margin). A robot sent a new plan within this long of a pause is
        # resumed until the pause can no longer arrive. Zero turns it off.
        self.declare_parameter("stale_pause_window_s", 12.0)
        self._stale_pause_window_s = float(
            self.get_parameter("stale_pause_window_s").value or 0.0
        )

        # Subscriptions (private; remap at launch time).
        self._goal_sub = self.create_subscription(
            ConstrainedPddlGoalMsg, "~/commanded_goal", self._commanded_goal_cb, 10
        )
        self._failed_sub = self.create_subscription(
            String, "~/planner_failed", self._planner_failed_cb, 10
        )
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # Latched, so a restarted goal manager still knows who is executing.
        self._robots_sub = self.create_subscription(
            String, "~/plan_robots", self._plan_robots_cb, latched
        )
        self._replan_sub = self.create_subscription(
            String, "~/replan_request", self._replan_request_cb, 10
        )
        self._map_version_sub = self.create_subscription(
            Int64, "~/map_version", self._map_version_cb, latched
        )
        # Optional: an id written at every map load. A change is a reload even
        # if the version did not go down.
        self._map_epoch_sub = self.create_subscription(
            String, "~/map_epoch", self._map_epoch_cb, latched
        )
        self._action_done_sub = self.create_subscription(
            ActionDoneMsg, "/action_done", self._action_done_cb, 50
        )

        # Publishers (private; remap at launch time).
        self._goal_pub = self.create_publisher(
            ConstrainedPddlGoalMsg, "~/planner_goal", 10
        )
        # Messages for the operator: shown in the rviz instruction panel
        # (~/operator_display) and published for any other UI (~/operator_alert).
        self._display_pub = self.create_publisher(String, "~/operator_display", 10)
        self._alert_pub = self.create_publisher(String, "~/operator_alert", 10)
        self._states_pub = self.create_publisher(String, "~/robot_plan_states", latched)

        for robot in sorted(self._platforms):
            self._track_robot(robot)
        period = 5.0
        if self._progress_timeout_s > 0:
            period = min(period, max(self._progress_timeout_s / 4, 0.5))
        self._watchdog = self.create_timer(period, self._watchdog_cb)
        self._unpause_timer = self.create_timer(0.5, self._unpause_tick)

        self.get_logger().info(
            f"goal_manager up on {self._base_station or 'this host'}; "
            "awaiting goals on ~/commanded_goal"
        )

    # ------------- robots: platform and topics -------------

    def _now(self) -> float:
        return time.monotonic()

    @property
    def _stamp(self) -> Tuple[int, int]:
        return (self._map_epoch, self._map_version)

    def _platform(self, robot: str) -> str:
        platform = self._platforms.get(robot)
        if platform is None:
            self._discover_robots()
            platform = self._platforms.get(robot, "spot")
        return platform

    def _discover_robots(self) -> None:
        """Robots the planner advertises a plan topic for, and their platform
        (a Husky's plan is a Path), for robots the config does not name."""
        for name, types in self.get_topic_names_and_types():
            parts = name.strip("/").split("/")
            if parts[1:] != ["omniplanner_node", "compiled_plan_out"]:
                continue
            robot = parts[0]
            if robot in self._platforms or robot == self._base_station:
                continue
            self._platforms[robot] = HUSKY if "nav_msgs/msg/Path" in types else "spot"
            self._track_robot(robot)

    def _track_robot(self, robot: str) -> None:
        """Follow one robot's plans: its action sequence (not a Husky's Path,
        which is another type) and the POIs its plan visits."""
        if robot in self._robot_subs or robot == self._base_station:
            return
        subs = [
            self.create_subscription(
                String,
                f"/{robot}/omniplanner_node/plan_visited_pois",
                lambda msg, r=robot: self._robot_coverage_cb(r, msg),
                10,
            ),
            self.create_subscription(
                String,
                f"/{robot}/omniplanner_node/path_status",
                lambda msg, r=robot: self._path_status_cb(r, msg),
                10,
            ),
        ]
        if self._platform(robot) == HUSKY:
            subs += [
                self.create_subscription(
                    Path,
                    plan_topic(robot),
                    lambda msg, r=robot: self._path_cb(r, msg),
                    1,
                ),
            ]
        else:
            subs += [
                self.create_subscription(
                    ActionSequenceMsg,
                    plan_topic(robot),
                    lambda msg, r=robot: self._sequence_cb(r, msg),
                    10,
                ),
                # Every pause, whoever sends it (this node, the planner on a
                # scene change, the console), so a newer one is never undone.
                self.create_subscription(
                    Bool,
                    f"/{robot}/spot_executor_node/pause",
                    lambda msg, r=robot: self._pause_seen_cb(r, msg),
                    10,
                ),
            ]
        self._robot_subs[robot] = subs

    # ------------- callbacks: per-robot plan state -------------

    def _sequence_cb(self, robot: str, msg: ActionSequenceMsg) -> None:
        """A robot was sent a new action sequence: it replaces its plan."""
        now = self._now()
        self._sequences[robot] = (msg.plan_id, len(msg.actions), now)
        old = self._robots.get(robot)
        if old is not None and old.status == RUNNING and old.plan_id != msg.plan_id:
            self._log_robot(robot, old, f"preempted by plan {msg.plan_id[:8]}")
        pending = self._pending
        entry = RobotPlan(
            plan_id=msg.plan_id,
            total=len(msg.actions),
            since=now,
            stamp=pending.stamp if pending else self._stamp,
        )
        self._robots[robot] = entry
        if entry.total == 0:
            self._finish(robot, "its new plan is empty")
            return
        self._log_robot(robot, entry, "got a new plan")
        self._schedule_unpause(robot, msg.plan_id, now)

    # ------------- pauses that must not outlive the plan they were for -------------

    def _note_pause(self, robot: str, own: bool = False) -> None:
        """A pause was sent to the robot: a resume scheduled for an earlier
        plan must not undo it."""
        now = self._now()
        self._last_pause[robot] = now
        if own:
            self._own_pauses.setdefault(robot, []).append(now)
        if self._unpause.pop(robot, None) is not None:
            self.get_logger().info(f"{robot} was paused again: not resuming it")

    def _pause_seen_cb(self, robot: str, msg: Bool) -> None:
        if not msg.data:
            return
        now = self._now()
        own = [
            t for t in self._own_pauses.get(robot, []) if now - t <= OWN_PAUSE_ECHO_S
        ]
        if own:
            own.pop(0)  # the pause this node sent, coming back
            self._own_pauses[robot] = own
            return
        self._own_pauses.pop(robot, None)
        self._note_pause(robot)

    def _schedule_unpause(self, robot: str, plan_id: str, now: float) -> None:
        """A pause sent shortly before this plan may reach the robot after it,
        and would stop the new plan for good (the executor clears its pause
        only when a plan arrives). Resume the robot until it cannot arrive."""
        window = self._stale_pause_window_s
        last = self._last_pause.get(robot)
        if window <= 0 or last is None or now - last > window:
            return
        self._unpause[robot] = [plan_id, last + window, now + UNPAUSE_DELAY_S]
        self.get_logger().info(
            f"{robot}: a pause sent {now - last:.1f} s before plan {plan_id[:8]} "
            f"may still arrive; resuming it until {window:.0f} s after that pause"
        )

    def _unpause_tick(self) -> None:
        now = self._now()
        for robot, (plan_id, until, at) in list(self._unpause.items()):
            entry = self._robots.get(robot)
            if (
                entry is None
                or entry.plan_id != plan_id
                or entry.status not in (RUNNING, STALLED)
            ):
                self._unpause.pop(robot, None)
                continue
            if now < at:
                continue
            self._executor_cmd(robot, "resume")
            if now >= until:
                self._unpause.pop(robot, None)
            else:
                self._unpause[robot][2] = min(now + UNPAUSE_EVERY_S, until)

    def _robot_coverage_cb(self, robot: str, msg: String) -> None:
        """The POIs a robot's new plan visits, and so which conjuncts it has."""
        try:
            pois = {str(p).lower() for p in json.loads(msg.data)}
        except Exception as exc:
            self.get_logger().warning(f"Bad plan_visited_pois for {robot}: {exc}")
            return
        entry = self._robots.get(robot)
        if (
            entry is not None
            and self._last_listed is not None
            and robot not in self._last_listed
        ):
            # The latest plan left this robot alone: its own plan stands.
            return
        if entry is None:
            if not pois:
                return
            # No sequence seen (a Husky, or a robot not yet followed).
            entry = RobotPlan(since=self._now(), stamp=self._stamp)
            self._robots[robot] = entry
        entry.coverage = pois
        goal = self._pending.conjuncts if self._pending else self._mission
        entry.conjuncts = [
            c for c in goal if _eval_goal_against_visited(_parse_goal(c.lower()), pois)
        ]
        self._log_robot(robot, entry, f"plan covers {sorted(pois)}")

    def _action_done_cb(self, msg: ActionDoneMsg) -> None:
        self._note_held(msg)
        entry = self._robots.get(msg.robot_name)
        if entry is None or not entry.tracked or msg.plan_id != entry.plan_id:
            return
        if entry.status not in (RUNNING, STALLED):
            return
        entry.done += 1
        entry.since = self._now()
        obj = (msg.object_id or "").lower()
        action = msg.action_type.upper()
        if entry.status == STALLED:
            entry.status = RUNNING
            self._log_robot(msg.robot_name, entry, "is making progress again")
        if not msg.success:
            self._lose(msg.robot_name, FAILED, f"{action} {obj}".strip() + " failed")
            return
        if obj and action in ("GAZE", "PICK", "PLACE"):
            entry.reached.add(obj)
        if entry.done >= entry.total:
            self._finish(msg.robot_name, "finished every action")
        else:
            self._publish_states()

    def _note_held(self, msg: ActionDoneMsg) -> None:
        obj = (msg.object_id or "").lower()
        if not msg.success or not obj:
            return
        action = msg.action_type.upper()
        if action == "PICK":
            for held in self._held.values():
                held.discard(obj)
            self._held.setdefault(msg.robot_name, set()).add(obj)
        elif action == "PLACE":
            self._held.get(msg.robot_name, set()).discard(obj)

    def _path_cb(self, robot: str, msg: Path) -> None:
        """A Husky's new Path. An empty one stops it: it is free, not busy."""
        self._paths[robot] = (len(msg.poses), self._now())
        entry = self._robots.get(robot)
        if not msg.poses and entry is not None and not entry.tracked:
            self._drop(robot, "was sent an empty Path: it is idle")

    def _path_status_cb(self, robot: str, msg: String) -> None:
        """A robot's progress, as the planner sees its pose: a Husky's along
        its Path; a Spot's movement, which counts as progress during an action
        (one long follow reports nothing until it ends)."""
        try:
            data = json.loads(msg.data)
            status = data["status"]
        except Exception as exc:
            self.get_logger().warning(f"Bad path_status for {robot}: {exc}")
            return
        entry = self._robots.get(robot)
        if entry is None or entry.status not in (RUNNING, STALLED):
            return
        if entry.tracked:
            if status == "progress" and data.get("plan_id") == entry.plan_id:
                entry.since = self._now()
                if entry.status == STALLED:
                    entry.status = RUNNING
                    self._log_robot(robot, entry, "is making progress again")
            return
        if status == "stalled":
            self._lose(robot, STALLED, "has not moved along its path")
            return
        entry.since = self._now()
        if entry.status == STALLED:
            entry.status = RUNNING
            self._log_robot(robot, entry, "is making progress again")
        if status == "done":
            self._finish(robot, "reached the end of its path")

    def _finish(self, robot: str, why: str) -> None:
        """Every action of the robot's plan succeeded: what it covered is
        achieved, and its entry is cleared. Other robots' entries stay."""
        entry = self._robots.pop(robot)
        entry.status = DONE
        self._standing.pop(robot, None)
        for poi in entry.coverage | entry.reached:
            self._achieved[poi] = (*self._stamp, robot)
        self._plan_robots.discard(robot)
        done = [c for c in entry.conjuncts if self._is_done(c)]
        self._log_robot(
            robot,
            entry,
            f"{why}: achieved {sorted(entry.coverage | entry.reached)}"
            + (f"; done {done}" if done else ""),
        )

    def _lose(self, robot: str, status: str, why: str) -> None:
        """The robot will not finish its plan: only its conjuncts are open again."""
        entry = self._robots.get(robot)
        if entry is None:
            return
        entry.status = status
        for poi in entry.reached:
            self._achieved[poi] = (*self._stamp, robot)
        open_again = [c for c in entry.conjuncts if not self._is_done(c)]
        self._lost |= set(open_again)
        if status != STALLED:
            # A stalled robot keeps its entry, out of the running set, in case
            # it reports progress after all.
            self._robots.pop(robot)
            self._plan_robots.discard(robot)
        self._log_robot(robot, entry, why)
        if status in (FAILED, STALLED):
            part = (
                f" Its part of the goal, {' '.join(open_again)}, is open again and "
                "is replanned with the next goal or plan repair."
                if open_again
                else ""
            )
            self._notify_operator(f"{robot}: {why}.{part}")

    def _watchdog_cb(self) -> None:
        if not self._platforms or len(self._robot_subs) < len(self._platforms):
            self._discover_robots()
        self._check_progress()

    def _check_progress(self) -> None:
        """A robot that got a plan and reports no action for it (P11)."""
        if self._progress_timeout_s <= 0:
            return
        now = self._now()
        for robot, entry in sorted(self._robots.items()):
            if (
                entry.status != RUNNING
                or not entry.tracked
                or entry.done
                or robot in self._paused
            ):
                continue
            idle = now - entry.since
            if idle >= self._progress_timeout_s:
                self._lose(
                    robot,
                    STALLED,
                    f"no progress on plan {entry.plan_id[:8]} for {idle:.0f} s "
                    f"(0 of {entry.total} actions done). Check the robot, or "
                    "re-send the goal",
                )

    def _log_robot(self, robot: str, entry: RobotPlan, what: str) -> None:
        self.get_logger().info(f"{robot} {what} [{entry.describe()}]")
        self._publish_states()

    def _robot_states(self) -> dict:
        return {
            robot: {
                "status": e.status,
                "plan_id": e.plan_id,
                "done": e.done,
                "total": e.total,
                "conjuncts": e.conjuncts,
                "remaining": sorted(e.remaining),
            }
            for robot, e in sorted(self._robots.items())
        }

    def _publish_states(self) -> None:
        self._states_pub.publish(String(data=json.dumps(self._robot_states())))

    def _running(self) -> Dict[str, RobotPlan]:
        """Robots executing their plan: a Spot reporting its actions, or a
        Husky that has not reached the end of its Path yet."""
        return {r: e for r, e in self._robots.items() if e.status == RUNNING}

    def robots_mid_task(self) -> Dict[str, RobotPlan]:
        """Robots in the middle of a plan: those a new plan would replace."""
        return {r: e for r, e in self._robots.items() if e.status == RUNNING}

    # ------------- callbacks: plans and the map -------------

    def _plan_robots_cb(self, msg: String) -> None:
        """A new plan went out: it now decides who is executing.

        A robot in it was preempted by its new action sequence, which also
        clears its pause. A paused robot left out of it would otherwise sit
        paused for ever, or, once resumed, carry on with the superseded plan,
        so it is stopped. A robot that was not paused (busy with its part while
        an additional task was planned) carries on.
        """
        try:
            listed = set(json.loads(msg.data))
        except Exception as exc:
            self.get_logger().warning(f"Bad plan_robots payload: {exc}")
            return
        robots = listed - {self._base_station}
        if robots != listed:
            self.get_logger().warning(
                f"plan_robots lists {self._base_station}, the base station; "
                "it is not a robot and is left out"
            )
        left_out = self._paused - robots
        for robot in sorted(left_out):
            self.get_logger().info(f"{robot} is not in the new plan; stopping it")
            self._executor_cmd(robot, "stop")
            self._drop(robot, "stopped: not in the new plan")
        self._paused = set()
        self._last_listed = set(robots)
        self._plan_robots = robots | (self._plan_robots - left_out)
        pending = self._pending
        for robot in sorted(robots):
            self._track_robot(robot)
            self._warm_executor(robot)
            if self._platform(robot) == HUSKY:
                poses, when = self._paths.get(robot, (None, 0.0))
                if poses == 0 and self._now() - when < EMPTY_PATH_WINDOW_S:
                    # Its new Path (just before this list) is empty: it is idle.
                    continue
                entry = RobotPlan(
                    since=self._now(), stamp=pending.stamp if pending else self._stamp
                )
                self._robots[robot] = entry
                self._log_robot(robot, entry, "got a new plan (a Path)")
        if pending is not None:
            self._commit_plan(pending, robots)

    def _commit_plan(self, pending: Pending, robots: Set[str]) -> None:
        """The plan for the pending goal arrived: it is now the running one."""
        self._pending = None
        self._replan_reason = None  # the replan succeeded
        if pending.kind == ADDITIONAL:
            self._mission += [c for c in pending.conjuncts if c not in self._mission]
        elif pending.kind == SCOPED:
            self._commit_scoped(pending)
        else:
            self._mission = list(pending.conjuncts)
            self._lost = set()
            self._standing = {}  # a goal for the whole team replaces every one
            self._running_goal = pending.msg
            self._running_constraints = self._plan_constraints
            self._running_epoch = pending.stamp[0]
        # Every robot was sent an empty sequence: the goal already holds (P8).
        fresh = [self._sequences.get(r) for r in robots]
        if (
            robots
            and all(s is not None and s[2] >= pending.at for s in fresh)
            and all(s[1] == 0 for s in fresh)
        ):
            self._goal_holds(pending.conjuncts)
            self._notify_operator(
                f"{self._describe(pending.msg)} already holds: the plan has "
                "nothing to do."
            )

    def _commit_scoped(self, pending: Pending) -> None:
        """A robot-scoped plan: its robots' standing goals are replaced, every
        other robot's is kept, and the mission is what they all add up to."""
        scope = set(pending.robots)
        old = {c for r in scope for c in self._standing.get(r, [])}
        others = {c for r, cs in self._standing.items() if r not in scope for c in cs}
        gone = (old | set(pending.dropped)) - others - set(pending.conjuncts)
        self._mission = [c for c in self._mission if c not in gone]
        self._mission += [c for c in pending.conjuncts if c not in self._mission]
        self._lost -= gone
        for robot in scope:
            self._standing[robot] = list(pending.conjuncts)
        self._running_constraints = self._plan_constraints
        self._running_epoch = pending.stamp[0]
        if self._running_goal is None:
            self._running_goal = pending.msg

    def _goal_holds(self, conjuncts) -> None:
        """The planner says these hold on the current map. Where a robot is
        or what it holds is not remembered: it changes as soon as the robot
        moves, on the same map."""
        for c in conjuncts:
            if _states_where_a_robot_is(c):
                continue
            self._facts[c.lower()] = self._stamp
        self._lost -= set(conjuncts)

    def _drop(self, robot: str, why: str) -> None:
        entry = self._robots.pop(robot, None)
        if entry is not None:
            entry.status = PREEMPTED
            self._log_robot(robot, entry, why)

    def _map_version_cb(self, msg: Int64) -> None:
        version = int(msg.data)
        if version < self._map_version:
            # The version only goes up on one map, so this is a reload: the same
            # number may come back on a different map.
            self._map_reloaded(
                f"map_version went down, {self._map_version} -> {version}"
            )
        self._map_version = version

    def _map_epoch_cb(self, msg: String) -> None:
        if self._map_epoch_id and msg.data != self._map_epoch_id:
            self._map_reloaded(f"map epoch {self._map_epoch_id} -> {msg.data}")
        self._map_epoch_id = msg.data

    def _map_reloaded(self, why: str) -> None:
        """Nothing known about the old map holds: forget every plan's coverage
        and every achieved fact. The mission stays the operator's."""
        self._map_epoch += 1
        robots = sorted(self._robots)
        self._robots.clear()
        self._achieved.clear()
        self._facts.clear()
        self._lost = set()
        self._publish_states()
        self._notify_operator(
            f"The map was reloaded ({why}). Forgetting what was done and what "
            f"{_names(robots) or 'the robots'} were doing; the next goal is "
            "planned from the new map."
        )

    def _planner_failed_cb(self, msg: String) -> None:
        """The forwarded goal produced no plan.

        The executor was paused when the goal arrived and only a new plan would
        have released it, so without this the robot waits for ever -- possibly
        holding an object. The goal is dropped; the operator has to restate it.

        The robots stop and wait for the next goal: the operator replaced the
        goal, so the running plan is no longer what they want. With
        resume_on_plan_failure they carry on with the current plan instead,
        but only if it was made under the same constraints -- if the failed
        goal changed them (e.g. "avoid o14"), the current plan may break them
        -- and on the same map: after a reload the old plan is never resumed.

        A planner that finds the goal already satisfied says so here: the
        operator is told, and the same goal is not planned again on this map.
        """
        why = msg.data.split(": ", 1)[-1]  # drop the plugin name
        holds = "already satisfied" in why.lower()
        goal = self._pending_goal
        pending, self._pending = self._pending, None
        target = self._describe(goal)
        if holds and pending is not None:
            self._goal_holds(pending.conjuncts)
        elif holds and goal is not None:
            self._goal_holds(goal_conjuncts(goal.goal.pddl_goal))
        same_constraints = goal is None or (
            constraint_signature(goal.constraints) == self._running_constraints
        )
        self._plan_constraints = self._running_constraints
        if pending is not None and pending.kind == ADDITIONAL:
            # Nobody was paused for it: the running plans go on.
            lead = (
                f"{target} already holds; nothing to do"
                if holds
                else f"Could not plan the additional task {target} ({why})"
            )
            self._notify_operator(f"{lead}. The running plans continue.")
            return
        if self._replan_reason is not None:
            reason, self._replan_reason = self._replan_reason, None
            stopped = self._stop_summary(*self._stop_executor())
            if holds:
                self._notify_operator(
                    f"Replanned after {reason}: {target} already holds, nothing "
                    f"is left to do. {stopped}."
                )
                return
            # The world changed under the plan and nothing fits the goal any
            # more. Resuming would drive the robots on to what is no longer
            # there, so they stop whatever resume_on_plan_failure says.
            self._notify_operator(
                f"Cannot repair the plan: {reason}. "
                f"No plan achieves {target} ({why}). "
                f"{stopped}. Please give a new instruction."
            )
            return
        lead = (
            f"{target} already holds; nothing to do"
            if holds
            else f"Could not plan {target} ({why})"
        )
        same_map = getattr(self, "_running_epoch", 0) == getattr(self, "_map_epoch", 0)
        if same_constraints and same_map and getattr(self, "_resume_on_failure", False):
            self._resume_executor()
            self._notify_operator(f"{lead}. Continuing the current plan.")
            return
        stopped = self._stop_summary(*self._stop_executor())
        if not same_constraints:
            self._notify_operator(
                f"{lead}. {stopped}: the current plan was made without these "
                "constraints. Waiting for a new goal."
            )
        elif not same_map and getattr(self, "_resume_on_failure", False):
            self._notify_operator(
                f"{lead}. {stopped}: the map was reloaded since the current plan "
                "was made. Waiting for a new goal."
            )
        else:
            self._notify_operator(f"{lead}. {stopped}; waiting for a new goal.")

    def _replan_request_cb(self, msg: String) -> None:
        """The plan no longer fits the world (e.g. the scene changed): replan.

        Re-sends what is left of the running mission, as a repair: the planner
        may give any robot a new plan. It plans from the current map and
        mission state, so only what is left is planned, around the change.
        """
        goal = self._running_goal or self._pending_goal
        if goal is None:
            self.get_logger().info(f"replan requested ({msg.data}), but no goal yet")
            return
        mission = self._mission or goal_conjuncts(goal.goal.pddl_goal)
        todo = [c for c in mission if not self._is_done(c)]
        if self._standing and any(c.predicate == USE_ROBOTS for c in goal.constraints):
            # The mission is several robot-scoped goals: one goal's use-robots
            # would keep the others' robots out of their own conjuncts.
            stripped = ConstrainedPddlGoalMsg()
            stripped.goal.robot_id = goal.goal.robot_id
            stripped.goal.pddl_goal = goal.goal.pddl_goal
            stripped.constraints = [
                c for c in goal.constraints if c.predicate != USE_ROBOTS
            ]
            goal = stripped
        if not todo:
            self.get_logger().info(
                f"replan requested ({msg.data}), but every conjunct of the "
                f"mission is achieved: {mission}"
            )
            return
        self._pause_executor()
        self._replan_reason = msg.data
        out = self._goal_msg(goal, todo, [])
        self._pending = Pending(REPAIR, out, todo, self._stamp, self._now())
        self._goal_pub.publish(out)
        self.get_logger().info(
            f"replanning: {msg.data}. goal={out.goal.pddl_goal!r}; forwarded to planner"
        )

    def _notify_operator(self, text: str) -> None:
        self.get_logger().warning(f"operator: {text}")
        self._display_pub.publish(String(data=text))
        self._alert_pub.publish(String(data=text))

    def _commanded_goal_cb(self, msg: ConstrainedPddlGoalMsg) -> None:
        # Flow: decide → (pause →) forward, or leave the robots be → log.
        self._replan_reason = None  # a new instruction supersedes it
        new_mission = self._starts_new_mission(msg)
        if new_mission:
            self._forget_finished_mission()
        decision = self._decide(msg)
        decision.new_mission = new_mission and decision.action in (REPAIR, SCOPED)
        self._log_decision(msg, decision)
        if decision.action in (SKIP, WAIT):
            self._notify_operator(f"{self._describe(msg)}: {decision.reason}.")
            return
        if decision.conflicts:
            self._notify_operator(self._conflict_text(msg, decision))
        self._forward_for_replanning(msg, decision)

    def _conflict_text(self, msg, d: Decision) -> str:
        parts = [
            f"{r}'s running plan acts on or holds {', '.join(objs)}"
            for r, objs in sorted(d.conflicts.items())
        ]
        text = (
            f"{self._describe(msg)} needs what {'; '.join(parts)}: replanning "
            f"{_names(d.conflicts)} with it"
        )
        if d.dropped:
            text += f"; dropping {' '.join(d.dropped)}, which the new goal replaces"
        kept = [c for c in d.todo if c not in goal_conjuncts(msg.goal.pddl_goal)]
        if kept:
            text += f"; keeping {' '.join(kept)}"
        return text + "."

    def _starts_new_mission(self, msg: ConstrainedPddlGoalMsg) -> bool:
        """Every robot is idle and nothing is being planned: the goal starts
        a new mission, unless it asks again for work a lost robot left."""
        if self._pending is not None or self._paused:
            return False
        if any(e.status in (RUNNING, STALLED) for e in self._robots.values()):
            return False
        return not (self._lost & set(goal_conjuncts(msg.goal.pddl_goal)))

    def _forget_finished_mission(self) -> None:
        """P16: what earlier, finished goals achieved does not count for a
        new mission, so a goal sent again after it finished is planned again.
        Facts the planner found already holding (a placement) are about the
        map, not the mission, and are kept until the map changes."""
        if self._achieved or self._lost:
            self.get_logger().info(
                "every robot is idle: the goal starts a new mission; forgetting "
                f"what earlier goals achieved ({len(self._achieved)} POIs)"
            )
        self._achieved.clear()
        self._lost = set()
        self._mission = []
        self._standing = {}

    # ------------- decision (domain-specific) -------------

    def _visited(self) -> Set[str]:
        return {p for p, a in self._achieved.items() if a[0] == self._map_epoch}

    def _done_detail(self, conjunct: str) -> str:
        c = conjunct.lower()
        if self._facts.get(c) == self._stamp:
            return "the planner found it holds"
        who = [
            f"{p} by {a[2]} at map_version {a[1]}"
            for p, a in sorted(self._achieved.items())
            if p in _conjunct_symbols(_parse_goal(c))
        ]
        return "visited " + ", ".join(who) if who else "achieved"

    def _is_done(self, conjunct: str) -> bool:
        """An achieved fact: a visit the fleet made (on any version of this
        map), or a fact the planner said holds on this very map."""
        c = conjunct.lower()
        if self._facts.get(c) == self._stamp:
            return True
        ast = _parse_goal(c)
        if ast is None or not _positive(ast):
            return False
        if (
            isinstance(ast, (list, tuple))
            and len(ast) == 3
            and str(ast[0]).endswith("-by")
            and str(ast[0])[: -len("-by")] in _VISIT_PREDS
        ):
            # a visit by a named robot: done once that robot reached the POI
            a = self._achieved.get(str(ast[2]))
            return a is not None and a[0] == self._map_epoch and a[2] == str(ast[1])
        visited = self._visited()
        return _eval_goal_against_visited(ast, visited) is True

    def _classify(self, conjuncts, forbidden: Set[str]) -> Dict[str, Tuple[str, str]]:
        """Each conjunct: done (achieved), in progress (a running robot's
        remaining plan reaches it, on the current map), or new."""
        running = {r: e for r, e in self._running().items() if e.stamp == self._stamp}
        stale = sorted(r for r, e in self._running().items() if e.stamp != self._stamp)
        visited = self._visited()
        out = {}
        for c in conjuncts:
            ast = _parse_goal(c.lower())
            if self._is_done(c):
                out[c] = (CONJ_DONE, self._done_detail(c))
                continue
            if ast is None or not _positive(ast):
                out[c] = (CONJ_NEW, "not judged from plans")
                continue
            by = sorted(
                r
                for r, e in running.items()
                if _eval_goal_against_visited(ast, visited | e.remaining) is True
            )
            if not by:
                why = "no running plan reaches it"
                if stale:
                    why += f" ({_names(stale)} planned on an older map)"
                out[c] = (CONJ_NEW, why)
            elif set().union(*(running[r].remaining for r in by)) & forbidden:
                out[c] = (CONJ_NEW, "the plan reaching it visits a forbidden POI")
            else:
                out[c] = (CONJ_IN_PROGRESS, _names(by))
        return out

    def _decide(self, msg: ConstrainedPddlGoalMsg) -> Decision:
        """What the planner still has to do for this goal.

        Skip only when every conjunct is an achieved fact -- never because a
        plan merely covers it. Leave the robots be when the rest is in progress.
        Otherwise forward, as a repair (the default: the planner may interrupt
        any robot) or, when the goal does not touch the running mission while
        robots are busy, as an additional task for the free robots.
        """
        conjuncts = goal_conjuncts(msg.goal.pddl_goal)
        forbidden = extract_forbidden_pois(msg.constraints)
        classes = self._classify(conjuncts, forbidden)
        todo = [c for c in conjuncts if classes[c][0] != CONJ_DONE]
        new = [c for c in todo if classes[c][0] == CONJ_NEW]
        lost = sorted(self._lost & set(todo))
        running = self._running()
        changed = bool(running) and (
            constraint_signature(msg.constraints) != self._running_constraints
        )
        d = Decision(REPAIR, "", classes, todo)
        if conjuncts and not todo:
            d.action = SKIP
            d.reason = "already done, nothing to plan: " + "; ".join(
                f"{c} ({classes[c][1]})" for c in conjuncts
            )
            return d
        if todo and not new and not lost and not changed:
            in_progress = sorted({classes[c][1] for c in todo})
            d.action = WAIT
            d.reason = f"already in progress ({', '.join(in_progress)}); not replanning"
            return d
        # What is left to do decides: an achieved robot-free task does not
        # make the rest the team's.
        involved = self._goal_scope(msg, todo)
        if involved:
            return self._decide_scoped(d, involved)
        mission_open = [c for c in self._mission if not self._is_done(c)]
        mission_symbols = set().union(
            *(_conjunct_symbols(_parse_goal(c.lower())) for c in mission_open)
        )
        goal_symbols = set().union(
            *(_conjunct_symbols(_parse_goal(c.lower())) for c in todo)
        )
        if lost:
            d.reason = f"{' '.join(lost)} lost its robot"
        elif changed:
            d.reason = "the constraints changed"
        elif not running:
            d.reason = "no robot is busy"
        elif self._paused or self._pending is not None:
            d.reason = "a goal is still being planned"
        elif not mission_open:
            d.reason = "cannot tell what the running robots are doing"
        elif msg.goal.robot_id in running:
            # Addressed to a busy robot: it is that robot's new instruction.
            d.reason = f"it is addressed to {msg.goal.robot_id}, which is busy"
        elif goal_symbols & mission_symbols or set(todo) & set(mission_open):
            d.reason = (
                "it revises the running mission (shares "
                f"{sorted(goal_symbols & mission_symbols)})"
            )
        else:
            named = set().union(*(_named_robots(_parse_goal(c.lower())) for c in todo))
            d.action = ADDITIONAL
            d.busy = sorted(set(running) - named - {self._base_station})
            d.reason = "it does not touch the running mission; busy robots: " + (
                ", ".join(d.busy) or "none"
            )
        return d

    def _fleet(self) -> Set[str]:
        """Every robot this node knows of."""
        robots = set(self._platforms) | set(self._robots) | self._plan_robots
        return robots - {self._base_station, ""}

    def _goal_scope(self, msg, conjuncts) -> Set[str]:
        """The robots a goal involves, if not the whole fleet: its use-robots,
        else the robots its conjuncts name (-by, holding, at-*, in-region)
        when every one of them names a robot. Empty for a goal with a task
        naming no robot (and no use-robots), which is the whole team's: its
        robot-free tasks are for anyone, not for the robots named elsewhere."""
        own = {self._base_station}
        used = use_robots(msg.constraints) - own
        if not used and any(not (goal_robots([c]) - own) for c in conjuncts):
            return set()
        robots = (goal_robots(conjuncts) - own) | used
        if not robots or not (self._fleet() - robots):
            return set()
        return robots

    def _robot_objects(self, robot: str) -> Set[str]:
        """Objects a robot holds or its running plan still acts on."""
        out = set(self._held.get(robot, set()))
        entry = self._robots.get(robot)
        if entry is not None and entry.status in (RUNNING, STALLED):
            out |= entry.remaining
            out |= goal_objects(entry.conjuncts)
            out |= goal_objects(self._standing.get(robot, []))
        return out

    def _robot_conjuncts(self, robot: str) -> List[str]:
        """The open conjuncts a robot is working on."""
        entry = self._robots.get(robot)
        if robot in self._standing:
            mine = self._standing[robot]
        elif entry is not None and entry.status in (RUNNING, STALLED):
            mine = entry.conjuncts
        else:
            mine = []
        return [c for c in mine if not self._is_done(c)]

    def _decide_scoped(self, d: Decision, involved: Set[str]) -> Decision:
        """Plan only for the robots the goal involves; every other robot's
        plan, standing goal and held objects are left alone -- unless the goal
        moves or holds an object another robot holds or its running plan acts
        on: that robot is replanned too, keeping the rest of its work. (Two
        robots visiting the same object is no conflict.)"""
        wanted = goal_objects(d.todo, moved_only=True)
        for robot in sorted(self._fleet() - involved):
            objs = self._robot_objects(robot) & wanted
            if objs:
                d.conflicts[robot] = sorted(objs)
        replanned = involved | set(d.conflicts)
        for robot, objs in sorted(d.conflicts.items()):
            for c in self._robot_conjuncts(robot):
                if c in d.todo or c in d.dropped:
                    continue
                if goal_robots([c]) - replanned:
                    # another robot's part of a shared goal (that robot is left
                    # alone, so the planner has no such robot): not carried over
                    continue
                if goal_objects([c], moved_only=True) & set(objs):
                    d.dropped.append(c)
                else:
                    d.todo.append(c)
        d.action = SCOPED
        d.robots = sorted(involved | set(d.conflicts))
        others = sorted(self._fleet() - set(d.robots))
        d.reason = f"it involves only {_names(involved)}; " + (
            f"leaving {_names(others)} alone" if others else "nobody else to leave"
        )
        if d.conflicts:
            d.reason += (
                f"; {_names(d.conflicts)} replanned too: the goal needs "
                f"{sorted({o for objs in d.conflicts.values() for o in objs})}"
            )
        return d

    # ------------- side effects -------------

    def _warm_executor(self, robot: str) -> None:
        """Create a robot's executor publishers ahead of need, so they are
        matched by the time a pause or stop is sent."""
        verbs = (
            ("stop",) if self._platform(robot) == HUSKY else ("pause", "resume", "stop")
        )
        for verb in verbs:
            self._executor_pub(robot, verb)

    def _executor_pub(self, robot: str, verb: str):
        husky = self._platform(robot) == HUSKY
        key = (robot, "path" if husky else verb)
        pub = self._executor_pubs.get(key)
        if pub is None:
            if husky:
                pub = self.create_publisher(Path, plan_topic(robot), 1)
            else:
                pub = self.create_publisher(
                    Bool, f"/{robot}/spot_executor_node/{verb}", 10
                )
            self._executor_pubs[key] = pub
        return pub

    def _executor_cmd(self, robot: str, verb: str) -> bool:
        """Pause, resume or stop one robot. True if it was reached: something
        listens on its topic. A Husky's phoenix executor has no pause; it is
        stopped by an empty Path, which replaces the one it is driving."""
        if robot == self._base_station:
            return False
        if self._platform(robot) == HUSKY:
            if verb != "stop":
                return False
            pub = self._executor_pub(robot, verb)
            path = Path()
            path.header.frame_id = self._plan_frame
            path.header.stamp = self.get_clock().now().to_msg()
            pub.publish(path)
        else:
            pub = self._executor_pub(robot, verb)
            pub.publish(Bool(data=True))
        return pub.get_subscription_count() > 0

    def _pause_executor(self, only=None) -> None:
        """Pause every robot of the running plan (or those of them in only);
        an idle one ignores it."""
        robots = self._plan_robots - {self._base_station}
        if only is not None:
            robots &= set(only)
        robots = sorted(robots)
        huskies = [r for r in robots if self._platform(r) == HUSKY]
        unreached = []
        for r in robots:
            if r in huskies:
                continue
            self._note_pause(r, own=True)
            if not self._executor_cmd(r, "pause"):
                unreached.append(r)
        if huskies:
            self.get_logger().warning(
                f"Pause is not supported on a Husky: {_names(huskies)} keep driving "
                "until the new plan replaces their path"
            )
        if unreached:
            self.get_logger().warning(f"Could not reach {_names(unreached)} to pause")
        self._paused |= set(robots)

    def _resume_executor(self) -> None:
        now = self._now()
        for robot in sorted(self._paused):
            if self._platform(robot) != HUSKY:
                self._executor_cmd(robot, "resume")
            if robot in self._robots:
                self._robots[robot].since = now  # paused time is not a stall
        self._paused = set()

    def _stop_executor(self):
        """Stop every paused robot. Returns (stopped, unreached)."""
        stopped, unreached = [], []
        for robot in sorted(self._paused):
            (stopped if self._executor_cmd(robot, "stop") else unreached).append(robot)
            self._drop(robot, "stopped")
        self._paused = set()
        return stopped, unreached

    @staticmethod
    def _stop_summary(stopped, unreached) -> str:
        """Names only the robots a stop reached; never claims the others."""
        parts = []
        if stopped:
            parts.append(f"Stopped {_names(stopped)}")
        if unreached:
            parts.append(
                f"could not reach {_names(unreached)} to stop "
                "(still driving the old plan, unless stopped by hand)"
            )
        text = "; ".join(parts) or "No robot was running"
        return text[0].upper() + text[1:]

    def _goal_msg(
        self, msg, todo, busy, new_mission=False, robots=None
    ) -> ConstrainedPddlGoalMsg:
        """The goal as sent: only what is not done, plus the busy robots, or
        the mark that it starts a new mission, or (robot-scoped) use-robots
        naming the only robots to plan for."""
        if (
            todo == goal_conjuncts(msg.goal.pddl_goal)
            and not busy
            and not new_mission
            and not robots
        ):
            return msg
        out = ConstrainedPddlGoalMsg()
        out.goal.robot_id = msg.goal.robot_id
        out.goal.pddl_goal = todo[0] if len(todo) == 1 else f"(and {' '.join(todo)})"
        out.constraints = list(msg.constraints)
        if robots:
            out.constraints = [c for c in out.constraints if c.predicate != USE_ROBOTS]
            out.constraints.append(
                ConstraintFact(predicate=USE_ROBOTS, symbols=list(robots))
            )
        if busy:
            out.constraints.append(ConstraintFact(predicate=BUSY_ROBOTS, symbols=busy))
        if new_mission:
            out.constraints.append(ConstraintFact(predicate=NEW_MISSION, symbols=[]))
        return out

    def _forward_for_replanning(self, msg: ConstrainedPddlGoalMsg, d: Decision) -> None:
        if d.action == REPAIR:
            mid = self.robots_mid_task()
            if mid:
                # P14: whose work the new plan may take over.
                self.get_logger().info(
                    "the new goal is a repair: the planner may replace the plans of "
                    + "; ".join(
                        f"{r} ({e.describe()}; {' '.join(e.conjuncts) or 'no conjuncts'})"
                        for r, e in sorted(mid.items())
                    )
                )
            self._pause_executor()
        elif d.action == SCOPED:
            mid = {r: e for r, e in self.robots_mid_task().items() if r in d.robots}
            if mid:
                self.get_logger().info(
                    "the new goal is robot-scoped: the planner may replace the "
                    "plans of "
                    + "; ".join(
                        f"{r} ({e.describe()}; {' '.join(e.conjuncts) or 'no conjuncts'})"
                        for r, e in sorted(mid.items())
                    )
                )
            self._pause_executor(only=d.robots)
        out = self._goal_msg(msg, d.todo, d.busy, d.new_mission, d.robots)
        # Record what the incoming plan will be grounded under, so the next goal
        # can tell whether the constraints have since changed.
        self._plan_constraints = constraint_signature(msg.constraints)
        self._pending_goal = msg
        # The planner plans on this map or a newer one, so a plan never looks
        # newer than it is.
        self._pending = Pending(
            d.action,
            out,
            list(d.todo),
            self._stamp,
            self._now(),
            list(d.robots),
            list(d.dropped),
        )
        self._goal_pub.publish(out)

    # ------------- logging -------------

    def _describe(self, goal) -> str:
        """How the operator is told about a goal. One addressed to the base
        station is the fleet's: the base station is not a robot."""
        if goal is None:
            return "the goal"
        owner = goal.goal.robot_id
        if not owner or owner == self._base_station:
            return f"the fleet goal '{goal.goal.pddl_goal}'"
        return f"{owner}'s goal '{goal.goal.pddl_goal}'"

    def _log_decision(self, msg: ConstrainedPddlGoalMsg, d: Decision) -> None:
        classes = "; ".join(
            f"{c} = {cls}" + (f" ({detail})" if detail else "")
            for c, (cls, detail) in d.classes.items()
        )
        robots = "; ".join(
            f"{r}: {e.describe()}" for r, e in sorted(self._robots.items())
        )
        verb = {
            SKIP: "skipping",
            WAIT: "not replanning",
            REPAIR: "replanning (repair)",
            ADDITIONAL: "planning an additional task",
            SCOPED: f"replanning for {_names(d.robots)} only (robot-scoped)",
        }[d.action]
        self.get_logger().info(
            f"{verb}: {d.reason}. {self._describe(msg)} "
            f"[{classes or 'no conjuncts'}] robots [{robots or 'none'}] "
            f"map {self._stamp}"
            + (
                "; forwarded to planner"
                if d.action in (REPAIR, ADDITIONAL, SCOPED)
                else ""
            )
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
