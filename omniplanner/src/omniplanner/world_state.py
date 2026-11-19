"""What a mission has already done, so a replan only plans what is left.

Planners plan from a goal and a map, and the map says nothing about progress:
without this, every replan starts the mission over -- revisiting places,
re-inspecting objects, and picking up an object the robot is already holding.

Progress comes from the executors' action reports, which every planner's plans
produce alike:

* a completed Follow visits every place and object near its waypoints (a path
  passes through them, as fast-downward's goto-poi does);
* a successful Gaze inspects its object;
* a successful Pick / Place picks up / puts down its object.

Holding is physical state, so the map is the authority when it tracks it (the
heracles publisher puts it in the scene graph's metadata); the action reports
stand in when no map does, e.g. with a prior-map JSON. They also stand in for
an object the map has not caught up with: a successful PICK the map never
heard of (the holding update was lost, or a reload wiped HOLDS) still leaves
the object in that robot's hand, until the map has once agreed with it.

Symbols are kept lowercase, the form both planners use.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import spark_dsg
from dsg_pddl.pddl_utils import lisp_string_to_ast

logger = logging.getLogger(__name__)

# Key the heracles publisher writes map state under in the graph's metadata.
MAP_METADATA_KEY = "heracles"

# The constraint the goal manager attaches to a goal that ADDS work while the
# current mission runs: (busy-robots r1 r2 ...) names the robots still busy
# with it. Its presence makes the goal an additional task, planned on free
# robots only; with no symbols, every robot is free. Without it the goal is a
# repair of the current mission, and every robot may be replanned.
BUSY_ROBOTS = "busy-robots"
# A goal the goal manager sends when every robot is idle starts a new mission:
# (new-mission). Earlier missions' visits and inspections are forgotten, so a
# goal sent again after it finished is planned again; holding, and who put each
# object down, are kept.
NEW_MISSION = "new-mission"


@dataclass
class WorldState:
    """Mission progress a planner should treat as already achieved."""

    visited: Set[str] = field(default_factory=set)
    inspected: Set[str] = field(default_factory=set)
    # {robot -> objects it holds}
    holding: Dict[str, List[str]] = field(default_factory=dict)
    # {robot -> what that robot itself visited / inspected}, for goals that
    # name the robot: (visited-object-by r o), (inspected-by r o).
    visited_by: Dict[str, Set[str]] = field(default_factory=dict)
    inspected_by: Dict[str, Set[str]] = field(default_factory=dict)
    # {robot -> objects that robot put down}, for (object-placed-by r o p).
    placed_by: Dict[str, Set[str]] = field(default_factory=dict)
    # Robots with no arm or gaze (a Husky): they cannot inspect, pick or place.
    cannot_manipulate: Set[str] = field(default_factory=set)
    # {robot -> platform}, e.g. {"gauss": "husky", "euclid": "spot"}. A
    # platform may avoid some places (a Husky: those marked {"curb": true});
    # see PLATFORM_AVOIDS in dsg_pddl_grounding_multirobot. A robot not listed
    # goes anywhere.
    platform: Dict[str, str] = field(default_factory=dict)

    def held_by(self, robot: str) -> List[str]:
        return list(self.holding.get(robot.lower(), []))

    def holder_of(self, obj: str) -> Optional[str]:
        for robot, objects in self.holding.items():
            if obj.lower() in objects:
                return robot
        return None


def holding_from_dsg(dsg) -> Optional[Dict[str, List[str]]]:
    """What the map says each robot holds, or None if the map does not track it."""
    try:
        state = dsg.metadata.get().get(MAP_METADATA_KEY)
    except Exception:
        return None
    if not isinstance(state, dict) or "holding" not in state:
        return None
    return {
        str(robot).lower(): [str(o).lower() for o in objects]
        for robot, objects in (state["holding"] or {}).items()
        if objects
    }


def busy_robots(constraints) -> Optional[Set[str]]:
    """Robots a busy-robots constraint says are busy, or None for a repair."""
    found = None
    for c in constraints or []:
        if getattr(c, "predicate", None) == BUSY_ROBOTS:
            found = (found or set()) | {str(s).lower() for s in c.symbols}
    return found


def new_mission(constraints) -> bool:
    return any(getattr(c, "predicate", None) == NEW_MISSION for c in constraints or [])


def goal_tokens(pddl_goal: str) -> Set[str]:
    """Every name a goal string mentions, lowercase."""
    return set(re.findall(r"[a-z0-9_]+", (pddl_goal or "").lower()))


def unavailable_robots(constraints, world_state=None, named: Iterable = ()) -> Set[str]:
    """Robots an additional task must leave alone: those the goal manager says
    are busy and those holding an object (busy until it is put down), unless
    the goal names them (named). Empty for a repair, which may replan every
    robot."""
    busy = busy_robots(constraints)
    if busy is None:
        return set()
    holding = getattr(world_state, "holding", None) or {}
    busy |= {r.lower() for r, objects in holding.items() if objects}
    return busy - {str(n).lower() for n in named}


def _layer_nodes(dsg, layer):
    try:
        return list(dsg.get_layer(layer).nodes)
    except Exception:
        return []


def symbols_near(dsg, points: Iterable, radius_m: float) -> Set[str]:
    """Objects and mesh places a path visited.

    A place or object within radius_m (in the plane) of a waypoint, and an
    object whose nearest mesh place is a waypoint: a route to an object stops
    at that place, which can be farther
    from the object than any sensible radius (2.2 m for one object in b45).
    """
    pts = np.array([[float(p[0]), float(p[1])] for p in points], dtype=float)
    if len(pts) == 0:
        return set()
    places = np.array(
        [
            np.asarray(n.attributes.position, dtype=float)[:2]
            for n in _layer_nodes(dsg, spark_dsg.DsgLayers.MESH_PLACES)
        ]
    ).reshape(-1, 2)
    near = set()
    for layer in (spark_dsg.DsgLayers.OBJECTS, spark_dsg.DsgLayers.MESH_PLACES):
        for node in _layer_nodes(dsg, layer):
            xy = np.asarray(node.attributes.position, dtype=float)[:2]
            reached = np.min(np.linalg.norm(pts - xy, axis=1)) <= radius_m
            if not reached and layer == spark_dsg.DsgLayers.OBJECTS and len(places):
                stop = places[np.argmin(np.linalg.norm(places - xy, axis=1))]
                reached = np.min(np.linalg.norm(pts - stop, axis=1)) <= 0.05
            if reached:
                near.add(node.id.str(True).lower())
    return near


def kept_objects(pddl_goal: str) -> Set[str]:
    """Objects a goal still wants moved or held: (object-in-place o p), (holding r o).

    A robot holding any other object puts it down before the new plan starts.
    """
    try:
        ast = lisp_string_to_ast((pddl_goal or "").replace(")(", ") ("))
    except Exception:
        return set()
    kept: Set[str] = set()

    def walk(node):
        if isinstance(node, str) or not node:
            return
        head = node[0]
        if not isinstance(head, str):
            for sub in node:
                walk(sub)
            return
        if head == "object-in-place" and len(node) >= 3:
            kept.add(str(node[1]).lower())
        elif head == "object-placed-by" and len(node) >= 4:
            kept.add(str(node[2]).lower())
        elif head == "holding" and len(node) >= 2:
            kept.add(str(node[-1]).lower())
        else:
            for sub in node[1:]:
                walk(sub)

    walk(ast)
    return kept


def split_held(
    state: WorldState, kept: Set[str]
) -> Tuple[WorldState, Dict[str, List[str]]]:
    """(state to plan from, {robot -> objects to put down first}).

    Objects the goal no longer mentions are released before the new plan runs,
    so the planner plans as if those hands were already empty.
    """
    holding, release = {}, {}
    for robot, objects in state.holding.items():
        keep = [o for o in objects if o in kept]
        drop = [o for o in objects if o not in kept]
        if keep:
            holding[robot] = keep
        if drop:
            release[robot] = drop
    return (
        WorldState(
            set(state.visited),
            set(state.inspected),
            holding,
            {r: set(s) for r, s in state.visited_by.items()},
            {r: set(s) for r, s in state.inspected_by.items()},
            {r: set(s) for r, s in state.placed_by.items()},
        ),
        release,
    )


def goal_parts(pddl_goal: str) -> Set[str]:
    """The top-level conjuncts of a goal, lower case, spaces normalised."""
    text = " ".join((pddl_goal or "").lower().split())
    if not text:
        return set()
    if not re.match(r"\(\s*and\b", text):
        return {text}
    parts, depth, start = set(), 0, None
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
            if depth == 2:
                start = i
        elif ch == ")":
            if depth == 2 and start is not None:
                parts.add(text[start : i + 1])
            depth -= 1
    return parts


def is_team_part(part: str) -> bool:
    """A conjunct that leaves the robot open: any of its predicates is not a
    ``-by`` one. It is accomplished by the team, not by one robot."""
    heads = [
        h for h in re.findall(r"\(\s*([\w-]+)", part) if h not in ("and", "or", "not")
    ]
    return any(not h.endswith("-by") for h in heads)


def visits_to_forget(goals, running, new_goal, planned_for=None) -> Set[str]:
    """Whose visits a plan for ``new_goal`` starts without.

    A robot's visits are the history of its goal: kept for as long as the goal
    is not accomplished, so that no repair of it goes anywhere twice, and
    forgotten once it is.

    goals: {robot: (the goal its visits were made for, accomplished?)}
    running: robots with a plan running now; planned_for: the robots the new
    goal may give a plan (None: any).

    * Running: kept.
    * Accomplished: forgotten -- unless a part of its goal that names no robot
      is still being worked on by a running robot. That part is the team's,
      and the team has not accomplished it.
    * Stopped short of its goal: kept while the new goal still asks for a part
      of it (a replan, or the goal sent again). Forgotten when the robot is
      planned for a goal that has none of it: that goal was dropped.
    """
    running = {r.lower() for r in running}
    new_parts = goal_parts(new_goal)
    live_team = set()
    for r in running:
        if r in goals:
            live_team |= {c for c in goal_parts(goals[r][0]) if is_team_part(c)}
    forget = set()
    for robot, (goal, accomplished) in goals.items():
        if robot in running:
            continue
        parts = goal_parts(goal)
        if accomplished:
            if not (parts & live_team):
                forget.add(robot)
        elif (planned_for is None or robot in planned_for) and not (parts & new_parts):
            forget.add(robot)
    return forget


class WorldStateTracker:
    """Accumulates mission progress from executor action reports."""

    def __init__(self, visited_radius_m: float = 1.0):
        self.visited_radius_m = visited_radius_m
        self.visited: Set[str] = set()
        self.inspected: Set[str] = set()
        self.visited_by: Dict[str, Set[str]] = {}
        self.inspected_by: Dict[str, Set[str]] = {}
        self.placed_by: Dict[str, Set[str]] = {}
        # Holding as the action reports tell it: used when the map has none,
        # and merged with the map's (see snapshot).
        self._reported_holding: Dict[str, Set[str]] = {}
        # {object -> (last successful PICK or PLACE, robot)}, and the objects
        # whose last report the map has since agreed with: from then on the
        # map is newer than the report, and wins.
        self._last_report: Dict[str, Tuple[str, str]] = {}
        self._confirmed: Set[str] = set()
        self._disagreements: Set[tuple] = set()

    def reset(self):
        """Forget mission progress (visits, inspections). Holding and who put
        an object down are physical state and survive: a new mission does not
        empty anybody's gripper, and an object stays where its robot left it
        (a later pick forgets that, and the planners also check its place)."""
        self.visited.clear()
        self.inspected.clear()
        self.visited_by.clear()
        self.inspected_by.clear()

    def forget_visits(self, keep: Iterable[str] = ()) -> Set[str]:
        """Forget what every robot not in keep visited and inspected; returns
        the robots forgotten. What a kept robot also visited stays visited.
        Holding and placements are physical state, and stay (see reset)."""
        keep = {str(r).lower() for r in keep}
        gone = {r for r in (*self.visited_by, *self.inspected_by) if r not in keep}
        for done, by in (
            (self.visited, self.visited_by),
            (self.inspected, self.inspected_by),
        ):
            theirs = set().union(*(by.pop(r, set()) for r in gone))
            done -= theirs - set().union(*by.values())
        return gone

    def forget_unclaimed(self) -> None:
        """Forget visits and inspections that are no robot's."""
        self.visited &= set().union(*self.visited_by.values())
        self.inspected &= set().union(*self.inspected_by.values())

    def record(
        self,
        action_type: str,
        success: bool,
        robot: str,
        object_id: str = "",
        points: Iterable = (),
        dsg=None,
    ) -> Set[str]:
        """Fold one action report in; returns the symbols it newly achieved."""
        if not success:
            return set()
        robot, obj = (robot or "").lower(), (object_id or "").lower()
        new: Set[str] = set()
        if action_type == "FOLLOW" and dsg is not None:
            near = symbols_near(dsg, points, self.visited_radius_m)
            new = near - self.visited
            self.visited |= new
            if robot:
                self.visited_by.setdefault(robot, set()).update(near)
        elif action_type == "GAZE" and obj:
            new = {obj} - self.inspected
            self.inspected.add(obj)
            if robot:
                self.inspected_by.setdefault(robot, set()).add(obj)
        elif action_type == "PICK" and obj:
            for objects in self._reported_holding.values():
                objects.discard(obj)
            self._reported_holding.setdefault(robot, set()).add(obj)
            for objects in self.placed_by.values():
                objects.discard(obj)  # picked again: no longer where it was put
            self._note_report(obj, "PICK", robot)
            new = {obj}
        elif action_type == "PLACE" and obj:
            self._reported_holding.get(robot, set()).discard(obj)
            self.placed_by.setdefault(robot, set()).add(obj)
            self._note_report(obj, "PLACE", robot)
            new = {obj}
        return new

    def _note_report(self, obj: str, action: str, robot: str) -> None:
        self._last_report[obj] = (action, robot)
        self._confirmed.discard(obj)

    def snapshot(self, dsg=None) -> WorldState:
        holding = holding_from_dsg(dsg) if dsg is not None else None
        if holding is None:
            holding = {
                r: sorted(objs) for r, objs in self._reported_holding.items() if objs
            }
        else:
            holding = self._merge_holding(holding)
        return WorldState(
            set(self.visited),
            set(self.inspected),
            holding,
            {r: set(s) for r, s in self.visited_by.items()},
            {r: set(s) for r, s in self.inspected_by.items()},
            {r: set(s) for r, s in self.placed_by.items()},
        )

    def _merge_holding(self, mapped: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """The map's holding, corrected by action reports it has not caught up with.

        A PICK the map does not show leaves the object in that robot's hand; a
        PLACE it does not show takes it out. Once the map agrees with an
        object's last report, the map is the newer word on it.
        """
        holding = {r: list(objs) for r, objs in mapped.items()}
        holder = {o: r for r, objs in holding.items() for o in objs}
        for obj, (action, robot) in sorted(self._last_report.items()):
            if obj in self._confirmed:
                continue
            agrees = (holder.get(obj) == robot) == (action == "PICK")
            if agrees:
                self._confirmed.add(obj)
                continue
            if action == "PICK" and obj not in holder:
                holding.setdefault(robot, []).append(obj)
                why = f"the map says nobody holds {obj}, but {robot} picked it"
            elif action == "PLACE" and holder.get(obj) == robot:
                holding[robot].remove(obj)
                why = f"the map says {robot} holds {obj}, but it put it down"
            else:
                continue  # the map has another robot holding it: trust the map
            if (obj, action, robot) not in self._disagreements:
                self._disagreements.add((obj, action, robot))
                logger.warning("Holding: %s; planning with the action report", why)
        return {r: sorted(objs) for r, objs in holding.items() if objs}
