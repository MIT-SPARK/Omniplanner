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
    at that place -- GRSTAPS-X snaps every target to it -- which can be farther
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
