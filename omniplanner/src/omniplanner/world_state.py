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
only stand in when no map does, e.g. with a prior-map JSON.

Symbols are kept lowercase, the form both planners use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import spark_dsg
from dsg_pddl.pddl_utils import lisp_string_to_ast

# Key the heracles publisher writes map state under in the graph's metadata.
MAP_METADATA_KEY = "heracles"


@dataclass
class WorldState:
    """Mission progress a planner should treat as already achieved."""

    visited: Set[str] = field(default_factory=set)
    inspected: Set[str] = field(default_factory=set)
    # {robot -> objects it holds}
    holding: Dict[str, List[str]] = field(default_factory=dict)

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


def _layer_nodes(dsg, layer):
    try:
        return list(dsg.get_layer(layer).nodes)
    except Exception:
        return []


def symbols_near(dsg, points: Iterable, radius_m: float) -> Set[str]:
    """Objects and mesh places within radius_m (in the plane) of any point."""
    pts = np.array([[float(p[0]), float(p[1])] for p in points], dtype=float)
    if len(pts) == 0:
        return set()
    near = set()
    for layer in (spark_dsg.DsgLayers.OBJECTS, spark_dsg.DsgLayers.MESH_PLACES):
        for node in _layer_nodes(dsg, layer):
            xy = np.asarray(node.attributes.position, dtype=float)[:2]
            if np.min(np.linalg.norm(pts - xy, axis=1)) <= radius_m:
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
        WorldState(set(state.visited), set(state.inspected), holding),
        release,
    )


class WorldStateTracker:
    """Accumulates mission progress from executor action reports."""

    def __init__(self, visited_radius_m: float = 1.5):
        self.visited_radius_m = visited_radius_m
        self.visited: Set[str] = set()
        self.inspected: Set[str] = set()
        # Holding as the action reports tell it; used only when the map has none.
        self._reported_holding: Dict[str, Set[str]] = {}

    def reset(self):
        """Forget mission progress (visits, inspections). Holding is physical
        state and survives: a new mission does not empty anybody's gripper."""
        self.visited.clear()
        self.inspected.clear()

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
            new = symbols_near(dsg, points, self.visited_radius_m) - self.visited
            self.visited |= new
        elif action_type == "GAZE" and obj:
            new = {obj} - self.inspected
            self.inspected.add(obj)
        elif action_type == "PICK" and obj:
            self._reported_holding.setdefault(robot, set()).add(obj)
            new = {obj}
        elif action_type == "PLACE" and obj:
            self._reported_holding.get(robot, set()).discard(obj)
            new = {obj}
        return new

    def snapshot(self, dsg=None) -> WorldState:
        holding = holding_from_dsg(dsg) if dsg is not None else None
        if holding is None:
            holding = {
                r: sorted(objs) for r, objs in self._reported_holding.items() if objs
            }
        return WorldState(set(self.visited), set(self.inspected), holding)
