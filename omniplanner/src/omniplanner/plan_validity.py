"""Does a scene change invalidate the plan being executed?

A plan is built on the map as it was. When the map changes underneath it --
an object moved, removed or added -- the plan may now send a robot to pick an
object that is no longer there, or visit a place where nothing is left to see.
This decides which changes matter, so the robot replans for those and carries
on for the rest.

A change matters when it touches:

* an object the plan still acts on (an unfinished Pick or Gaze), if it moved
  farther than the threshold or was removed;
* an object the goal names and the mission has not finished with -- a visit
  not yet made, an inspection not yet done, or any relocation, since the goal
  is about where that object ends up.

An object one of our robots holds is never affected by an observation: the
robot is carrying it. Changes our own executors made are filtered out before
this, by their source.

Symbols are compared lowercase, positions in the plane (map frame).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from dsg_pddl.pddl_utils import lisp_string_to_ast

ADDED, REMOVED, MOVED = 0, 1, 2
KIND_NAMES = {ADDED: "added", REMOVED: "removed", MOVED: "moved"}

# {symbol -> (action, (x, y))}: objects the plan still acts on, and where it
# expects them.
Dependencies = Dict[str, Tuple[str, Tuple[float, float]]]


def plan_dependencies(plans: Iterable) -> Dependencies:
    """Objects a set of compiled plans acts on, from their Pick and Gaze actions.

    A Place is left out: its object is in a gripper, not in the scene, and its
    destination is a place, which observations do not move.
    """
    deps: Dependencies = {}
    for seq in plans:
        for a in getattr(seq, "actions", []):
            kind = type(a).__name__
            if kind == "Pick":
                xy = a.object_point
            elif kind == "Gaze":
                xy = a.gaze_point
            else:
                continue
            obj = (getattr(a, "object_id", "") or "").lower()
            if obj:
                deps.setdefault(obj, (kind.lower(), (float(xy[0]), float(xy[1]))))
    return deps


def goal_objects(pddl_goal: str) -> Dict[str, set]:
    """The symbols a goal names, by what it asks of them."""
    out = {"visit": set(), "inspect": set(), "relocate": set()}
    try:
        ast = lisp_string_to_ast((pddl_goal or "").replace(")(", ") ("))
    except Exception:
        return out

    def walk(node):
        if isinstance(node, str) or not node:
            return
        head = node[0]
        if not isinstance(head, str):
            for sub in node:
                walk(sub)
            return
        args = [str(a).lower() for a in node[1:] if isinstance(a, str)]
        if head in ("visited-object", "visited-place", "visited-poi"):
            out["visit"].update(args[:1])
        elif head in ("visited-object-by", "visited-place-by", "visited-poi-by"):
            out["visit"].update(args[-1:])  # (visited-object-by <robot> <target>)
        elif head in ("at-object", "at-place"):
            out["visit"].update(args[-1:])
        elif head in ("safe", "suspicious"):
            out["inspect"].update(args[:1])
        elif head == "inspected-by":
            out["inspect"].update(args[-1:])  # (inspected-by <robot> <object>)
        elif head == "object-in-place":
            out["relocate"].update(args[:1])
        elif head == "holding":
            out["relocate"].update(args[-1:])
        else:
            for sub in node[1:]:
                walk(sub)

    walk(ast)
    return out


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def affecting_changes(
    changes: Iterable[dict],
    deps: Dependencies,
    pddl_goal: str,
    world_state=None,
    move_threshold_m: float = 0.5,
) -> List[Tuple[str, str]]:
    """[(symbol, reason)] for the changes that invalidate the plan.

    changes: {"kind": ADDED|REMOVED|MOVED, "symbol", "old": (x, y) or None,
              "new": (x, y) or None}
    """
    goal = goal_objects(pddl_goal)
    visited = set(getattr(world_state, "visited", set()) or set())
    inspected = set(getattr(world_state, "inspected", set()) or set())
    holder_of = getattr(world_state, "holder_of", lambda _s: None)
    out = []
    for c in changes:
        sym = (c.get("symbol") or "").lower()
        kind = c.get("kind")
        if not sym or holder_of(sym) is not None:
            continue
        moved: Optional[float] = None
        if kind == MOVED and c.get("new") is not None:
            ref = deps[sym][1] if sym in deps else c.get("old")
            moved = _dist(c["new"], ref) if ref is not None else math.inf
            if moved <= move_threshold_m:
                continue
        what = KIND_NAMES.get(kind, "changed") + (
            f" {moved:.1f} m" if moved is not None and moved != math.inf else ""
        )
        if sym in deps:
            verb = {"gaze": "inspect"}.get(deps[sym][0], deps[sym][0])
            out.append((sym, f"{sym} {what}, and the plan still has to {verb} it"))
        elif sym in goal["relocate"]:
            out.append((sym, f"{sym} {what}, and the goal says where it must end up"))
        elif sym in goal["visit"] and sym not in visited:
            out.append((sym, f"{sym} {what} before the visit the goal asks for"))
        elif sym in goal["inspect"] and sym not in inspected:
            out.append((sym, f"{sym} {what} before the inspection the goal asks for"))
    return out


# {robot -> [(action index, [place symbols the Follow passes, in order])]}
Route = Dict[str, List[Tuple[int, List[str]]]]


def route_places(plans: Dict[str, object], place_at) -> Route:
    """The places each robot's Follow actions drive through.

    place_at((x, y)) names the place at a path point, or None for a point that
    is not one (the robot's start, an object it walks up to).
    """
    route: Route = {}
    for robot, seq in plans.items():
        legs = []
        for i, a in enumerate(getattr(seq, "actions", [])):
            if type(a).__name__ != "Follow":
                continue
            places = [place_at((float(p[0]), float(p[1]))) for p in a.path2d]
            legs.append((i, places))
        route[robot] = legs
    return route


def broken_route(
    route: Route, done: Dict[str, int], has_place, connected
) -> List[Tuple[str, str]]:
    """[(robot, why)] for routes still ahead that the map no longer allows.

    A place on a Follow not yet finished is gone, or two places the robot
    drives between in a row are no longer connected. done[robot] is how many
    of its plan's actions have finished.
    """
    broken = []
    for robot, legs in route.items():
        for i, places in legs:
            if i < done.get(robot, 0):
                continue
            why = None
            for a, b in zip(places, places[1:]):
                if a and not has_place(a):
                    why = f"place {a} on {robot}'s route was removed"
                elif b and not has_place(b):
                    why = f"place {b} on {robot}'s route was removed"
                elif a and b and a != b and not connected(a, b):
                    why = f"{robot}'s route from {a} to {b} is no longer connected"
                if why:
                    break
            if (
                why is None
                and len(places) == 1
                and places[0]
                and not has_place(places[0])
            ):
                why = f"place {places[0]} on {robot}'s route was removed"
            if why:
                broken.append((robot, why))
                break
    return broken
