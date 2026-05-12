"""Helpers for plan-repair: extract visited goals and check plan satisfaction."""

import re
from typing import Set

from dsg_pddl.dsg_pddl_planning import PddlPlan

# Match visited-place, visited-object, and visited-poi goal predicates.
VISITED_RE = re.compile(r"\(visited-(?:place|object|poi)\s+([^\s\)]+)\)")


def extract_visited_places(pddl_goal: str) -> Set[str]:
    """Extract all point-of-interest IDs from visited-place/visited-object goals."""
    return set(VISITED_RE.findall(pddl_goal))


def extract_visited_pois_from_plan(plan: PddlPlan) -> Set[str]:
    """Extract all POI IDs visited by goto-poi actions in the plan."""
    pois: Set[str] = set()
    for sym in plan.symbolic_actions:
        if len(sym) >= 2 and sym[0] == "goto-poi":
            pois.add(sym[-1])
    return pois


def extract_forbidden_pois(constraints_json: str) -> Set[str]:
    """Extract forbidden POI IDs from a JSON constraints string."""
    import json

    forbidden: Set[str] = set()
    if not constraints_json:
        return forbidden
    try:
        for c in json.loads(constraints_json):
            if len(c) >= 2 and c[0] == "forbidden-poi":
                forbidden.add(c[1])
    except Exception:
        pass
    return forbidden


def plan_visits_all(
    visited_pois: Set[str], requested: Set[str], forbidden: Set[str] = set()
) -> bool:
    """Check whether the plan covers all requested targets without visiting forbidden POIs."""
    if not requested:
        return False
    if forbidden and forbidden & visited_pois:
        return False  # plan visits a forbidden POI → must replan
    return requested.issubset(visited_pois)
