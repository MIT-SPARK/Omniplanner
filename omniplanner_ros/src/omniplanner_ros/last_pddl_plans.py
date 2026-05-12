"""Symbolic plan cache for plan-repair: record latest PDDL plan per robot."""

from typing import Dict, Optional

from dsg_pddl.dsg_pddl_planning import PddlPlan

_last_plans: Dict[str, PddlPlan] = {}


def set_last_plan(robot_name: str, plan: PddlPlan) -> None:
    _last_plans[robot_name] = plan


def get_last_plan(robot_name: str) -> Optional[PddlPlan]:
    return _last_plans.get(robot_name)
