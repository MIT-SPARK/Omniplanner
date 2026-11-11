import ast
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import parse
from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from nlu_interface.interface import LLMInterface
from plum import dispatch

from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal

logger = logging.getLogger(__name__)


def _log_llm_usage(llm_interface) -> None:
    """Log token usage of the last LLM call (OpenAI responses carry ``usage``)."""
    try:
        resp = llm_interface.response_history[-1]
        usage = resp.usage
        logger.info(
            "LLM usage: model=%s input_tokens=%s output_tokens=%s",
            getattr(resp, "model", "?"),
            usage.input_tokens,
            usage.output_tokens,
        )
    except Exception:  # noqa: BLE001 - usage is best-effort telemetry
        pass


def _names_robot(command: str, robot: str) -> bool:
    """True if ``command`` names ``robot`` as a whole word (case-insensitive)."""
    return re.search(rf"\b{re.escape(robot.lower())}\b", command.lower()) is not None


def _check_known_robot(robot_id: str, robot_states: dict) -> None:
    """Fail early if the goal is addressed to a robot the planner doesn't know
    (``robot_states`` has an entry per configured robot, even without a pose)."""
    if robot_id and robot_id not in robot_states:
        known = ", ".join(sorted(robot_states)) or "none"
        raise ValueError(
            f"Language goal addressed to unknown robot {robot_id!r} (known: {known})"
        )


def _assign_to_addressed_robot(goal_dict, goal):
    """If the operator addressed one robot and the LLM gave the whole goal to a
    single *different* robot the command never mentions, give it back."""
    if not goal.robot_id or not isinstance(goal_dict, dict) or len(goal_dict) != 1:
        return goal_dict
    ((assigned, pddl_goal),) = goal_dict.items()
    if assigned == goal.robot_id or _names_robot(goal.command, assigned):
        return goal_dict
    logger.warning(
        "LLM assigned the goal to %r but the instruction was issued to %r; reassigning",
        assigned,
        goal.robot_id,
    )
    return {goal.robot_id: pddl_goal}


@dataclass
class LanguageDomain:
    domain_type: str
    pddl_domain: PddlDomain = None
    llm_interface: LLMInterface = None


@dataclass
class LanguageGoal:
    robot_id: str
    command: str


@dispatch
def ground_problem(
    domain: LanguageDomain,
    dsg: Any,
    robot_states: Mapping,
    goal: LanguageGoal,
    feedback: Any = None,
):
    if domain.domain_type == "goto_points":
        # No LLM here: the command must already be a list of node symbols.
        points = goal.command.split()
        bad = [p for p in points if parse.parse("{}({:d})", p) is None]
        if not points or bad:
            raise ValueError(
                f"domain_type 'goto_points' expects node symbols like 'O(12) P(850)', "
                f"got {bad or goal.command!r}; use domain_type 'Pddl' (or 'default') "
                "for natural-language commands"
            )
        if not goal.robot_id:
            raise ValueError("domain_type 'goto_points' needs a robot_id")
        _check_known_robot(goal.robot_id, robot_states)
        language_grounded_goal = GotoPointsGoal(
            goal_points=points, robot_id=goal.robot_id
        )
        problem_type = GotoPointsDomain()
        return ground_problem(
            problem_type, dsg, robot_states, language_grounded_goal, feedback
        )
    elif domain.domain_type == "Pddl":
        _check_known_robot(goal.robot_id, robot_states)
        # Query the LLM & Parse the response
        instruction = goal.command
        if goal.robot_id:
            # The prompt describes a whole robot team; without this the LLM picks
            # an arbitrary robot even though the operator addressed one.
            instruction = (
                f"{goal.command}\n(This instruction was issued to robot "
                f"'{goal.robot_id}'. Assign the tasks to '{goal.robot_id}' unless the "
                "instruction explicitly names other robots.)"
            )
        response = domain.llm_interface.request_plan_specification(instruction, dsg)
        _log_llm_usage(domain.llm_interface)
        try:
            goal_dict = ast.literal_eval(response)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"LLM returned an unparsable goal: {response!r}") from e
        goal_dict = _assign_to_addressed_robot(goal_dict, goal)
        # Publish feedback to the rviz interface
        if feedback is not None:
            publish = feedback.plugin_feedback_collectors["language_planner"].publish[
                "llm_response"
            ]
            publish(str(goal_dict))

        problems = []
        for robot_name, goal in goal_dict.items():
            # Construct the PddlGoal object for the PDDL planner
            pddl_language_grounded_goal = PddlGoal(pddl_goal=goal, robot_id=robot_name)

            grounded_problem = ground_problem(
                domain.pddl_domain,
                dsg,
                robot_states,
                pddl_language_grounded_goal,
                feedback,
            )
            problems.append(grounded_problem)

        return problems

    else:
        raise Exception(f"Unexpected domain_type: {domain.domain_type}")
