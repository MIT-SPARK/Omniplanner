import logging
from dataclasses import dataclass
from functools import total_ordering
from typing import Dict, List, Optional

import numpy as np

from dsg_pddl.pddl_utils import ast_to_string, lisp_string_to_ast

logger = logging.getLogger(__name__)


@total_ordering
@dataclass
class PddlSymbol:
    symbol: str
    layer: str  # object, place, etc
    unary_predicates_to_apply: List[str]
    position: Optional[np.ndarray] = None

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __lt__(self, other):
        return self.symbol < other.symbol


@dataclass
class PddlProblem:
    """Representing Fast-Downward's flavor of PDDL"""

    name: str
    domain: str
    objects: dict
    initial_facts: tuple
    goal: tuple
    optimizing: bool

    def to_string(self):
        object_pddl_str = self.to_object_string(self.objects)
        init_pddl_str = self.to_init_string(self.initial_facts)
        goal_pddl_str = self.to_goal_string(self.goal)

        problem = f"""(define (problem {self.name})
        (:domain {self.domain})
        {object_pddl_str}
        {init_pddl_str}
        {goal_pddl_str}
        """
        if self.optimizing:
            metric_pddl = "(:metric minimize (total-cost))"
            problem += "\n" + metric_pddl

        problem += ")"

        print("pddl problem: ")
        # print(problem)
        return problem

    def to_goal_string(self, goal):
        goal_pddl_clause = ast_to_string(goal)
        goal_pddl_string = f"(:goal {goal_pddl_clause})"
        return goal_pddl_string

    def to_object_string(self, object_dict: dict):
        object_pddl = "(:objects\n"
        for typ, objects in object_dict.items():
            if len(objects) > 0:
                if typ == "":
                    line = " ".join(objects) + "\n"
                else:
                    line = " ".join(objects) + f" - {typ}\n"
                object_pddl += line
        object_pddl += ")"

        return object_pddl

    def to_init_string(self, facts):
        fact_strings = map(ast_to_string, facts)
        fact_string = "\n".join(fact_strings)
        init_pddl = f"(:init\n {fact_string} )"
        return init_pddl


@dataclass
class PddlGoal:
    pddl_goal: str
    robot_id: str


# TODO: need to reexamine this whole parsing framework as some point.
# It's brittle and requires the :type section which should be optional
# similar for :functions
def ensure_pddl_domain(ast):
    if ast[0] != "define":
        raise Exception("Malformed PDDL Domain ast, missing define")

    if ast[2][0] != ":requirements":
        raise Exception("Missing :requirements, must go after name")

    if ast[3][0] != ":types":
        raise Exception("Missing :types, must go after requirements")

    if ast[4][0] != ":predicates":
        raise Exception("Missing :predicates, must go after types")

    if ast[5][0] != ":functions":
        raise Exception("Missing :functions, must go after predicates")

    if ast[5][0] != ":functions":
        raise Exception("Missing :functions, must go after predicates")

    for clause in ast[6:]:
        if clause[0] not in [":derived", ":action"]:
            raise Exception(f"Expected a :deried or :action, not {clause[0]}")


class PddlDomain:
    def __init__(self, domain_str):
        self.domain_ast = lisp_string_to_ast(domain_str)
        self.domain_name = get_domain_name(self.domain_ast)
        self.requirements = get_domain_requirements(self.domain_ast)
        self.get_domain_types = get_domain_types(self.domain_ast)
        self.predicates = get_domain_predicates(self.domain_ast)
        self.functions = get_functions(self.domain_ast)
        self.derived = get_derived(self.domain_ast)
        self.actions = get_actions(self.domain_ast)

    def to_string(self):
        return ast_to_string(self.domain_ast)


def get_domain_name(ast):
    return ast[1][1]


def get_domain_requirements(ast):
    return ast[2][1:]


def get_domain_types(ast):
    # TODO: this is arguably incomplete, because we don't
    # parse the type/subtype relationship
    return ast[3][1:]


def get_domain_predicates(ast):
    return ast[4][1:]


def get_functions(ast):
    return ast[5][1:]


def get_derived(ast):
    derived = ()
    for clause in ast:
        if type(clause) is not tuple:
            continue
        if clause[0] == ":derived":
            derived += clause[1:]
    return derived


def get_actions(ast):
    actions = ()
    for clause in ast:
        if type(clause) is not tuple:
            continue
        if clause[0] == ":action":
            actions += clause[1:]
    return actions


@dataclass
class GroundedPddlProblem:
    domain: PddlDomain
    problem_str: str
    symbols: Dict[str, PddlSymbol]
