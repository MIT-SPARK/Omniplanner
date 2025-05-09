import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import reduce, total_ordering
from typing import Dict, List, Optional, overload

import numpy as np
import spark_dsg
from multipledispatch import dispatch

from omniplanner.tsp import LayerPlanner

logger = logging.getLogger(__name__)


def extract_facts(goal, predicate):
    match goal:
        case tuple() | list():
            if len(goal) == 0:
                return ()
            elif goal[0] == predicate:
                return (goal,)
            else:
                children = (extract_facts(g, predicate) for g in goal)
                return reduce(lambda x, y: x + y, children)
        case _:
            return ()


def tokenize_lisp(string):
    return string.replace("\n", "").replace("(", "( ").replace(")", " )").split()


def get_lisp_ast(toks):
    t = toks.pop(0)
    if t == "(":
        exp = ()
        while toks[0] != ")":
            exp += (get_lisp_ast(toks),)
        toks.pop(0)
        return exp
    elif t != ")":
        return t
    else:
        raise SyntaxError("Unexpected )")


def lisp_string_to_ast(string):
    return get_lisp_ast(tokenize_lisp(string))


def generate_symbol_connectivity(G, symbols):
    # layer_planner = LayerPlanner(dsg, spark_dsg.DsgLayers.MESH_PLACES)
    layer_planner = LayerPlanner(G, 20)

    connections = []
    for si in symbols:
        for sj in symbols:
            if si <= sj:
                continue

            distance = layer_planner.get_external_distance(si.position, sj.position)
            connections.append((si, sj, distance))

    return connections


def symbol_connectivity_to_pddl(connectivity):
    connections_init = []

    for info_s, info_t, dist in connectivity:
        s = info_s.symbol
        t = info_t.symbol
        d = int(dist)

        connected_str = f"(connected {s} {t})"
        distance = f"(= (distance {s} {t}) {d})"
        distance_rev = f"(= (distance {t} {s}) {d})"

        connections_init.append(connected_str)
        connections_init.append(distance)
        connections_init.append(distance_rev)

    return "\n".join(connections_init)


def generate_initial_pddl(G, symbols_of_interest, start_symbol):
    connectivity = generate_symbol_connectivity(G, symbols_of_interest)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    init_pddl = f"""(:init
  (= (total-cost) 0)
  (at-poi {start_symbol.symbol})
  {connectivity_pddl}
  )"""
    return init_pddl


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


def extract_symbols_of_interest(G, pddl_goal):
    place_facts = extract_facts(pddl_goal, "visited-place")
    place_facts += extract_facts(pddl_goal, "at-place")

    object_facts = extract_facts(pddl_goal, "visited-object")
    object_facts += extract_facts(pddl_goal, "at-object")

    place_symbols = [PddlSymbol(f[1], "place", []) for f in place_facts]
    object_symbols = [PddlSymbol(f[1], "object", []) for f in object_facts]

    return place_symbols + object_symbols


def generate_object_pddl(pddl_symbols):
    places = []
    objects = []
    for s in pddl_symbols:
        if s.layer == "place":
            places.append(s.symbol)
        elif s.layer == "object":
            objects.append(s.symbol)

    if len(places) > 0:
        place_line = " ".join(places) + " - place\n"
    else:
        place_line = ""

    if len(objects) > 0:
        object_line = " ".join(objects) + " - dsg_object\n"
    else:
        object_line = ""
    object_pddl = f"(:objects\n{place_line}{object_line})"

    return object_pddl


def simplify(pddl):
    return pddl


def ast_to_string(ast):
    match ast:
        case list() | tuple():
            elements = [ast_to_string(e) for e in ast]
            elements_str = " ".join(elements)
            return f"({elements_str})"
        case _:
            return str(ast)


# PDDL is case-insensitive, but spark_dsg is case sensitive.
# Here to convert back and forth, but this is *extremely* brittle.
def pddl_char_to_dsg_char(c):
    match c:
        case "r":
            return "R"
        case "o":
            return "O"
        case _:
            return c


def add_symbol_positions(G, symbols):
    for s in symbols:
        if s.position is not None:
            continue
        else:
            pddl_symbol_char = s.symbol[0]
            dsg_symbol_char = pddl_char_to_dsg_char(pddl_symbol_char)
            ns = spark_dsg.NodeSymbol(dsg_symbol_char, int(s.symbol[1:]))
            position = G.get_node(ns).attributes.position[:2]
            if position is None:
                raise Exception(f"Could not find node {ns} in DSG")
            s.position = position
    return symbols


def normalize_symbols(symbols):
    for s in symbols:
        s.symbol = s.symbol.lower()


def generate_inspection_pddl(G, raw_pddl_goal_string, initial_position):
    problem_name = "goto-object-problem"
    problem_domain = "goto-object-domain"

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)
    goal_symbols_of_interest = extract_symbols_of_interest(G, parsed_pddl_goal)
    normalize_symbols(goal_symbols_of_interest)
    logger.info(f"Extracted goal_symbols of interest: {goal_symbols_of_interest}")
    goal_pddl = simplify(parsed_pddl_goal)
    # ideally we check the goal here and see if we can run a more specialized planner based on the simplified goal

    goal_pddl_clause = ast_to_string(goal_pddl)
    goal_pddl_string = f"(:goal {goal_pddl_clause})"
    metric_pddl = "(:metric minimize (total-cost))"

    start_place_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    symbols_of_interest = [start_place_symbol] + goal_symbols_of_interest

    add_symbol_positions(G, symbols_of_interest)

    init_pddl = generate_initial_pddl(G, symbols_of_interest, start_place_symbol)
    object_pddl = generate_object_pddl(symbols_of_interest)
    problem = f"""(define (problem {problem_name})
    (:domain {problem_domain})
    {object_pddl}
    {init_pddl}
    {goal_pddl_string}
    {metric_pddl}
    )"""

    return problem, symbols_of_interest


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


@dataclass
class PddlGoal:
    pddl_goal: str
    robot_id: str


@dataclass
class GroundedPddlProblem:
    domain: PddlDomain
    problem_str: str
    symbols: Dict[str, PddlSymbol]


@dataclass
class PddlPlan:
    symbolic_actions: List[tuple]
    parameterized_actions: List
    symbols: Dict[str, PddlSymbol]


@overload
@dispatch(PddlDomain, object, dict, PddlGoal)
def ground_problem(domain, dsg, robot_states, goal) -> GroundedPddlProblem:
    logger.info(f"Grounding PDDL Problem {domain.domain_name}")

    start = robot_states[goal.robot_id][:2]

    # TODO: TBD whether we want to check the domain here and choose how
    # to instantiate the PDDL problem, or if that should be in a separately
    # ground_problem function.
    if domain.domain_name == "goto-object-domain":
        pddl_problem, symbols = generate_inspection_pddl(dsg, goal.pddl_goal, start)
    else:
        raise NotImplementedError(
            f"I don't know how to ground a domain of type {domain.domain_name}!"
        )

    symbol_dict = {s.symbol: s for s in symbols}
    return GroundedPddlProblem(domain, pddl_problem, symbol_dict)


@dispatch(GroundedPddlProblem, object)
def make_plan(grounded_problem, map_context) -> PddlPlan:
    plan = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        problem_fn = os.path.join(tmpdirname, "problem.pddl")
        domain_fn = os.path.join(tmpdirname, "domain.pddl")
        plan_fn = os.path.join(tmpdirname, "plan.txt")

        with open(problem_fn, "w") as fo:
            fo.write(grounded_problem.problem_str)

        with open(domain_fn, "w") as fo:
            fo.write(grounded_problem.domain.to_string())

        command = ["fast-downward"]
        command += ["--plan-file", plan_fn]
        command += [domain_fn]
        command += [problem_fn]
        command += [
            "--search",
            "let(hff, ff(), let(hcea, cea(), lazy_greedy([hff, hcea], preferred=[hff, hcea])))",
        ]

        logger.warning(f"Calling: {command}")
        return_code = subprocess.run(command)
        logger.warning(f"Return code: {return_code}")

        with open(plan_fn, "r") as fo:
            lines = fo.readlines()

    plan = [lisp_string_to_ast(line) for line in lines[:-1]]

    logger.warning(f"Made plan {plan}")

    parameterized_plan = []

    layer_planner = LayerPlanner(map_context, 20)
    last_pose = np.zeros(2)
    for p in plan:
        match p[0]:
            case "goto-poi":
                path = layer_planner.get_external_path(
                    grounded_problem.symbols[p[1]].position,
                    grounded_problem.symbols[p[2]].position,
                )
                parameterized_plan.append(path)
                last_pose = path[-1]
            case "inspect":
                parameterized_plan.append(
                    [last_pose, grounded_problem.symbols[p[1]].position]
                )
            case _:
                raise Exception(
                    f"Plan contains {p[0]} action, but I don't know how to parameterize!"
                )

    return PddlPlan(plan, parameterized_plan, grounded_problem.symbols)


if __name__ == "__main__":
    dsg_fn = "/home/ubuntu/lxc_datashare/west_point_fused_map_wregions.json"
    G = spark_dsg.DynamicSceneGraph.load(dsg_fn)

    goal = "(or (and (visited-place R35) (visited-place R69)) (and (visited-place R71) (visited-place R83)))"

    pddl_problem = generate_inspection_pddl(G, goal, np.array([1, 2]))
    with open("dsg_planning_problem.pddl", "w") as fo:
        fo.write(pddl_problem)
