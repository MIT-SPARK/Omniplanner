"""GRSTAPS-X as an omniplanner planning domain (single-agent proof of concept).

Slots into the existing pipeline the same way every other planner does --
`ground_problem` on the domain type, then `make_plan` -- so nothing in the PDDL
or TSP paths changes.

Where it differs from the PDDL planners: omniplanner normally compiles the DSG
*into* the PDDL problem (hundreds of symbols, `connected` facts, distances).
GRSTAPS-X keeps geometry out of the PDDL and takes a separate euclidean motion
graph, so grounding here means writing four files:

    domain.pddl               mission logic only (visit-location)
    problem.pddl              which locations to visit
    action_trait_config.json  robots, species traits, action -> subtask traits
    maps/ground_graph.json    the DSG mesh-places layer as a motion graph

Deliberately minimal: one species, trivial trait vectors, no coalitions. The
point is to prove the pipeline drives the solver end to end; traits and
multi-robot allocation are the next layer.
"""

import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import spark_dsg
from dsg_pddl.pddl_utils import lisp_string_to_ast
from plum import dispatch

from omniplanner.omniplanner import (
    MultiRobotWrapper,
    PlanningDomain,
    RobotWrapper,
)
from omniplanner.tsp import LayerPlanner

logger = logging.getLogger(__name__)

TRAIT_DIM = 8
DEFAULT_DURATION = 5.0

# Trait dimensions. Species supply capability, action subtasks demand it, and
# ITAGS matches the two. Only the first four are used; the rest are reserved so
# adding a capability later does not renumber existing vectors.
#   0 ground-mobile   1 air-mobile   2 sensor   3 manipulator
GROUND, AIR, SENSOR, MANIP = 0, 1, 2, 3


def _traits(*dims):
    v = [0] * TRAIT_DIM
    for d in dims:
        v[d] = 1
    return v


def _coalition_traits(requirements):
    """A requirement vector with explicit magnitudes, e.g. {GROUND: 1, AIR: 1}.

    Requirements are met by the SUM of the assigned robots' trait vectors, so a
    magnitude above what any one species carries is what forces a coalition.
    """
    v = [0] * TRAIT_DIM
    for dim, magnitude in requirements.items():
        v[dim] = magnitude
    return v


# Speed is what lets the makespan objective discriminate between platforms that
# are otherwise equally capable: a UAV covers the same distance in half the
# travel time, so an allocator minimising makespan has a reason to send it.
SPECIES = {
    "spot": {"traits": _traits(GROUND, SENSOR), "speed": 1.0, "bounding_radius": 1.0},
    "spot_arm": {
        "traits": _traits(GROUND, SENSOR, MANIP),
        "speed": 1.0,
        "bounding_radius": 1.0,
    },
    "uav": {"traits": _traits(AIR, SENSOR), "speed": 2.0, "bounding_radius": 0.5},
}
DEFAULT_SPECIES = "spot"

# robot_type as declared in omniplanner_plugins.yaml -> species above.
ROBOT_TYPE_SPECIES = {
    "spot": "spot",
    "simulated-spot": "spot",
    "spot-with-manipulator": "spot_arm",
    "simulated-spot-with-manipulator": "spot_arm",
    "uav": "uav",
    "drone": "uav",
    "quadrotor": "uav",
}


def species_for(robot_type, overrides=None, robot_name=None):
    """Map a declared robot_type onto a species, defaulting rather than failing.

    An unknown platform still plans -- it just gets the plain ground profile.
    Refusing to plan because a robot_type is unrecognised would make adding a
    platform a planning outage rather than a missed optimisation.
    """
    # A per-deployment override, keyed by robot name, wins over the declared
    # type. The shared robot registry says what a platform generally is; a
    # particular rig may differ (a sim spot standing in for one with an arm),
    # and that belongs in the planner's config rather than in a registry every
    # other experiment reads.
    if overrides and robot_name and robot_name in overrides:
        chosen = overrides[robot_name]
        if chosen not in SPECIES:
            logger.warning(
                "robot_species override %r for %s is not a known species %s; "
                "falling back to the declared type.",
                chosen,
                robot_name,
                sorted(SPECIES),
            )
        else:
            return chosen
    if not robot_type:
        return DEFAULT_SPECIES
    key = str(robot_type).strip().lower()
    if key in ROBOT_TYPE_SPECIES:
        return ROBOT_TYPE_SPECIES[key]
    logger.warning(
        "Unknown robot_type %r; treating it as %r. Add it to ROBOT_TYPE_SPECIES "
        "if it has different capabilities.",
        robot_type,
        DEFAULT_SPECIES,
    )
    return DEFAULT_SPECIES


VISIT_DURATION = 5.0
INSPECT_DURATION = 6.0
PICK_DURATION = 8.0
PLACE_DURATION = 8.0

# Robot-agnostic by construction, like every GRSTAPS-X domain: no robot objects,
# no robot parameters, no trait predicates. Who performs a task is decided
# downstream by ITAGS from the trait vectors in action_trait_config.json.
#
# Two idioms here are load-bearing, not style:
#   * `is-location` is a static marker giving each action a POSITIVE
#     precondition. Without one the grounder's reachability analysis prunes the
#     action and the whole chain collapses.
#   * Only latching positives are ever written. Negating a predicate absent from
#     :init breaks the SAS translator's binary-variable assumption.
#
# Precedence is not declared anywhere -- it falls out of the chain
# picked -> placed, because a task whose precondition is another's effect can
# only be scheduled after it. That is what gives the MILP scheduler an ordering
# to respect.
DOMAIN_PDDL_TEMPLATE = """(define (domain dsg-visit)
  (:requirements :typing :durative-actions :negative-preconditions)
  (:types location - object)
  (:predicates
    (is-location ?l - location)
    (visited ?l - location)
    (inspected ?l - location)
    (picked ?l - location)
    (placed ?l - location))
  (:durative-action visit-location
    :parameters (?l - location)
    :duration (= ?duration %(visit)s)
    :condition (and
      (at start (is-location ?l))
      (at start (not (visited ?l))))
    :effect (and
      (at end (visited ?l))))
  (:durative-action inspect-object
    :parameters (?l - location)
    :duration (= ?duration %(inspect)s)
    :condition (and
      (at start (is-location ?l))
      (at start (not (inspected ?l))))
    :effect (and
      (at end (inspected ?l))))
%(manipulation)s)
"""

# Pick and place as one indivisible task. Nothing else can be scheduled between
# them, so a robot physically cannot be holding two objects at once -- the
# scheduler already serialises one robot's tasks, and an atomic relocation
# makes that serialisation mean "one object at a time".
RELOCATE_ACTION = """  (:durative-action relocate-object
    :parameters (?l - location)
    :duration (= ?duration %(relocate)s)
    :condition (and
      (at start (inspected ?l))
      (at start (not (placed ?l))))
    :effect (and
      (at end (picked ?l))
      (at end (placed ?l))))"""

# Split, so a schedule shows the pick and the drop separately. Only sound when
# a robot handles one object at a time by some other means -- with several
# relocations it can interleave them and hold two at once.
PICK_PLACE_ACTIONS = """  (:durative-action pick-object
    :parameters (?l - location)
    :duration (= ?duration %(pick)s)
    :condition (and
      (at start (inspected ?l))
      (at start (not (picked ?l))))
    :effect (and
      (at end (picked ?l))))
  (:durative-action place-object
    :parameters (?l - location)
    :duration (= ?duration %(place)s)
    :condition (and
      (at start (picked ?l))
      (at start (not (placed ?l))))
    :effect (and
      (at end (placed ?l))))"""


def _pddl_duration(seconds):
    """Render a duration for PDDL.

    Durations must be CONSTANTS, not fluents: `(= ?duration (cost ...))` trips a
    preprocessing out-of-bounds bug in the solver, which is why task duration is
    per action type rather than per instance.
    """
    return f"{float(seconds):g}"


def render_domain_pddl(durations, atomic_manipulation=True):
    """The mission domain with this run's durations substituted in.

    Generated rather than a module constant so the PDDL and the action-trait
    config are filled from the same numbers. They are read by different parts of
    the solver -- the task planner takes the PDDL duration, the scheduler takes
    the subtask duration -- and if they disagree the schedule silently stops
    matching the plan.
    """
    fill = {
        "visit": _pddl_duration(durations["visit-location"]),
        "inspect": _pddl_duration(durations["inspect-object"]),
        "pick": _pddl_duration(durations["pick-object"]),
        "place": _pddl_duration(durations["place-object"]),
        "relocate": _pddl_duration(
            durations["pick-object"] + durations["place-object"]
        ),
    }
    fill["manipulation"] = (
        RELOCATE_ACTION if atomic_manipulation else PICK_PLACE_ACTIONS
    ) % fill
    return DOMAIN_PDDL_TEMPLATE % fill


@dataclass
class GrstapsDomain(PlanningDomain):
    """Where GRSTAPS-X lives and how to run it.

    `data_root` must be the repo's data dir: the solver resolves
    graph_filepath as /grstapsx/<scenario>/... relative to it.
    """

    # Paths default from the environment so nothing machine-specific has to be
    # written into a tracked config; override per-deployment via ADT4_GRSTAPS_*.
    repo_root: str = os.environ.get(
        "ADT4_GRSTAPS_ROOT", os.path.expanduser("~/grstapsx")
    )
    docker_image: str = "grstapsx:amd64"
    license_path: str = os.environ.get(
        "ADT4_GUROBI_LICENSE", os.path.expanduser("~/gurobi.lic")
    )
    scenario_name: str = "omniplanner_run"
    solver_params_from: str = (
        "wpc_spread"  # scenario to copy itags/scheduler params from
    )
    run_mode: str = "docker"  # "docker" (WLS licence) or "native" (Named-User)
    native_binary: str = os.environ.get(
        "ADT4_GRSTAPS_BINARY",
        os.path.expanduser("~/grstapsx/build-native/grstapsx_example"),
    )
    gurobi_home: str = os.environ.get(
        "ADT4_GUROBI_HOME", os.path.expanduser("~/opt/gurobi1103/linux64")
    )
    conda_prefix: str = os.environ.get(
        "ADT4_GRSTAPS_CONDA_PREFIX", os.path.expanduser("~/miniconda3/envs/grstapsx")
    )
    # How long each task type takes, in seconds. The scheduler places tasks in
    # time from these plus travel, so they are what makes a makespan mean
    # anything. Constants per action type, not per instance -- see
    # _pddl_duration for why.
    visit_duration_s: float = VISIT_DURATION
    inspect_duration_s: float = INSPECT_DURATION
    pick_duration_s: float = PICK_DURATION
    place_duration_s: float = PLACE_DURATION
    # "pddl"  -> domain/problem.pddl through grstapsx_example (task planner runs)
    # "itags" -> itags_input.json straight into the itags binary, which skips
    #            task planning and takes precedence_constraints directly, so an
    #            instruction-level ordering ("visit o2 before picking o42") is
    #            expressible. Both end at the same itags_solution.json.
    # One robot, one object. A relocation is emitted as a single task rather
    # than a pick and a place, so nothing can be scheduled between them and a
    # robot cannot end up holding two objects at once. Set False to see the
    # pick and the drop as separate tasks, at the cost of that guarantee.
    atomic_manipulation: bool = True
    entry: str = "pddl"
    itags_binary: str = os.environ.get(
        "ADT4_ITAGS_BINARY",
        os.path.expanduser("~/grstapsx/build-native/itags"),
    )
    # {robot_name -> species}, overriding what robot_type implies. Use when a
    # specific robot's capabilities differ from its declared platform, e.g.
    # {"hilbert": "spot_arm"} for a sim spot standing in for one with an arm.
    robot_species: Dict[str, str] = field(default_factory=dict)
    forbidden_radius_m: float = 5.0  # matches the PDDL grounder's expansion
    license_retries: int = 4  # WLS token checkout is flaky; see _run_solver
    license_retry_delay_s: float = 3.0


@dataclass
class GrstapsGoal:
    """Symbols the robot must visit, e.g. ["O15", "O18"].

    `constraints` mirrors ConstrainedPddlGoalMsg's ConstraintFact list, so the
    plan-repair flow can drive this planner: forbidden POIs are removed from the
    motion graph outright, which is stronger than the PDDL path's edge dropping
    -- A* cannot route through a vertex that no longer exists.
    """

    goal_points: List[str]
    robot_id: str
    constraints: List[Any] = field(default_factory=list)
    # Symbols to pick up and put down rather than merely visit. These are what
    # make traits bite: pick/place demand a manipulator, so only a spot_arm can
    # be allocated to them however close a plain spot happens to be.
    manipulate_points: List[str] = field(default_factory=list)
    # Symbols needing a COALITION inspection: the inspect-object template has a
    # ground-observer and an aerial-observer subtask, so ITAGS must field two
    # robots of different species at once. Also lengthens the precedence chain
    # to inspect -> pick -> place when a symbol is in both lists.
    inspect_points: List[str] = field(default_factory=list)
    # {object symbol -> destination symbol}. A manipulation with a destination
    # is a relocation: picked where it lies, put down somewhere else. Without
    # one the object is set back down where it was found.
    manipulate_destinations: Dict[str, str] = field(default_factory=dict)
    # Precedence in GRSTAPS-X's own form: [[i, j], ...] meaning task i finishes
    # before task j starts. Indices are into the task list this goal generates
    # -- run once and read the enumeration to find them. `before` constraints
    # are a convenience that compiles down to these same pairs; both end up in
    # the one precedence_constraints array the solver reads.
    extra_precedence: List[List[int]] = field(default_factory=list)
    # {robot_name -> robot_type} as declared in omniplanner_plugins.yaml. Left
    # empty every robot falls back to the default species, which is the old
    # homogeneous behaviour.
    robot_types: Dict[str, str] = field(default_factory=dict)


@dataclass
class GroundedGrstapsProblem:
    scenario_dir: str
    domain: GrstapsDomain
    location_of: Dict[str, str] = field(default_factory=dict)  # pddl name -> dsg symbol
    symbol_of_vertex: Dict[int, str] = field(
        default_factory=dict
    )  # graph id -> dsg symbol
    # {object pddl name -> destination pddl name}. Needed after solving, when
    # the goal is out of scope but a relocation still has to be told where the
    # object goes -- the solver's task name only ever carries the object.
    destination_of: Dict[str, str] = field(default_factory=dict)


@dataclass
class GrstapsPlan:
    """Per-robot ordered tasks plus the schedule GRSTAPS-X computed."""

    makespan: float
    tasks: List[dict]
    agents: List[dict] = field(default_factory=list)
    raw: Any = None
    location_of: Dict[str, str] = field(default_factory=dict)
    symbol_of_vertex: Dict[int, str] = field(default_factory=dict)
    # {task id -> geometry}, see _parameterize_tasks. Empty for a plan built
    # without a DSG, which compiles to navigation only.
    task_geometry: Dict[int, dict] = field(default_factory=dict)
    # [[i, j], ...]: task i finishes before task j starts. The solver's own
    # array, which already merges the intrinsic chains with whatever `before`
    # constraints compiled down to. This is what the executor gates on --
    # timepoints assume the durations the solver was given and stop being true
    # the moment a real action overruns, whereas an ordering does not decay.
    precedence: List[List[int]] = field(default_factory=list)


def _graph_config(vertex):
    """A task/robot position in the form ITAGS expects."""
    return {
        "configuration_type": "graph",
        "graph_type": "euclidean",
        "id": vertex["id"],
        "x": vertex["x"],
        "y": vertex["y"],
    }


def _task(name, traits, duration, vertex, end_vertex=None):
    """One ITAGS task, optionally ending somewhere other than it starts.

    Most tasks are stationary, so terminal == initial. A relocation is not: the
    robot picks an object up where it lies and puts it down elsewhere, and the
    scheduler needs the terminal configuration to charge that carry to the task
    and to know where the robot ends up.

    The name follows the solver's own "<action> <target> :: <role>" convention
    so everything downstream -- task_location, the visited-POI cache, the
    schedule topic -- keeps working unchanged.
    """
    cfg = _graph_config(vertex)
    return {
        "name": name,
        "desired_traits": [float(t) for t in traits],
        "duration": float(duration),
        "mp_index": 0,
        "initial_configuration": cfg,
        "terminal_configuration": _graph_config(end_vertex or vertex),
    }


_VISIT_PREDS = {
    "visited-place",
    "visited-object",
    "visited-poi",
    "at-place",
    "at-object",
}


# Which kind of task a goal predicate asks for, and which argument names the
# symbol. The vocabulary is the multirobot FD domain's, unchanged, so the agent
# prompt needs no new predicates -- only the freedom to use the ones it already
# documents. Index -1 skips the leading robot argument that at-*/holding carry.
_GOAL_PREDS = {
    # predicate            kind          arg index
    "visited-object": ("visit", -1),
    "visited-place": ("visit", -1),
    "visited-poi": ("visit", -1),
    "at-object": ("visit", -1),
    "at-place": ("visit", -1),
    # inspect removes suspicion; `safe` is the same idea stated positively
    "safe": ("inspect", -1),
    "suspicious": ("inspect", -1),  # only meaningful under (not ...)
    # pick leaves the robot holding; place puts the object somewhere
    "holding": ("manipulate", -1),
    "object-in-place": ("manipulate", 1),  # (object-in-place ?o ?p) -- ?o
}


def parse_goal_targets(pddl_goal: str):
    """Split a PDDL goal into visit / inspect / manipulate targets.

    GRSTAPS-X needs to know what KIND of task each symbol wants, because the
    kind selects the trait requirement and the precedence chain. The flat
    "visit everything" reading discards that. Reuses the existing FD predicate
    vocabulary so the agent prompt does not need a new one.

    Everything else stays predefined, exactly as the FD domain file is: species,
    trait vectors and durations are checked-in constants, not something the LLM
    is asked to invent. A goal that says nothing special gets plain visits and
    no precedence -- the defaults do the right thing when the LLM is silent.
    """
    normalised = (pddl_goal or "").replace(")(", ") (")
    try:
        ast = lisp_string_to_ast(normalised)
    except Exception:
        logger.warning("Could not parse PDDL goal %r", pddl_goal)
        return {"visit": [], "inspect": [], "manipulate": [], "destinations": {}}

    # destinations: {object -> place to leave it}, from (object-in-place ?o ?p)
    found = {"visit": [], "inspect": [], "manipulate": [], "destinations": {}}

    def walk(node, negated=False):
        if isinstance(node, str) or not node:
            return
        head = node[0]
        if not isinstance(head, str):
            # ((pred a)) -- an extra pair of parens wraps the clause in a list.
            # Descend rather than choking on an unhashable head.
            for sub in node:
                walk(sub, negated)
            return
        if head in ("and", "or"):
            for sub in node[1:]:
                walk(sub, negated)
            return
        if head == "not":
            for sub in node[1:]:
                walk(sub, not negated)
            return
        entry = _GOAL_PREDS.get(head)
        if entry is None or len(node) < 2:
            return
        kind, idx = entry
        # (not (suspicious o)) asks for an inspection; a bare (suspicious o)
        # asks to MAKE something suspicious, which is not a thing we can plan.
        if head == "suspicious" and not negated:
            logger.warning("Ignoring goal (suspicious %s): not achievable", node[-1])
            return
        # Every other predicate here states something to ACHIEVE, so negating it
        # asks for the opposite -- "do not visit", "do not hold". Nothing in the
        # domain can achieve an absence, and silently planning the positive
        # would do the very thing the goal forbids.
        if negated and head != "suspicious":
            logger.warning(
                "Ignoring goal (not (%s %s)): a negative goal is not achievable. "
                "Drop the `not` to ask for it.",
                head,
                " ".join(str(a) for a in node[1:]),
            )
            return
        try:
            symbol = node[idx]
        except IndexError:
            return
        if isinstance(symbol, str) and symbol not in found[kind]:
            found[kind].append(symbol)
        if head == "object-in-place" and len(node) > 2:
            # (object-in-place ?o ?p) names where the object must END UP. The
            # place task carries it there via its terminal configuration.
            found["destinations"][node[1]] = node[2]

    walk(ast)
    return found


def _visit_targets(pddl_goal: str):
    """Backwards-compatible flat list of every symbol a goal mentions."""
    found = parse_goal_targets(pddl_goal)
    out = []
    for kind in ("visit", "inspect", "manipulate"):
        for s in found[kind]:
            if s not in out:
                out.append(s)
    return out


def enumerate_tasks(goal, durations, vertex_of, atomic_manipulation=True):
    """Goal -> (tasks, {symbol -> [task indices]}, intrinsic precedence pairs).

    This is the job the PDDL task planner does for us on the other path. Our
    domain is thin enough that the mapping is a lookup rather than a search:
    a visit is one task, an inspection is one (coalition) task, and a
    manipulation is pick-then-place.

    The intrinsic ordering that PDDL preconditions used to guarantee -- pick
    before place, inspect before pick -- has to be emitted explicitly here.
    That is the one correctness burden this entry point takes on, so it is
    returned rather than left to the caller to remember.
    """
    tasks: List[dict] = []
    of_symbol: Dict[str, List[int]] = {}
    precedence: List[List[int]] = []

    def add(name, traits, duration, sym, end=None):
        idx = len(tasks)
        end_v = vertex_of.get(end.lower()) if end else None
        if end and end_v is None:
            logger.warning("Destination %s is not in the graph; ignoring it.", end)
        tasks.append(_task(name, traits, duration, vertex_of[sym.lower()], end_v))
        of_symbol.setdefault(sym.lower(), []).append(idx)
        return idx

    inspect_set = {s.lower() for s in (goal.inspect_points or [])}
    manipulate_set = {s.lower() for s in (goal.manipulate_points or [])}

    for sym in goal.goal_points or []:
        if sym.lower() in inspect_set or sym.lower() in manipulate_set:
            continue  # a richer task for this symbol is added below
        add(
            f"visit-location {sym.lower()} :: visit",
            _traits(SENSOR),
            durations["visit-location"],
            sym,
        )

    for sym in goal.inspect_points or []:
        add(
            f"inspect-object {sym.lower()} :: joint_observation",
            _coalition_traits({GROUND: 1, AIR: 1, SENSOR: 2}),
            durations["inspect-object"],
            sym,
        )

    for sym in goal.manipulate_points or []:
        dest = (getattr(goal, "manipulate_destinations", None) or {}).get(sym.lower())
        if atomic_manipulation:
            # One task from "lift it" to "set it down". The scheduler already
            # gives a robot one task at a time, so making the carry atomic is
            # what turns "carries one object" from likely into guaranteed --
            # split into pick and place, a robot can pick a second object
            # before putting the first down.
            first = add(
                f"relocate-object {sym.lower()} :: relocate",
                _traits(MANIP),
                durations["pick-object"] + durations["place-object"],
                sym,
                end=dest,
            )
        else:
            first = add(
                f"pick-object {sym.lower()} :: pick",
                _traits(MANIP),
                durations["pick-object"],
                sym,
            )
            place = add(
                f"place-object {sym.lower()} :: place",
                _traits(MANIP),
                durations["place-object"],
                sym,
                end=dest,
            )
            precedence.append([first, place])
        # An inspection of the same symbol gates the manipulation, matching the
        # PDDL path where it requires (inspected ?l).
        for prior in of_symbol.get(sym.lower(), []):
            if tasks[prior]["name"].startswith("inspect-object"):
                precedence.append([prior, first])

    return tasks, of_symbol, precedence


def ordering_precedence(constraints, of_symbol):
    """`before(a, b)` constraints -> task-index pairs.

    Every task of `a` precedes every task of `b`. Symbol-level rather than
    task-level because that is the granularity an instruction like "visit o2
    first, then pick o42" actually specifies, and it stays unambiguous when a
    symbol expands to several tasks.
    """
    pairs = []
    for c in constraints or []:
        if getattr(c, "predicate", None) != "before":
            continue
        symbols = getattr(c, "symbols", None) or []
        if len(symbols) < 2:
            continue
        first = of_symbol.get(symbols[0].lower(), [])
        second = of_symbol.get(symbols[1].lower(), [])
        if not first or not second:
            logger.warning(
                "Ordering before(%s, %s) names a symbol with no task; ignoring.",
                symbols[0],
                symbols[1],
            )
            continue
        for a in first:
            for b in second:
                pairs.append([a, b])
    return pairs


def task_location(task_name: str):
    """Pull the location symbol out of a GRSTAPS-X task name.

    The solver names tasks "<action> <arg> :: <role>", e.g.
    "visit-location o0 :: visit". Grounding lowercases the DSG symbol to form
    the PDDL location name, and the PDDL planners do the same, so the symbol
    recovered here matches what goal_manager compares its goal against.
    """
    head = (task_name or "").split("::")[0].split()
    return head[-1] if len(head) >= 2 else None


def _mesh_place_graph(dsg):
    """DSG mesh-places layer -> euclidean motion graph + symbol/position lookup."""
    try:
        layer = dsg.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        layer = dsg.get_layer(20)

    vertices, index_of, pos_of, symbol_of = [], {}, {}, {}
    for node in layer.nodes:
        sym = node.id.str(True)
        p = node.attributes.position
        index_of[sym] = len(vertices)
        pos_of[sym] = (float(p[0]), float(p[1]))
        symbol_of[len(vertices)] = sym
        vertices.append({"id": len(vertices), "x": float(p[0]), "y": float(p[1])})

    seen, edges = set(), []
    for node in layer.nodes:
        ia = index_of[node.id.str(True)]
        for nid in node.siblings():
            other = dsg.get_node(nid)
            ib = index_of.get(other.id.str(True))
            if ib is None or ib == ia:
                continue
            key = (min(ia, ib), max(ia, ib))
            if key in seen:
                continue
            seen.add(key)
            va, vb = vertices[ia], vertices[ib]
            edges.append(
                {
                    "cost": math.hypot(va["x"] - vb["x"], va["y"] - vb["y"]),
                    "vertex_a": ia,
                    "vertex_b": ib,
                }
            )

    graph = {
        "configuration_type": "graph",
        "graph_type": "euclidean",
        "euclidean_graph_type": "singular",
        "is_complete": False,
        "edges": edges,
        "vertices": vertices,
    }
    # symbol_of inverts the vertex numbering the solver works in, so a returned
    # route can be read back as DSG place symbols instead of bare coordinates.
    return graph, vertices, pos_of, symbol_of


def _resolve_symbol(dsg, symbol):
    """DSG position of a constraint symbol, trying both cases."""
    for candidate in (symbol, symbol.upper(), symbol.lower()):
        xy = _symbol_position(dsg, candidate)
        if xy is not None:
            return np.array(xy)
    return None


def _forbidden_sets(dsg, constraints):
    """Split constraints into forbidden points and forbidden edge endpoints.

    Mirrors the PDDL grounder's `_extract_forbidden_sets`, which handles both
    predicates. Handling only forbidden-poi here would let a forbidden-edge
    constraint look accepted while having no effect at all.
    """
    points, edges = [], []
    for c in constraints or []:
        predicate = getattr(c, "predicate", None)
        symbols = getattr(c, "symbols", None) or []
        if predicate == "forbidden-poi" and symbols:
            xy = _resolve_symbol(dsg, symbols[0])
            if xy is None:
                logger.warning(
                    "forbidden-poi %s not found in the DSG; ignoring", symbols[0]
                )
            else:
                points.append(xy)
        elif predicate == "forbidden-edge" and len(symbols) >= 2:
            a = _resolve_symbol(dsg, symbols[0])
            b = _resolve_symbol(dsg, symbols[1])
            if a is None or b is None:
                logger.warning(
                    "forbidden-edge %s-%s: endpoint not in the DSG; ignoring",
                    symbols[0],
                    symbols[1],
                )
            else:
                edges.append((a, b))
    return points, edges


def _prune_graph(graph, forbidden_xy, forbidden_edges_xy, radius, distance_fn=None):
    """Isolate vertices near a forbidden point and cut forbidden edges.

    Vertex ids are indices into the vertex list, so removing entries would
    renumber everything; instead the vertices stay and only their edges go,
    leaving them isolated and therefore unreachable.

    `distance_fn(a_xy, b_xy)` measures the exclusion radius. It defaults to
    euclidean, but the caller passes navigable path distance so that this
    planner and the PDDL grounder agree on what a radius means -- a place that
    is metrically close but only reachable the long way round should not be
    excluded.
    """
    banned = set()
    if forbidden_xy:
        for v in graph["vertices"]:
            for f in forbidden_xy:
                # Euclidean distance is a lower bound on path distance, so a
                # vertex further away than the radius in a straight line cannot
                # be within it by any route. Cheap prefilter before the
                # expensive graph search.
                if math.hypot(v["x"] - f[0], v["y"] - f[1]) >= radius:
                    continue
                d = (
                    distance_fn(f, (v["x"], v["y"]))
                    if distance_fn is not None
                    else math.hypot(v["x"] - f[0], v["y"] - f[1])
                )
                if d < radius:
                    banned.add(v["id"])
                    break

    cut_pairs = set()
    for a, b in forbidden_edges_xy or []:
        va = _nearest_vertex(graph["vertices"], float(a[0]), float(a[1]))
        vb = _nearest_vertex(graph["vertices"], float(b[0]), float(b[1]))
        if va["id"] != vb["id"]:
            cut_pairs.add((min(va["id"], vb["id"]), max(va["id"], vb["id"])))

    if not banned and not cut_pairs:
        return graph, 0

    before = len(graph["edges"])
    if cut_pairs:
        graph["edges"] = [
            e
            for e in graph["edges"]
            if (min(e["vertex_a"], e["vertex_b"]), max(e["vertex_a"], e["vertex_b"]))
            not in cut_pairs
        ]
        logger.info(
            "Constraints: cut %d forbidden edge(s)", before - len(graph["edges"])
        )
    if not banned:
        return graph, 0
    before = len(graph["edges"])
    graph["edges"] = [
        e
        for e in graph["edges"]
        if e["vertex_a"] not in banned and e["vertex_b"] not in banned
    ]
    logger.info(
        "Constraints: isolated %d vertices within %.1f m of %d forbidden POI(s); "
        "dropped %d motion-graph edges",
        len(banned),
        radius,
        len(forbidden_xy),
        before - len(graph["edges"]),
    )
    return graph, len(banned)


def _nearest_vertex(vertices, x, y):
    return min(vertices, key=lambda v: math.hypot(v["x"] - x, v["y"] - y))


def _find_node(dsg, symbol):
    """Any DSG node by its symbol string, case-insensitively.

    The DSG spells object symbols uppercase (O16) but PDDL requires lowercase,
    so a goal arriving from the repair flow names o16 while the graph holds O16.
    Matching exactly would make every goal from that path unresolvable.
    """
    wanted = (symbol or "").lower()
    for layer_name in ("OBJECTS", "MESH_PLACES", "PLACES", "ROOMS"):
        try:
            layer = dsg.get_layer(getattr(spark_dsg.DsgLayers, layer_name))
        except Exception:
            continue
        for node in layer.nodes:
            if node.id.str(True).lower() == wanted:
                return node
    return None


def _symbol_position(dsg, symbol):
    """2D position of a symbol, for the motion graph."""
    node = _find_node(dsg, symbol)
    if node is None:
        return None
    p = node.attributes.position
    return float(p[0]), float(p[1])


def _symbol_position_3d(dsg, symbol):
    """3D position of a symbol, which is what Pick/Place/Gaze act on."""
    node = _find_node(dsg, symbol)
    if node is None:
        return None
    return np.array(node.attributes.position, dtype=float)


def _semantic_label(dsg, symbol):
    """Object class for a symbol, as Pick/Place's object_class expects.

    The same labelspace lookup DsgContextProvider does for the fast-downward
    path, done here because the DSG is in scope while a plan is parameterised
    and is not by the time it is compiled.
    """
    node = _find_node(dsg, symbol)
    if node is None:
        return ""
    try:
        labelspace = dsg.get_labelspace(node.layer.layer, node.layer.partition)
        return labelspace.get_node_category(node) or ""
    except Exception:
        return ""


# GRSTAPS-X task name -> the kind of executor action it becomes. Names follow
# the solver's own "<action> <target> :: <role>" convention.
TASK_ACTION_KINDS = {
    "visit-location": "visit",
    "inspect-object": "inspect",
    "relocate-object": "relocate",
    "pick-object": "pick",
    "place-object": "place",
}


def _parameterize_tasks(dsg, tasks, location_of, destination_of):
    """Geometry each scheduled task needs before it can become a robot action.

    The solver returns symbols and timings; every coordinate an executor acts on
    still has to come from the DSG. This mirrors `dsg_pddl_planning`'s
    `parameterize_*` helpers: in both pipelines the symbolic planner produces no
    geometry, and a later pass supplies it.

    The robot's own pose is deliberately absent. It is the end of whichever leg
    that robot travelled to reach the task, which differs per robot for a
    coalition task, so compile_plan supplies it. Everything here is a property
    of the task and is shared by every robot assigned to it.
    """
    layer_planner = LayerPlanner(dsg, spark_dsg.DsgLayers.MESH_PLACES)
    geometry = {}
    for t in tasks or []:
        name = t.get("name", "")
        head = name.split()
        kind = TASK_ACTION_KINDS.get(head[0]) if head else None
        if kind is None:
            logger.warning("Task %r has no executor action; skipping it", name)
            continue
        target = task_location(name)
        position = _symbol_position_3d(dsg, location_of.get(target, target))
        if position is None:
            logger.warning(
                "Task %r targets %s, which is not in the DSG; skipping it", name, target
            )
            continue

        entry = {"action": kind, "target": target, "object_point": position}
        if kind != "visit":
            entry["object_class"] = _semantic_label(
                dsg, location_of.get(target, target)
            )

        if kind in ("relocate", "place"):
            dest = destination_of.get(target)
            dest_point = (
                _symbol_position_3d(dsg, location_of.get(dest, dest)) if dest else None
            )
            if dest_point is None:
                # No destination means "put it back where it was found", which
                # needs neither a carry nor a second position.
                pass
            elif kind == "place":
                # place-object is named after the object but happens at the
                # destination, so that is where the robot sets it down.
                entry["object_point"] = dest_point
            else:
                entry["dest_point"] = dest_point
                # An atomic relocation carries the object inside a single task,
                # so the solver charges the carry to that task and never reports
                # it as a transition. Without a path here the executor would be
                # told to put the object down while still standing where it
                # picked it up.
                entry["carry_path"] = np.array(
                    [
                        [float(q[0]), float(q[1])]
                        for q in layer_planner.get_external_path(
                            position[:2], dest_point[:2]
                        )
                    ],
                    dtype=float,
                )
        geometry[t.get("id")] = entry
    return geometry


@dispatch
def ground_problem(
    domain: GrstapsDomain,
    dsg: Any,
    robot_states: dict,
    goal: GrstapsGoal,
    feedback: Any = None,
) -> Any:
    # Return type is deliberately Any: a single-robot fleet yields a
    # RobotWrapper and a real fleet a MultiRobotWrapper, and plum enforces a
    # declared return type strictly enough that naming either one rejects the
    # other at runtime.
    logger.info("Grounding GRSTAPS-X problem for %s", goal.robot_id)

    pose = robot_states.get(goal.robot_id)
    if pose is None:
        raise ValueError(
            f"No pose for robot {goal.robot_id}; cannot place its start vertex."
        )

    graph, vertices, _, symbol_of_vertex = _mesh_place_graph(dsg)
    if not vertices:
        raise ValueError("DSG has no mesh-places layer; nothing to build a graph from.")

    forbidden_pts, forbidden_edges = _forbidden_sets(
        dsg, getattr(goal, "constraints", None)
    )
    path_distance = None
    if forbidden_pts:
        # Same measure the PDDL grounder uses, so a 5 m exclusion means the same
        # thing to both planners. Built lazily -- it is only needed when a
        # forbidden POI is actually present.
        try:
            layer_planner = LayerPlanner(dsg, spark_dsg.DsgLayers.MESH_PLACES)

            def path_distance(a, b):
                return layer_planner.get_external_distance(
                    np.array([a[0], a[1]]), np.array([b[0], b[1]])
                )
        except Exception as exc:
            logger.warning(
                "Could not build a path-distance planner (%s); falling back to "
                "euclidean exclusion, which may over-prune around walls.",
                exc,
            )
            path_distance = None

    graph, _ = _prune_graph(
        graph,
        forbidden_pts,
        forbidden_edges,
        domain.forbidden_radius_m,
        path_distance,
    )

    scenario_dir = os.path.join(
        domain.repo_root, "data", "grstapsx", domain.scenario_name
    )
    os.makedirs(os.path.join(scenario_dir, "maps"), exist_ok=True)
    with open(os.path.join(scenario_dir, "maps", "ground_graph.json"), "w") as fo:
        json.dump(graph, fo)

    # Goal symbols become the PDDL location names directly (lowercased, as the
    # PDDL planners do), so a GRSTAPS-X plan names the same DSG symbols a
    # fast-downward plan does and the two stay comparable. Each is snapped to
    # its nearest motion-graph vertex.
    nodes, location_of, loc_names = {}, {}, []
    vertex_of = {}  # pddl name -> its snapped motion-graph vertex
    missing = []
    manipulate = {s.lower() for s in (getattr(goal, "manipulate_points", None) or [])}
    inspect_targets = {s.lower() for s in (getattr(goal, "inspect_points", None) or [])}
    # A manipulation target still has to be reached, so it is a location like
    # any other; only its goal predicate differs.
    _seen_lower = set()
    all_targets = []
    for s in (
        list(goal.goal_points)
        + list(getattr(goal, "manipulate_points", None) or [])
        + list(getattr(goal, "inspect_points", None) or [])
        + list((getattr(goal, "manipulate_destinations", None) or {}).values())
    ):
        if s.lower() not in _seen_lower:
            _seen_lower.add(s.lower())
            all_targets.append(s)
    for sym in all_targets:
        xy = _symbol_position(dsg, sym)
        if xy is None:
            # A symbol the graph does not have is the agent's mistake, not a
            # reason to take the planner down: raising here propagates out of
            # the subscription callback and kills the whole omniplanner node,
            # losing every other robot's planning with it. Drop it loudly and
            # plan for the rest.
            missing.append(sym)
            continue
        v = _nearest_vertex(vertices, *xy)
        name = sym.lower()
        loc_names.append(name)
        location_of[name] = sym
        # Coordinates must be the VERTEX's, not the object's: the solver treats
        # x/y and id as the same point, and a mismatch makes allocation fail
        # with "Searched the entire space, but couldn't find a solution".
        nodes[name] = {"x": v["x"], "y": v["y"], "id": v["id"]}
        vertex_of[name] = v

    if missing:
        logger.warning(
            "Goal symbol(s) %s are not in the DSG; ignoring them. Planning for "
            "%d of %d requested target(s).",
            ", ".join(missing),
            len(loc_names),
            len(all_targets),
        )
    if not loc_names:
        raise ValueError(
            "None of the requested goal symbols "
            f"({', '.join(all_targets) or 'none given'}) are in the DSG; "
            "nothing to plan for."
        )

    # Every robot with a pose joins the fleet, so GRSTAPS-X can actually
    # allocate. Robots without one are skipped rather than failing the request.
    fleet = []
    for robot_id, robot_pose in robot_states.items():
        if robot_pose is None:
            logger.warning("Skipping robot %s: no pose available", robot_id)
            continue
        v = _nearest_vertex(vertices, float(robot_pose[0]), float(robot_pose[1]))
        # The allocator looks each start up in `nodes`; without it the config
        # fails to parse with "json is missing field 'x'".
        nodes[f"_start_{v['id']}"] = {"x": v["x"], "y": v["y"], "id": v["id"]}
        fleet.append((robot_id, v))

    if not fleet:
        raise ValueError("No robot in robot_states has a pose; nothing to plan for.")

    durations = {
        "visit-location": domain.visit_duration_s,
        "inspect-object": domain.inspect_duration_s,
        "pick-object": domain.pick_duration_s,
        "place-object": domain.place_duration_s,
    }
    with open(os.path.join(scenario_dir, "domain.pddl"), "w") as fo:
        fo.write(render_domain_pddl(durations, domain.atomic_manipulation))

    objs = " ".join(loc_names)
    init_lines = [f"    (is-location {n})" for n in loc_names]
    # pick-object requires (inspected ?l), which makes the full chain
    # inspect -> pick -> place available. Pre-satisfying it in :init for
    # locations nobody asked to inspect keeps a plain pick/place two steps long
    # instead of silently forcing a coalition inspection -- and a fleet with no
    # UAV can still manipulate.
    init_lines += [
        f"    (inspected {n})" for n in loc_names if n not in inspect_targets
    ]
    init = "\n".join(init_lines)

    # Each goal predicate asks for the last link of the chain it needs, since
    # (placed x) already implies (picked x) and, when requested, (inspected x).
    def _goal_for(n):
        if n in manipulate:
            return f"      (placed {n})"
        if n in inspect_targets:
            return f"      (inspected {n})"
        return f"      (visited {n})"

    goal_lines = "\n".join(_goal_for(n) for n in loc_names)
    with open(os.path.join(scenario_dir, "problem.pddl"), "w") as fo:
        fo.write(
            f"(define (problem omniplanner_grstaps)\n"
            f"  (:domain dsg-visit)\n"
            f"  (:objects\n    {objs} - location\n  )\n"
            f"  (:init\n{init}\n  )\n"
            f"  (:goal\n    (and\n{goal_lines}\n    )\n  )\n"
            f"  (:metric minimize (total-time))\n)\n"
        )

    template = json.load(
        open(
            os.path.join(
                domain.repo_root,
                "data",
                "grstapsx",
                domain.solver_params_from,
                "action_trait_config.json",
            )
        )
    )
    # Only emit species the fleet actually uses -- an unused species with a
    # manipulator would let the allocator believe a capability is available.
    robot_species = {
        robot_id: species_for(
            (goal.robot_types or {}).get(robot_id), domain.robot_species, robot_id
        )
        for robot_id, _ in fleet
    }
    used = sorted(set(robot_species.values()))
    logger.info(
        "Fleet: %s",
        ", ".join(f"{r}={s}" for r, s in sorted(robot_species.items())),
    )

    mp = json.loads(json.dumps(template["motion_planners"][0]))
    mp["environment_parameters"]["graph_filepath"] = (
        f"/grstapsx/{domain.scenario_name}/maps/ground_graph.json"
    )
    config = {
        "_comment": f"generated by omniplanner for robot {goal.robot_id}",
        "motion_planners": [mp],
        "species": [
            {
                "name": name,
                "traits": SPECIES[name]["traits"],
                "bounding_radius": SPECIES[name]["bounding_radius"],
                "speed": SPECIES[name]["speed"],
                "mp_index": 0,
            }
            for name in used
        ],
        "robots": [
            {
                "name": robot_id,
                "species": robot_species[robot_id],
                "initial_configuration": {
                    "configuration_type": "graph",
                    "graph_type": "euclidean",
                    "id": v["id"],
                },
            }
            for robot_id, v in fleet
        ],
        "itags_parameters": template["itags_parameters"],
        "scheduler_parameters": template["scheduler_parameters"],
        "nodes": nodes,
        # base_traits is what a subtask DEMANDS; species traits are what a robot
        # SUPPLIES. visit demands only a sensor, which every species has, so any
        # platform can take it and the choice falls to speed. pick/place demand
        # a manipulator, so only spot_arm is eligible however close another
        # robot happens to be.
        #
        # An all-zero requirement is NOT a valid way to say "anyone": ITAGS
        # fails before motion planning with "Request for time from unknown time
        # 'motion_planning_time'", because a zero requirement vector gives its
        # allocation heuristic nothing to normalise against.
        "action_templates": {
            "visit-location": {
                "subtasks": [
                    {
                        "role": "visit",
                        "base_traits": _traits(SENSOR),
                        "duration": durations["visit-location"],
                        "location": "arg0",
                    }
                ]
            },
            # A true coalition: ONE subtask whose requirement no single robot
            # meets. Trait vectors ADD UP across the robots assigned to a
            # subtask (data/README.md), so requiring ground>=1, air>=1 and
            # sensing>=2 forces a mixed ground+air team on the same task at the
            # same time -- e.g. spot [1,0,1,0] + uav [0,1,1,0] = [1,1,2,0].
            #
            # Splitting this into two subtasks instead would produce two
            # INDEPENDENTLY SCHEDULED tasks, one per role, which is a different
            # (and weaker) thing: each role filled, but not concurrently.
            "inspect-object": {
                "subtasks": [
                    {
                        "role": "joint_observation",
                        "base_traits": _coalition_traits(
                            {GROUND: 1, AIR: 1, SENSOR: 2}
                        ),
                        "duration": durations["inspect-object"],
                        "location": "arg0",
                    },
                ]
            },
            "relocate-object": {
                "subtasks": [
                    {
                        "role": "relocate",
                        "base_traits": _traits(MANIP),
                        "duration": durations["pick-object"]
                        + durations["place-object"],
                        "location": "arg0",
                    }
                ]
            },
            "pick-object": {
                "subtasks": [
                    {
                        "role": "pick",
                        "base_traits": _traits(MANIP),
                        "duration": durations["pick-object"],
                        "location": "arg0",
                    }
                ]
            },
            "place-object": {
                "subtasks": [
                    {
                        "role": "place",
                        "base_traits": _traits(MANIP),
                        "duration": durations["place-object"],
                        "location": "arg0",
                    }
                ]
            },
        },
    }
    with open(os.path.join(scenario_dir, "action_trait_config.json"), "w") as fo:
        json.dump(config, fo, indent=1)

    if domain.entry == "itags":
        # The ITAGS entry point takes already-enumerated tasks, so the PDDL
        # above is not read at all -- it is still written because it documents
        # the same problem in a form a human (and the other entry point) can
        # check against.
        tasks, of_symbol, precedence = enumerate_tasks(
            goal, durations, vertex_of, domain.atomic_manipulation
        )
        if not tasks:
            raise ValueError("No tasks to allocate; nothing to plan for.")
        user_pairs = ordering_precedence(getattr(goal, "constraints", None), of_symbol)
        raw_pairs = []
        for pair in getattr(goal, "extra_precedence", None) or []:
            a, b = int(pair[0]), int(pair[1])
            if not (0 <= a < len(tasks) and 0 <= b < len(tasks)):
                logger.warning(
                    "Precedence pair %s is out of range for %d task(s); ignoring.",
                    pair,
                    len(tasks),
                )
                continue
            raw_pairs.append([a, b])
        user_pairs += raw_pairs
        if user_pairs:
            logger.info(
                "Ordering added %d precedence pair(s) (%d from before(), %d given directly)",
                len(user_pairs),
                len(user_pairs) - len(raw_pairs),
                len(raw_pairs),
            )
        itags_input = {
            "tasks": tasks,
            # Both kinds of ordering end up in the same list: the intrinsic
            # chain the PDDL preconditions used to guarantee, plus whatever the
            # instruction asked for. ITAGS cannot tell them apart, and does not
            # need to.
            "precedence_constraints": precedence + user_pairs,
            "plan_task_indices": list(range(len(tasks))),
            "species": config["species"],
            # Robots need full coordinates here. On the PDDL path the separate
            # `nodes` block supplies them and the config carries only an id;
            # the ITAGS input has no such block, so an id-only configuration
            # fails to deserialise with "json is missing field 'x'".
            "robots": [
                {
                    "name": robot_id,
                    "species": robot_species[robot_id],
                    "initial_configuration": _graph_config(v),
                }
                for robot_id, v in fleet
            ],
            "motion_planners": config["motion_planners"],
            "itags_parameters": config["itags_parameters"],
            "scheduler_parameters": config["scheduler_parameters"],
        }
        with open(os.path.join(scenario_dir, "itags_input.json"), "w") as fo:
            json.dump(itags_input, fo, indent=1)
        logger.info(
            "Wrote ITAGS input: %d task(s), %d precedence pair(s) (%d intrinsic, %d from the instruction)",
            len(tasks),
            len(precedence) + len(user_pairs),
            len(precedence),
            len(user_pairs),
        )

    logger.info(
        "Wrote GRSTAPS-X scenario to %s (%d vertices, %d locations)",
        scenario_dir,
        len(vertices),
        len(loc_names),
    )
    grounded = GroundedGrstapsProblem(
        scenario_dir=scenario_dir,
        domain=domain,
        location_of=location_of,
        symbol_of_vertex=symbol_of_vertex,
        destination_of={
            str(k).lower(): str(v).lower()
            for k, v in (goal.manipulate_destinations or {}).items()
        },
    )

    # Wrap like the PDDL grounders do, because compile_plan dispatches on the
    # wrapper to decide how to reach the adaptors. Returning a bare problem
    # hands the whole adaptor dict to compile_plan, which then fails on
    # adaptor.name.
    #
    # A single-robot fleet stays a RobotWrapper so the existing single-robot
    # path is untouched; a real fleet becomes a MultiRobotWrapper, whose
    # compile dispatch splits the solved plan by agent. Without this only one
    # robot's assignment was ever published, which silently discarded the
    # allocation the solver had just computed.
    fleet_names = [robot_id for robot_id, _ in fleet]
    if len(fleet_names) <= 1:
        return RobotWrapper(fleet_names[0] if fleet_names else goal.robot_id, grounded)

    wrapper = MultiRobotWrapper(fleet_names, grounded)
    # GRSTAPS-X uses the real robot names throughout the scenario config and
    # echoes them back in agents[].name, so inner and outer names coincide and
    # the remap is the identity.
    for name in fleet_names:
        wrapper.set_name_remap(name, name)
    return wrapper


def _solver_command(d: GrstapsDomain, rel: str):
    """Build the solver invocation for whichever run mode is configured.

    docker : needs a WLS licence (node-locked ones are bound to a HOSTID and
             fail inside a container), so it is capped by WLS concurrent
             sessions and needs network at solve time.
    native : runs a locally built binary, which a Named-User licence supports
             with no session cap and no network once installed.
    """
    args = [
        f"{rel}/domain.pddl",
        f"{rel}/problem.pddl",
        f"{rel}/action_trait_config.json",
    ]
    if d.run_mode == "native":
        return [d.native_binary] + args
    if d.run_mode != "docker":
        raise ValueError(
            f"Unknown run_mode {d.run_mode!r}; expected 'docker' or 'native'."
        )
    return [
        "docker",
        "run",
        "--rm",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{d.repo_root}:/ws/grstapsx",
        "-v",
        f"{d.license_path}:/opt/gurobi/gurobi.lic:ro",
        d.docker_image,
        "./build/grstapsx_example",
    ] + args


def _solver_env(d: GrstapsDomain):
    """Environment for the solver process.

    The native binary links libgurobi110.so and the conda toolchain, so it needs
    LD_LIBRARY_PATH set or it dies with "error while loading shared libraries".
    Docker carries its own environment, so nothing to add there.
    """
    if d.run_mode != "native":
        return None
    env = dict(os.environ)
    lib_dirs = [os.path.join(d.gurobi_home, "lib")]
    if d.conda_prefix:
        lib_dirs.append(os.path.join(d.conda_prefix, "lib"))
    if env.get("LD_LIBRARY_PATH"):
        lib_dirs.append(env["LD_LIBRARY_PATH"])
    env["GUROBI_HOME"] = d.gurobi_home
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)
    env["GRB_LICENSE_FILE"] = d.license_path
    return env


def _run_solver(problem: GroundedGrstapsProblem):
    """Invoke the solver binary, as pddl_planning does for fast-downward."""
    d = problem.domain
    rel = os.path.relpath(problem.scenario_dir, d.repo_root)
    cmd = _solver_command(d, rel)
    # Gurobi uses a Web License Service: every solve checks a token out over the
    # network, and that checkout is flaky under repeated use -- observed ~30%
    # success at 25s spacing, failing with a bare "No Gurobi license" (10009) or
    # "Too many sessions" (10030). Neither is a property of the problem, so
    # retry rather than reporting the plan as unsolvable.
    last = None
    for attempt in range(1, d.license_retries + 1):
        logger.warning("Calling (attempt %d): %s", attempt, " ".join(cmd))
        last = subprocess.run(
            cmd, cwd=d.repo_root, capture_output=True, text=True, env=_solver_env(d)
        )
        logger.warning("Return code: %s", last.returncode)
        if last.returncode == 0:
            return last
        output = (last.stdout or "") + (last.stderr or "")
        if "10009" not in output and "10030" not in output:
            return last  # a real planning failure; retrying will not help
        if attempt < d.license_retries:
            logger.warning(
                "Gurobi licence checkout failed; retrying in %.1fs",
                d.license_retry_delay_s,
            )
            time.sleep(d.license_retry_delay_s)
    return last


def _solution_dir(d: GrstapsDomain, scenario: str):
    """Where the solver writes its results.

    It writes under its own build tree, which differs by run mode: the docker
    image builds into build/, while the native build lives in build-native/.
    Looking in the wrong one makes a successful solve read as "no solution".
    """
    if d.run_mode == "native":
        build_dir = os.path.dirname(os.path.abspath(d.native_binary))
    else:
        build_dir = os.path.join(d.repo_root, "build")
    return os.path.join(build_dir, "solutions", scenario, "problem")


def _run_itags(problem: GroundedGrstapsProblem):
    """Run the itags binary on our generated problem inputs.

    Skips PDDL task planning entirely: the tasks and their ordering are already
    in the input file. The output is the same itags_solution.json the other
    entry point produces, so nothing downstream changes.
    """
    d = problem.domain
    in_path = os.path.join(problem.scenario_dir, "itags_input.json")
    out_path = os.path.join(problem.scenario_dir, "itags_solution.json")
    # The CLI requires the output path NOT to exist, and a stale file would be
    # read back as this run's answer.
    if os.path.exists(out_path):
        os.remove(out_path)
    if not os.path.exists(d.itags_binary):
        raise FileNotFoundError(
            f"itags binary not found at {d.itags_binary}. Build it "
            "(cmake target `itags`) or set ADT4_ITAGS_BINARY."
        )
    cmd = [d.itags_binary, in_path, out_path]
    logger.warning("Calling: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=d.repo_root, capture_output=True, text=True, env=_solver_env(d)
    )
    logger.warning("Return code: %s", result.returncode)
    return result, out_path


@dispatch
def make_plan(problem: GroundedGrstapsProblem, map_context: Any) -> GrstapsPlan:
    if problem.domain.entry == "itags":
        result, solution_file = _run_itags(problem)
    else:
        solution_dir = _solution_dir(problem.domain, problem.domain.scenario_name)
        # a stale solution would be silently reported as this run's answer
        shutil.rmtree(solution_dir, ignore_errors=True)

        result = _run_solver(problem)
        solution_file = os.path.join(solution_dir, "itags_files", "itags_solution.json")
    if not os.path.exists(solution_file):
        # The solver logs to stdout, not stderr, and reports licensing problems
        # only there -- surface the real reason rather than an empty stderr.
        combined = (result.stdout or "") + (result.stderr or "")
        reasons = [
            ln.strip()
            for ln in combined.splitlines()
            if "error]" in ln
            or "GRBException" in ln
            or "couldn't find a solution" in ln
        ]
        detail = reasons[-1] if reasons else (result.stderr or combined)[-300:]
        raise Exception(
            f"GRSTAPS-X produced no solution (exit {result.returncode}): {detail}"
        )

    with open(solution_file) as fo:
        payload = json.load(fo)

    # itags_solution.json is {"solution": {...}, "statistics": {...}}
    solution = payload.get("solution", payload)
    makespan = float(solution.get("makespan", 0.0))
    tasks = solution.get("tasks", [])
    agents = solution.get("agents", [])
    logger.info(
        "GRSTAPS-X solved: makespan=%.3f, %d task(s), %d agent(s)",
        makespan,
        len(tasks),
        len(agents),
    )
    # Log the schedule itself: the makespan alone does not say who does what,
    # in which order, or when -- which is the whole point of using this planner.
    for t in sorted(tasks, key=lambda t: t.get("start_timepoint", 0.0)):
        logger.info(
            "  %-28s %7.2f -> %7.2f s  coalition=%s",
            t.get("name", "?"),
            t.get("start_timepoint", 0.0),
            t.get("finish_timepoint", 0.0),
            t.get("coalition"),
        )
    symbol_of_vertex = problem.symbol_of_vertex
    by_id = {t.get("id"): t for t in tasks}
    for a in agents:
        legs = a.get("transitions") or []
        n_pts = sum(len(tr.get("path") or []) for tr in legs)
        logger.info(
            "  robot %s: task order %s, %d leg(s), %d waypoint(s)",
            a.get("name"),
            a.get("individual_plan"),
            len(legs),
            n_pts,
        )

        # The symbolic plan: which target each scheduled task goes to, in the
        # order this robot executes them. individual_plan holds task ids, so
        # this is the solver's actual sequencing, not a re-derivation.
        for step, task_id in enumerate(a.get("individual_plan") or []):
            t = by_id.get(task_id) or {}
            loc = task_location(t.get("name", ""))
            logger.info(
                "      %d. %-22s -> %-6s (%s)  %.2f -> %.2f s",
                step + 1,
                t.get("name", "?"),
                loc or "?",
                problem.location_of.get(loc, "?"),
                t.get("start_timepoint", 0.0),
                t.get("finish_timepoint", 0.0),
            )

        # The metric route, named by the DSG places it threads through. This is
        # what the executor actually follows, so seeing it as t#### symbols is
        # how you tell a sensible route from one that cuts through a wall.
        for leg_i, tr in enumerate(legs):
            route = []
            for cfg in tr.get("path") or []:
                sym = symbol_of_vertex.get(cfg.get("id"))
                label = sym if sym else f"({cfg.get('x'):.1f},{cfg.get('y'):.1f})"
                if not route or route[-1] != label:
                    route.append(label)
            if not route:
                continue
            logger.info(
                "      leg %d (%d pts): %s", leg_i + 1, len(route), " -> ".join(route)
            )

    return GrstapsPlan(
        makespan=makespan,
        tasks=tasks,
        agents=agents,
        raw=payload,
        location_of=problem.location_of,
        symbol_of_vertex=symbol_of_vertex,
        task_geometry=_parameterize_tasks(
            map_context, tasks, problem.location_of, problem.destination_of
        ),
        precedence=[
            list(pair) for pair in solution.get("precedence_constraints") or []
        ],
    )
