"""The plan-time component restriction must also fire for multirobot problems.

``make_plan`` restricts its waypoint-expansion ``LayerPlanner`` to the
component reachable from the robot start, so a goto endpoint near an island
place can't raise ``NetworkXNoPath``. It looked the start up as
``symbols["pstart"]`` -- but the multirobot grounding keys one start per robot
as ``pstart{robot_id}``, so the lookup missed and the restriction silently
no-opped for every multirobot plan.

    PYTHONPATH=src python -m pytest tests/test_multirobot_start_component.py
"""

import numpy as np
import spark_dsg
from dsg_pddl import dsg_pddl_planning
from dsg_pddl.dsg_pddl_planning import make_plan
from dsg_pddl.pddl_grounding import GroundedPddlProblem, PddlDomain, PddlSymbol
from test_pddl_island_places import build_island_dsg

from omniplanner.tsp import LayerPlanner


class _Domain(PddlDomain):
    def __init__(self, name):
        self.domain_name = name


def _grounded(domain_name, symbols):
    problem = GroundedPddlProblem.__new__(GroundedPddlProblem)
    problem.domain = _Domain(domain_name)
    problem.problem_str = ""
    problem.symbols = symbols
    return problem


def _run(monkeypatch, domain_name, symbols, action):
    seen = {}

    def fake_solve(_problem):
        return [action]

    def fake_parameterize(layer_planner, _symbols, _action):
        seen["planner"] = layer_planner
        return [np.zeros(2)], np.zeros(2)

    monkeypatch.setattr(dsg_pddl_planning, "solve_pddl", fake_solve)
    monkeypatch.setattr(
        dsg_pddl_planning, "parameterize_goto_poi_multirobot", fake_parameterize
    )
    monkeypatch.setattr(dsg_pddl_planning, "parameterize_goto_poi", fake_parameterize)
    make_plan(_grounded(domain_name, symbols), build_island_dsg())
    return seen["planner"]


def _start(name, xy):
    return PddlSymbol(name, "place", ["at-poi"], position=np.array(xy, dtype=float))


def _unrestricted_size():
    planner = LayerPlanner(build_island_dsg(), spark_dsg.DsgLayers.MESH_PLACES)
    return len(planner.node_ids)


def test_island_fixture_has_an_unreachable_place():
    assert _unrestricted_size() == 3


def test_single_robot_start_still_restricts(monkeypatch):
    planner = _run(
        monkeypatch,
        "region-object-rearrangement-domain",
        {"pstart": _start("pstart", [0.0, 0.0])},
        ("goto-poi", "p1"),
    )
    assert len(planner.node_ids) == 2


def test_multirobot_start_key_restricts(monkeypatch):
    planner = _run(
        monkeypatch,
        "region-object-rearrangement-domain-multirobot-fd",
        {"pstarteuclid": _start("pstarteuclid", [0.0, 0.0])},
        ("goto-poi", "euclid", "p1"),
    )
    assert len(planner.node_ids) == 2, (
        "multirobot starts are keyed pstart{robot_id}; the component "
        "restriction must still fire"
    )


def test_multiple_robot_starts_keep_the_union_of_components(monkeypatch):
    """One robot in the main component, one on the island -> keep both."""
    planner = _run(
        monkeypatch,
        "region-object-rearrangement-domain-multirobot-fd",
        {
            "pstarteuclid": _start("pstarteuclid", [0.0, 0.0]),
            "pstarthamilton": _start("pstarthamilton", [6.0, 0.0]),
        },
        ("goto-poi", "euclid", "p1"),
    )
    assert len(planner.node_ids) == 3, (
        "restricting to one robot's component would make the other robot's "
        "places unsnappable"
    )
