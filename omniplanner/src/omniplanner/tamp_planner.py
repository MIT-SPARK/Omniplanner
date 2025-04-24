from dataclasses import dataclass
from typing import List, overload

import numpy as np
import spark_dsg
import shapely.geometry as geo
from multipledispatch import dispatch

from omniplanner.omniplanner import PlanningDomain


from dsg_tamp.mapping.outdoor_dsg_utils import spark_dsg_to_tamp
from dsg_tamp.mapping.outdoor_dsg_viz import plot_dsg_objects, plot_dsg_places
from dsg_tamp.problems.dcist_2023.phoenix_planning_interface import (
    get_general_problem,
    plan_general_problem,
    hydra_pddl_indices_to_solver_indices,
)

from dsg_tamp.problems.pddl_helpers import (
    lisp_string_to_ast,
)

from dsg_tamp.problems.dcist_2023.generate_problem_maps import make_object_suspicious

def get_external_predicates(pose, dsg):
    pose_point = geo.Point(pose[0], pose[1])
    print("Planning from point: ", pose_point)
    start_places = [
        ix
        for ix, p in enumerate(dsg.places.boundaries_shapely)
        if p.contains(pose_point)
    ]
    print(start_places)
    external_predicates = []
    for sp in start_places:
        external_predicates.append(("PoseInPlace", "pose0", f"p{sp}"))
        print(f"PoseInPlace(pose0, p{sp})")
    return external_predicates


class TAMPDomain(PlanningDomain):
    def __init__(self):
        domain_path = '/home/rrg/dsg-tamp/dsg_tamp/problems/dcist_2023/manipulation/domain.pddl'
        with open(domain_path, "r") as fo:
            self.pddl_domain = fo.read()
        
        stream_path = '/home/rrg/dsg-tamp/dsg_tamp/problems/dcist_2023/manipulation/stream.pddl'
        with open(stream_path, "r") as fo:
            self.pddl_stream = fo.read()

class TAMPProblem:
    def __init__(self, problem):
        self.problem = problem

@dataclass
class TAMPPlan:
    plan = list

@dataclass
class TAMPGoal:
    goal: str
    robot_id: str

@dispatch(TAMPDomain, object, dict, TAMPGoal)
def ground_problem(domain, map_context, robot_states, goal) -> TAMPProblem:
    pddl_domain = domain.pddl_domain
    pddl_stream = domain.pddl_stream
    dsg = map_context
    start = robot_states[goal.robot_id]
    pddl_goal = lisp_string_to_ast(goal.goal)
    local_pddl_goal = hydra_pddl_indices_to_solver_indices(dsg, pddl_goal)
    affordances = None
    
    external_predicates = get_external_predicates(start, dsg)

    problem = get_general_problem(
            dsg,
            pddl_domain,
            pddl_stream,
            start,
            local_pddl_goal,
            external_predicates=external_predicates,
            affordances=affordances,
        )
    
    return TAMPProblem(problem)


@dispatch(TAMPProblem, object)
def make_plan(grounded_problem, map_context) -> TAMPPlan:
    dsg = map_context
    plan = plan_general_problem(grounded_problem.problem)
    
    return plan
