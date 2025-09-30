import logging
from dataclasses import dataclass
from typing import Any, List, overload

import networkx as nx
import numpy as np
import spark_dsg
import spark_dsg.networkx as dsg_nx
from plum import dispatch

from omniplanner.omniplanner import PlanningDomain, RobotWrapper
from omniplanner.utils import str_to_ns_value

logger = logging.getLogger(__name__)


class LayerPlanner:
    def __init__(self, dsg, layer, precompute_shortest_paths=False):
        if layer == spark_dsg.DsgLayers.MESH_PLACES:
            try:
                self.nx_layer = dsg_nx.layer_to_networkx(dsg.get_layer(layer))
            except Exception as _:
                logger.warning(
                    f"Failed to load layer {spark_dsg.DsgLayers.MESH_PLACES}, attempting to fall back to layer 20"
                )
                self.nx_layer = dsg_nx.layer_to_networkx(dsg.get_layer(20))
        else:
            self.nx_layer = dsg_nx.layer_to_networkx(dsg.get_layer(layer))

        self.node_ids = [n for n in self.nx_layer.nodes]
        self.node_positions = np.array(
            [dsg.get_node(v).attributes.position[:2] for v in self.node_ids]
        )
        self.node_value_to_position = {
            v: dsg.get_node(v).attributes.position[:2] for v in self.node_ids
        }

        if precompute_shortest_paths:
            self.stored_shortest_path_lengths = dict(
                nx.all_pairs_dijkstra_path_length(self.nx_layer)
            )
            self.stored_shortest_path = dict(nx.all_pairs_dijkstra_path(self.nx_layer))
        else:
            self.stored_shortest_path_lengths = None
            self.stored_shortest_path = None

    def get_shortest_distance(self, s, t):
        if self.stored_shortest_path_lengths is None:
            try:
                length = nx.shortest_path_length(self.nx_layer, s, t)
            except nx.exception.NetworkXNoPath:
                logger.warning(f"No connection in DSG between {s} and {t}")
                length = np.inf
            return length
        else:
            return self.stored_shortest_path_lengths[s][t]

    def get_shortest_path(self, s, t):
        if self.stored_shortest_path is None:
            return nx.shortest_path(self.nx_layer, s, t)
        else:
            return self.stored_shortest_path[s][t]

    def get_closest_node_id(self, point):
        closest_idx = np.argmin(np.linalg.norm(self.node_positions - point, axis=1))
        return self.node_ids[closest_idx]

    def get_closest_point(self, point):
        closest_idx = np.argmin(np.linalg.norm(self.node_positions - point, axis=1))
        return self.node_positions[closest_idx]

    def get_external_distance(self, point_a, point_b):
        idx_a = self.get_closest_node_id(point_a)
        idx_b = self.get_closest_node_id(point_b)

        nearest_a = self.get_closest_point(point_a)
        da = np.linalg.norm(point_a - nearest_a)

        nearest_b = self.get_closest_point(point_b)
        db = np.linalg.norm(point_b - nearest_b)

        return da + self.get_shortest_distance(idx_a, idx_b) + db

    def get_external_path(self, point_a, point_b, extend_ends=False):
        idx_a = self.get_closest_node_id(point_a)
        idx_b = self.get_closest_node_id(point_b)

        path_in_graph = self.get_shortest_path(idx_a, idx_b)
        path_points = [self.node_value_to_position[n] for n in path_in_graph]

        return [point_a] + path_points + [point_b]


def two_opt(route, cost_mat):
    def cost_change(n1, n2, n3, n4):
        return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                delta = cost_change(best[i - 1], best[i], best[j - 1], best[j])
                if delta < -0.0001:
                    best[i:j] = best[j - 1 : i - 1 : -1]
                    improved = True
        route = best
    return best


def solve_tsp_2opt(distance_matrix):
    initial_route = list(range(len(distance_matrix)))
    visit_order = two_opt(initial_route, distance_matrix)

    return visit_order


@dataclass
class FollowPathPrimitive:
    path: np.ndarray


@dataclass
class FollowPathPlan:
    steps: list


@dataclass
class TspDomain(PlanningDomain):
    solver: str


@dataclass
class GroundedTspProblem:
    start_point: np.ndarray
    goal_points: List[np.ndarray]
    distances: np.ndarray
    solver: str = "2opt"


@dataclass
class TspGoal:
    goal_points: List[str]
    robot_id: str


@overload
@dispatch
def ground_problem(
    domain: TspDomain, dsg: Any, robot_states: dict, goal: TspGoal, feedback: Any = None
) -> RobotWrapper[GroundedTspProblem]:
    logger.info("Grounding TSP Problem")

    start = robot_states[goal.robot_id][:2]

    def get_loc(symbol):
        node = dsg.find_node(str_to_ns_value(symbol))
        if node is None:
            raise Exception(f"Requested symbol {symbol} not in scene graph")
        return node.attributes.position[:2]

    referenced_points = np.array([get_loc(symbol) for symbol in goal.goal_points])
    referenced_points = np.vstack([start, referenced_points])

    layer_planner = LayerPlanner(dsg, spark_dsg.DsgLayers.MESH_PLACES)

    # Compute pairwise distance matrix based on places
    n = len(referenced_points)
    distance_matrix = np.zeros((n, n))

    for ix in range(n):
        for jx in range(ix + 1, n):
            distance_matrix[ix, jx] = layer_planner.get_external_distance(
                referenced_points[ix], referenced_points[jx]
            )
    distance_matrix += distance_matrix.T

    return RobotWrapper(
        goal.robot_id,
        GroundedTspProblem(start, referenced_points, distance_matrix, domain.solver),
    )


@dispatch
def make_plan(grounded_problem: GroundedTspProblem, map_context: Any) -> FollowPathPlan:
    logger.warning("Making TSP Plan")

    match grounded_problem.solver:
        case "2opt":
            logger.warning("Solving with 2opt")
            tsp_order = solve_tsp_2opt(grounded_problem.distances)
        case _:
            raise NotImplementedError(
                f"Requested TSP Solver not implemented: {grounded_problem.solver})"
            )

    tsp_points = grounded_problem.goal_points[tsp_order]

    plan = FollowPathPlan([])

    layer_planner = LayerPlanner(map_context, spark_dsg.DsgLayers.MESH_PLACES)
    for idx in range(len(tsp_points) - 1):
        path = layer_planner.get_external_path(tsp_points[idx], tsp_points[idx + 1])
        p = FollowPathPrimitive(path)
        plan.steps.append(p)

    return plan