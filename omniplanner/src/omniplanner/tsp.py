import copy
import logging
from collections.abc import Mapping
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


# Edge attribute LayerPlanner.weighted() adds: metres between the two nodes.
EDGE_LENGTH = "length"


class LayerPlanner:
    # Shortest paths count hops by default; weighted() views measure metres.
    weight = None

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
            v: np.array(dsg.get_node(v).attributes.position[:2]) for v in self.node_ids
        }
        # every node's position, also after a view narrows the snapping set
        self._all_positions = dict(self.node_value_to_position)

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
                length = nx.shortest_path_length(self.nx_layer, s, t, weight=self.weight)
            except nx.exception.NetworkXNoPath:
                logger.warning(f"No connection in DSG between {s} and {t}")
                length = np.inf
            return length
        else:
            return self.stored_shortest_path_lengths[s][t]

    def get_shortest_path(self, s, t):
        if self.stored_shortest_path is None:
            return nx.shortest_path(self.nx_layer, s, t, weight=self.weight)
        else:
            return self.stored_shortest_path[s][t]

    def restricted_to_component(self, point):
        """A view of this planner whose snapping (`get_closest_node_id`,
        `get_closest_point`, and therefore every `external_*` anchor) only
        considers nodes in the connected component containing the node
        nearest ``point``.

        Saved DSGs contain island place components (behind-walls clusters the
        robot can never reach); snapping an external position to one makes
        every path involving it come back inf, which grounding then drops --
        producing disconnected, unsolvable PDDL problems for physically
        reachable goals. Restricting the anchors to the start's component
        keeps every distance finite over the graph the robot can actually
        traverse. Path queries still run on the full layer graph. Returns
        ``self`` unchanged when the layer is empty or fully connected.
        """
        return self.restricted_to_components([point])

    def restricted_to_components(self, points):
        """Like ``restricted_to_component``, but keeps the UNION of the
        components containing the nodes nearest each of ``points``.

        Multirobot problems have one start per robot, and those starts can sit
        in different components of a saved DSG. Restricting to any single
        robot's component would make the other robots' places unsnappable, so
        the union is the right restriction there. ``None`` entries are ignored;
        an empty list leaves the planner unrestricted.
        """
        points = [np.asarray(p) for p in points if p is not None]
        if len(self.node_ids) == 0 or len(points) == 0:
            return self
        component = set()
        for point in points:
            anchor = self.get_closest_node_id(point)
            component |= nx.node_connected_component(self.nx_layer, anchor)
        if len(component) == len(self.node_ids):
            return self
        view = copy.copy(self)
        keep = [i for i, v in enumerate(self.node_ids) if v in component]
        view.node_ids = [self.node_ids[i] for i in keep]
        view.node_positions = self.node_positions[keep]
        view.node_value_to_position = {
            v: self.node_value_to_position[v] for v in view.node_ids
        }
        return view

    def weighted(self):
        """A view whose shortest paths and distances are in metres (straight-line
        length of each edge between its two nodes) instead of hop counts.

        Hops are fine for ordering one robot's visits on an evenly spaced graph,
        but not for deciding which of several robots is closer to a task.
        """
        view = copy.copy(self)
        g = self.nx_layer.copy()
        for u, v, data in g.edges(data=True):
            data[EDGE_LENGTH] = float(
                np.linalg.norm(self._all_positions[u] - self._all_positions[v])
            )
        view.nx_layer = g
        view.weight = EDGE_LENGTH
        # precomputed paths count hops
        view.stored_shortest_path_lengths = None
        view.stored_shortest_path = None
        return view

    def without(self, nodes=(), edges=()):
        """A view with ``nodes`` and ``edges`` (node-id pairs) taken out of the
        graph: paths route around them and nothing snaps to them.

        Runtime constraints (forbidden places / passages) are applied here, so
        the distances a planner is given and the paths its plan is expanded
        into come from the same graph.
        """
        nodes = {n for n in nodes if n in self.nx_layer}
        edges = [(u, v) for u, v in edges if self.nx_layer.has_edge(u, v)]
        if not nodes and not edges:
            return self
        view = copy.copy(self)
        g = self.nx_layer.copy()
        g.remove_nodes_from(nodes)
        g.remove_edges_from(edges)
        view.nx_layer = g
        keep = [i for i, v in enumerate(self.node_ids) if v not in nodes]
        view.node_ids = [self.node_ids[i] for i in keep]
        view.node_positions = self.node_positions[keep]
        view.node_value_to_position = {
            v: self.node_value_to_position[v] for v in view.node_ids
        }
        view.stored_shortest_path_lengths = None
        view.stored_shortest_path = None
        return view

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

    def external_distance_matrix(self, points):
        """Pairwise external distances among ``points`` (Nx2 array-like).

        Equivalent to calling ``get_external_distance`` for every pair, but runs
        a single-source shortest-path from each *distinct* snapped anchor node
        instead of a fresh search per pair. This turns O(pairs) graph searches
        into O(distinct-anchors), which matters when grounding connects many
        symbols. Graph distance as ``get_shortest_distance`` measures it: hop
        count, or metres on a ``weighted()`` view.
        """
        points = [np.asarray(p) for p in points]
        n = len(points)
        anchors = [self.get_closest_node_id(p) for p in points]
        offsets = [float(np.linalg.norm(points[i] - self.get_closest_point(points[i])))
                   for i in range(n)]

        # One search per distinct anchor, reused across all targets sharing it.
        ss_lengths = {}
        for a in set(anchors):
            if self.weight is None:
                ss_lengths[a] = nx.single_source_shortest_path_length(self.nx_layer, a)
            else:
                ss_lengths[a] = nx.single_source_dijkstra_path_length(
                    self.nx_layer, a, weight=self.weight
                )

        D = np.full((n, n), np.inf)
        for i in range(n):
            li = ss_lengths[anchors[i]]
            for j in range(n):
                gd = li.get(anchors[j], np.inf)
                D[i, j] = offsets[i] + gd + offsets[j]
        return D

    def get_external_path(self, point_a, point_b, extend_ends=False):
        idx_a = self.get_closest_node_id(point_a)
        idx_b = self.get_closest_node_id(point_b)

        path_in_graph = self.get_shortest_path(idx_a, idx_b)
        path_points = [self._all_positions[n] for n in path_in_graph]

        return [point_a] + path_points + [point_b]

    def get_external_path_nodes(self, point_a, point_b):
        """The graph nodes ``get_external_path`` passes through, in order."""
        return self.get_shortest_path(
            self.get_closest_node_id(point_a), self.get_closest_node_id(point_b)
        )


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
    domain: TspDomain,
    dsg: Any,
    robot_states: Mapping,
    goal: TspGoal,
    feedback: Any = None,
) -> RobotWrapper[GroundedTspProblem]:
    logger.info("Grounding TSP Problem")

    robot_pose = robot_states[goal.robot_id]
    if robot_pose is None:
        raise RuntimeError(
            f"Cannot plan for robot '{goal.robot_id}': no transform available "
            "(is the robot online and publishing TF?)"
        )
    start = robot_pose[:2]

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
