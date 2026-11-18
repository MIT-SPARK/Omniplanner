"""A Husky's visits, audited against where it drove.

A Husky reports no actions: the planner infers what it visited from its pose
against the waypoints of its Path. Each test here drives a robot honestly along
a path of some shape, one small step at a time, and after every step compares
the planner's bookkeeping with the track so far:

  * no place is visited unless the robot came within reach of it,
  * progress never runs ahead of the waypoint the robot is driving to,
  * the plan is not over before the robot is at the end of its path,
  * at the end every waypoint's place is visited and the plan is over.

The straight path is the shape the first version was written for. The others
are the ones a real mission produces: out and back, through a place twice,
loops, crossings. (Field test 2026-11-18: gauss was credited with a place
60 m away because its path came home over its own ground.)
"""

import numpy as np
import pytest
from test_willow_repair_node import _map, _node, _path

ROBOT = "testhusky"
REACH = 1.0  # the planner's visited_radius_m
STEP = 0.2

# Places 3 m apart, so that being in reach of one is never being in reach of
# its neighbour. Names are t<10 * column + row>.
GRID = {f"t{10 * i + j}": (3.0 * i, 3.0 * j) for i in range(1, 6) for j in range(0, 5)}


def at(i, j):
    return GRID[f"t{10 * i + j}"]


def line(*cells):
    return [at(*c) for c in cells]


OUT = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
SHAPES = {
    "straight": line(*OUT),
    "out and back": line(*OUT, *OUT[-2::-1]),
    "out, on, and back to the middle": line(
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (4, 0), (3, 0)
    ),
    "starts where it ends, one far place between": line((1, 0), (5, 0), (1, 0)),
    "a loop closing on its start": line((1, 0), (3, 0), (3, 2), (1, 2), (1, 0)),
    "twice round the same loop": line(
        (1, 0), (3, 0), (3, 2), (1, 2), (1, 0), (3, 0), (3, 2), (1, 2), (1, 0)
    ),
    "a figure of eight": line(
        (1, 1), (2, 2), (3, 1), (4, 0), (5, 1), (4, 2), (3, 1), (2, 0), (1, 1)
    ),
    "a spur: into a dead end and out again": line(
        (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 1), (3, 0), (4, 0), (5, 0)
    ),
    "through the same corner three times": line(
        (1, 0), (2, 0), (2, 1), (2, 0), (3, 0), (2, 0), (2, 2)
    ),
    "back and forth between two places": line(
        (1, 0), (2, 0), (1, 0), (2, 0), (1, 0), (2, 0)
    ),
}


def _track(waypoints, start, step=STEP):
    """The robot's positions, from start through each waypoint in order, with
    the index of the waypoint it is driving to at each one."""
    pos = np.asarray(start, dtype=float)
    yield pos.copy(), 0
    for i, w in enumerate(waypoints):
        w = np.asarray(w, dtype=float)
        while np.hypot(*(w - pos)) > step:
            pos = pos + (w - pos) / np.hypot(*(w - pos)) * step
            yield pos.copy(), i
        pos = w.copy()
        yield pos.copy(), i


def _audit(waypoints, start):
    node = _node(
        _map(places=GRID, o58=None), poses={ROBOT: np.array([*start, 0.0, 0.0])}
    )
    node.finalize_plans({ROBOT: _path(*waypoints)}, node.f_poses)
    assert ROBOT in node._dispatched
    place_xy = {name: np.asarray(xy) for name, xy in GRID.items()}
    closest = {name: np.inf for name in GRID}  # how near the robot has come to each
    track = list(_track(waypoints, start))
    for k, (pos, heading_for) in enumerate(track):
        for name, xy in place_xy.items():
            closest[name] = min(closest[name], float(np.hypot(*(xy - pos))))
        node.f_poses[ROBOT] = np.array([*pos, 0.0, 0.0])
        node._progress_tick()

        never_reached = sorted(
            p for p in node.world_tracker.visited if closest[p] > REACH + 1e-6
        )
        assert not never_reached, (
            f"step {k} at {pos.round(1)}: visited without coming within {REACH} m: "
            + ", ".join(f"{p} ({closest[p]:.1f} m)" for p in never_reached)
        )
        done = node._actions_done.get(ROBOT, 0)
        assert done <= heading_for, (
            f"step {k} at {pos.round(1)}: progress at waypoint {done} while "
            f"driving to waypoint {heading_for} of {len(waypoints) - 1}"
        )
        if k < len(track) - 1 and heading_for < len(waypoints) - 1:
            assert ROBOT in node._dispatched, (
                f"step {k} at {pos.round(1)}: plan called over while driving to "
                f"waypoint {heading_for} of {len(waypoints) - 1}"
            )
    return node


@pytest.mark.parametrize("shape", list(SHAPES))
def test_visits_match_the_track(shape):
    waypoints = SHAPES[shape]
    node = _audit(waypoints, start=(0.0, 0.0))
    wanted = {name for name, xy in GRID.items() if xy in waypoints}
    assert wanted <= node.world_tracker.visited
    assert ROBOT not in node._dispatched  # at the end of its path: over


@pytest.mark.parametrize("shape", [s for s in SHAPES if SHAPES[s][0] == SHAPES[s][-1]])
def test_a_robot_starting_on_its_last_waypoint_still_drives_the_path(shape):
    """Standing on the first waypoint of a path that ends there too."""
    waypoints = SHAPES[shape]
    node = _audit(waypoints, start=waypoints[0])
    assert ROBOT not in node._dispatched


def test_a_cut_corner_still_counts_as_progress():
    """The tolerance the last-match rule was written for must survive the fix:
    a robot that rounds a corner 0.8 m inside it has passed that waypoint."""
    waypoints = line((1, 0), (3, 0), (3, 2), (5, 2))
    node = _node(
        _map(places=GRID, o58=None), poses={ROBOT: np.array([0.0, 0.0, 0.0, 0.0])}
    )
    node.finalize_plans({ROBOT: _path(*waypoints)}, node.f_poses)
    corner, after = np.array(at(3, 0)), np.array(at(3, 2))
    inside = corner + np.array([-0.55, 0.55])  # 0.78 m from the corner
    driven = [waypoints[0], tuple(inside), tuple(after), waypoints[-1]]
    for pos, _ in _track(driven, (0.0, 0.0)):
        node.f_poses[ROBOT] = np.array([*pos, 0.0, 0.0])
        node._progress_tick()
    assert ROBOT not in node._dispatched
    assert {"t10", "t30", "t32", "t52"} <= node.world_tracker.visited


def test_a_robot_that_stops_short_is_not_finished():
    """Out and back, but it stops at the far end: the way home is not driven."""
    waypoints = SHAPES["out and back"]
    node = _node(
        _map(places=GRID, o58=None), poses={ROBOT: np.array([0.0, 0.0, 0.0, 0.0])}
    )
    node.finalize_plans({ROBOT: _path(*waypoints)}, node.f_poses)
    for pos, _ in _track(waypoints[:5], (0.0, 0.0)):
        node.f_poses[ROBOT] = np.array([*pos, 0.0, 0.0])
        node._progress_tick()
    assert ROBOT in node._dispatched
    assert node._actions_done[ROBOT] == 4
