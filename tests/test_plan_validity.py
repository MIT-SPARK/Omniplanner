"""Which scene changes invalidate the plan, and what the repair node does about them."""

import threading
from types import SimpleNamespace as NS

import numpy as np
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Gaze,
    Pick,
    Place,
)

from omniplanner.plan_validity import (
    ADDED,
    MOVED,
    REMOVED,
    affecting_changes,
    broken_route,
    goal_objects,
    plan_dependencies,
    route_places,
)
from omniplanner.world_state import WorldState, WorldStateTracker


def _plan():
    return ActionSequence(
        "p",
        "hilbert",
        [
            Follow("map", np.zeros((2, 2))),
            Gaze("map", np.zeros(3), np.array([5.0, 1.0, 0.0]), "o8"),
            Pick("map", "trash", np.zeros(3), np.array([26.0, -5.4, 0.3]), "o4"),
            Place("map", "", np.zeros(3), np.array([37.9, -9.0, 0.0]), "o4"),
        ],
    )


def _moved(sym, old, new):
    return {"kind": MOVED, "symbol": sym, "old": old, "new": new}


def test_dependencies_are_what_the_plan_picks_and_gazes_at():
    deps = plan_dependencies([_plan()])
    assert deps == {"o8": ("gaze", (5.0, 1.0)), "o4": ("pick", (26.0, -5.4))}


def test_goal_objects():
    g = goal_objects(
        "(and (object-in-place o4 t598)(visited-object o5)(not (suspicious o8)))"
    )
    assert g == {"visit": {"o5"}, "inspect": {"o8"}, "relocate": {"o4"}}


def test_goal_objects_of_the_by_forms_are_their_targets():
    g = goal_objects(
        "(and (visited-object-by hilbert o20)(visited-place-by gauss t12)"
        "(inspected-by hamilton o7))"
    )
    assert g == {"visit": {"o20", "t12"}, "inspect": {"o7"}, "relocate": set()}


def test_removed_object_a_robot_must_visit_invalidates():
    gone = [{"kind": REMOVED, "symbol": "O20", "old": (17.0, 3.0), "new": None}]
    hit = affecting_changes(gone, {}, "(visited-object-by hilbert o20)")
    assert [s for s, _ in hit] == ["o20"]


def test_object_to_pick_moved_far_invalidates():
    deps = plan_dependencies([_plan()])
    hit = affecting_changes([_moved("O4", (26.0, -5.4), (30.0, -2.0))], deps, "")
    assert [s for s, _ in hit] == ["o4"] and "pick" in hit[0][1]


def test_small_moves_and_unrelated_objects_do_not():
    deps = plan_dependencies([_plan()])
    assert affecting_changes([_moved("O4", (26.0, -5.4), (26.2, -5.5))], deps, "") == []
    assert (
        affecting_changes([_moved("O13", (0, 0), (9, 9))], deps, "(visited-object o5)")
        == []
    )


def test_removed_goal_object_invalidates_until_visited():
    gone = [{"kind": REMOVED, "symbol": "O5", "old": (33.8, -3.7), "new": None}]
    assert affecting_changes(gone, {}, "(visited-object o5)")
    assert (
        affecting_changes(gone, {}, "(visited-object o5)", WorldState(visited={"o5"}))
        == []
    )


def test_relocated_object_moved_again_invalidates():
    hit = affecting_changes(
        [_moved("O4", (37.9, -9.0), (20.0, -4.0))], {}, "(object-in-place o4 t598)"
    )
    assert hit and "end up" in hit[0][1]


def test_held_object_is_never_affected():
    state = WorldState(holding={"hilbert": ["o4"]})
    deps = plan_dependencies([_plan()])
    assert affecting_changes([_moved("O4", (26, -5), (40, 0))], deps, "", state) == []


def test_added_goal_object_invalidates():
    added = [{"kind": ADDED, "symbol": "O5", "old": None, "new": (10.0, 0.0)}]
    assert affecting_changes(added, {}, "(visited-object o5)")


# ---------- the repair node's reaction ----------


def _node(map_version=99):
    from omniplanner_ros.omniplanner_repair_node import OmniPlannerRepairRos as N

    published, timers, paused = [], [], []
    node = NS(
        _plan_lock=threading.Lock(),
        _world_lock=threading.Lock(),
        world_tracker=WorldStateTracker(),
        _plan_deps=plan_dependencies([_plan()]),
        _plan_robots=["hilbert", "lewis"],
        _pause_pubs={},
        _executor_pause_topic="/{robot}/spot_executor_node/pause",
        _pending_release={},
        _planning_goal="",
        _active_goal="(and (object-in-place o4 t598) (visited-object o5))",
        _pending_changes=[],
        _pending_version=0,
        _settle_timer=None,
        _change_settle_s=1.0,
        _move_threshold_m=0.5,
        _pick_failures={},
        _max_pick_retries=2,
        _route={},
        _plan_ids={},
        _actions_done={},
        _route_version=None,
        _place_finder=lambda dsg: lambda xy: None,
        _replan_pub=NS(publish=published.append),
        _current_dsg=lambda: None,
        _map_version=lambda: map_version,
        _map_state=lambda: ("", map_version),
        _announced=set(),
        _pending_epoch="",
        _pending_busy=set(),
        _left_alone=set(),
        _running_plans={},
        _dispatched={},
        _path_status_pubs={},
        robot_adaptors={},
        dsg_frame="map",
        get_clock=lambda: NS(now=lambda: NS(nanoseconds=0)),
        create_publisher=lambda _type, topic, _qos: NS(
            topic=topic, publish=lambda m, t=topic: paused.append((t, m.data))
        ),
        create_timer=lambda period, cb: timers.append(cb) or NS(cancel=lambda: None),
        destroy_timer=lambda t: None,
        get_logger=lambda: NS(
            info=lambda *_: None, warning=lambda *_: None, error=lambda *_: None
        ),
    )
    for name in (
        "_judge_changes",
        "_wait_for_map_then_replan",
        "_request_replan",
        "_scene_changes_callback",
        "_action_done_callback",
        "_pause_robots",
        "_pause_pub_for",
        "finalize_plans",
        "_put_down_first",
        "_leave_busy_alone",
        "_keep_carrying",
        "_is_path_robot",
        "_watch_progress",
        "_now_s",
    ):
        setattr(node, name, getattr(N, name).__get__(node))
    node.paused = paused
    return node, published, timers


def _change_msg(source, sym, old, new, version=5):
    from geometry_msgs.msg import Point
    from heracles_ros_interfaces.msg import SceneChange, SceneChangeMsg

    m = SceneChangeMsg(source=source, map_version=version)
    m.changes = [
        SceneChange(
            kind=SceneChange.MOVED,
            symbol=sym,
            old_position=Point(x=old[0], y=old[1]),
            new_position=Point(x=new[0], y=new[1]),
        )
    ]
    return m


def test_a_move_that_matters_requests_one_replan_after_settling():
    node, published, timers = _node()
    node._scene_changes_callback(
        _change_msg("change_detection", "O4", (26, -5.4), (30, -2))
    )
    node._scene_changes_callback(_change_msg("change_detection", "O13", (0, 0), (9, 9)))
    assert published == [] and len(timers) == 1  # waits for the burst to settle
    timers[0]()
    assert len(published) == 1 and "o4 moved" in published[0].data


def test_our_own_placement_is_ignored():
    node, published, timers = _node()
    node._scene_changes_callback(
        _change_msg("executor/hilbert", "O4", (26, -5.4), (37.9, -9))
    )
    assert timers == [] and published == []


def test_a_finished_pick_no_longer_depends_on_where_the_object_was():
    node, published, timers = _node()
    node._action_done_callback(
        NS(
            action_type="PICK",
            success=True,
            robot_name="hilbert",
            object_id="O4",
            points=[],
            plan_id="p1",
        )
    )
    assert "o4" not in node._plan_deps


def test_failed_picks_replan_up_to_the_cap():
    node, published, _ = _node()
    fail = NS(
        action_type="PICK",
        success=False,
        robot_name="hilbert",
        object_id="O4",
        points=[],
        plan_id="p1",
    )
    for _ in range(3):
        node._action_done_callback(fail)
    assert len(published) == 2  # attempts 1 and 2; the third gives up


def test_a_change_that_matters_pauses_every_robot_before_the_map_arrives():
    node, published, timers = _node(map_version=0)  # the map is not there yet
    node._scene_changes_callback(
        _change_msg("change_detection", "O4", (26, -5.4), (30, -2))
    )
    assert node.paused == []  # nothing judged while the burst settles
    timers[0]()
    assert node.paused == [
        ("/hilbert/spot_executor_node/pause", True),
        ("/lewis/spot_executor_node/pause", True),
    ]
    assert published == []  # the replan still waits for the map


def test_a_change_that_does_not_matter_pauses_nobody():
    node, published, timers = _node()
    node._scene_changes_callback(_change_msg("change_detection", "O13", (0, 0), (9, 9)))
    timers[0]()
    assert node.paused == [] and published == []


def test_the_pause_publishers_are_made_with_the_plan():
    node, _, _ = _node()
    node._plan_robots, node._pause_pubs = [], {}
    node.finalize_plans({"hilbert": _plan(), "euclid": _plan()}, {})
    assert node._plan_robots == ["euclid", "hilbert"]
    assert sorted(p.topic for p in node._pause_pubs.values()) == [
        "/euclid/spot_executor_node/pause",
        "/hilbert/spot_executor_node/pause",
    ]
    assert node.paused == []  # made, not used


def test_the_map_version_is_published_when_it_changes():
    from omniplanner_ros.omniplanner_repair_node import OmniPlannerRepairRos

    sent, version = [], {"v": 4}
    node = NS(
        _version_pub=NS(publish=lambda m: sent.append(m.data)),
        _published_version=None,
        _map_version=lambda: version["v"],
    )
    publish = OmniPlannerRepairRos._publish_map_version.__get__(node)
    publish()
    publish()  # a map with the same version: nothing new
    version["v"] = 5  # e.g. after "SGET: delete t444"
    publish()
    assert sent == [4, 5]


# Places t1..t4 in a row along x; the robot starts off the graph at (0, 1).
_PLACES = {(1.0, 0.0): "t1", (2.0, 0.0): "t2", (3.0, 0.0): "t3", (4.0, 0.0): "t4"}
_EDGES = {("t1", "t2"), ("t2", "t3"), ("t3", "t4")}


def _route_plan():
    def follow(*pts):
        return Follow(frame="map", path2d=np.array(pts))

    seq = ActionSequence(
        plan_id="p1",
        robot_name="hilbert",
        actions=[follow((0, 1), (1, 0), (2, 0)), follow((2, 0), (3, 0), (4, 0))],
    )
    return route_places({"hilbert": seq}, lambda xy: _PLACES.get(xy))


def _check(route, done=0, places=("t1", "t2", "t3", "t4"), edges=_EDGES):
    return broken_route(
        route,
        {"hilbert": done},
        has_place=lambda s: s in places,
        connected=lambda a, b: (a, b) in edges or (b, a) in edges,
    )


def test_a_route_is_the_places_its_follows_pass():
    assert _route_plan() == {
        "hilbert": [(0, [None, "t1", "t2"]), (1, ["t2", "t3", "t4"])]
    }


def test_an_intact_map_leaves_the_route_alone():
    assert _check(_route_plan()) == []


def test_a_removed_place_ahead_breaks_the_route():
    ((robot, why),) = _check(_route_plan(), places=("t1", "t2", "t4"))
    assert robot == "hilbert" and "t3" in why and "removed" in why


def test_a_removed_edge_ahead_breaks_the_route():
    ((_, why),) = _check(_route_plan(), edges={("t1", "t2"), ("t2", "t3")})
    assert "t3 to t4" in why


def test_a_place_already_driven_past_does_not_matter():
    # The first Follow (through t1) is done; t1 going away changes nothing.
    assert _check(_route_plan(), done=1, places=("t2", "t3", "t4")) == []


def test_a_cut_route_pauses_the_robots_and_replans():
    from omniplanner_ros.omniplanner_repair_node import OmniPlannerRepairRos as N

    paused, replans = [], []
    node = NS(
        _plan_lock=threading.Lock(),
        _route=_route_plan(),
        _actions_done={},
        _route_version=3,
        _plan_robots=["euclid", "hilbert"],
        _map_version=lambda: 4,
        _current_dsg=lambda: NS(has_edge=lambda a, b: True),
        _mesh_places=lambda dsg: {"t1": 1, "t2": 2, "t4": 4},  # t3 deleted
        _pause_robots=lambda robots, why: paused.append(robots),
        _request_replan=replans.append,
    )
    N._check_routes(node)
    assert paused == [["euclid", "hilbert"]] and "t3" in replans[0]
    assert node._route == {}  # judged once
    N._check_routes(node)
    assert len(replans) == 1
