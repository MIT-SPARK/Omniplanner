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
    goal_objects,
    plan_dependencies,
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


def _node():
    from omniplanner_ros.omniplanner_repair_node import OmniPlannerRepairRos as N

    published, timers = [], []
    node = NS(
        _plan_lock=threading.Lock(),
        _world_lock=threading.Lock(),
        world_tracker=WorldStateTracker(),
        _plan_deps=plan_dependencies([_plan()]),
        _active_goal="(and (object-in-place o4 t598) (visited-object o5))",
        _pending_changes=[],
        _pending_version=0,
        _settle_timer=None,
        _change_settle_s=1.0,
        _move_threshold_m=0.5,
        _pick_failures={},
        _max_pick_retries=2,
        _replan_pub=NS(publish=published.append),
        _current_dsg=lambda: None,
        _map_version=lambda: 99,
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
    ):
        setattr(node, name, getattr(N, name).__get__(node))
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
    )
    for _ in range(3):
        node._action_done_callback(fail)
    assert len(published) == 2  # attempts 1 and 2; the third gives up
