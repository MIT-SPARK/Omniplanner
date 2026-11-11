"""Plan repair in the willow field test (Nov 10): a base-station goal manager
named willow with no pose, a Husky (testhusky) whose plan is a nav_msgs/Path, Spots
(testbot_a, testbot_b) that pick and place, offline robots, and a map database
reloaded partway through the mission.

The repair node is built without ROS: its __init__ only creates publishers and
timers, which are faked here.
"""

import json
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
import spark_dsg
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from robot_executor_interface.action_descriptions import (
    ActionSequence,
    Follow,
    Pick,
    Place,
)

from omniplanner.plan_validity import route_places
from omniplanner.world_state import BUSY_ROBOTS, WorldStateTracker

repair = pytest.importorskip("omniplanner_ros.omniplanner_repair_node")
N = repair.OmniPlannerRepairRos

# Places t1..t4 along x, 1 m apart; o58 lies at t2.
PLACES = {"t1": (1.0, 0.0), "t2": (2.0, 0.0), "t3": (3.0, 0.0), "t4": (4.0, 0.0)}


def _map(epoch="e1", version=1, holding=None, places=PLACES, o58=(2.0, 0.2)):
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    ids = {}
    for i, (name, xy) in enumerate(places.items()):
        a = spark_dsg.PlaceNodeAttributes()
        a.position = np.array([*xy, 0.0])
        nid = spark_dsg.NodeSymbol("t", int(name[1:]))
        G.add_node(spark_dsg.DsgLayers.MESH_PLACES, nid.value, a)
        ids[name] = nid.value
    names = list(places)
    for a, b in zip(names, names[1:]):
        if int(b[1:]) - int(a[1:]) == 1:
            G.insert_edge(ids[a], ids[b])
    if o58 is not None:
        o = spark_dsg.KhronosObjectAttributes()
        o.position = np.array([*o58, 0.0])
        G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 58).value, o)
    state = {"map_version": version, "holding": holding or {}}
    if epoch is not None:
        state["map_epoch"] = epoch
    G.metadata.set({"heracles": state})
    return G


class _Pub(NS):
    def __init__(self, topic, subscribers=1):
        super().__init__(topic=topic, sent=[], subscribers=subscribers)

    def publish(self, msg):
        self.sent.append(msg)

    def get_subscription_count(self):
        return self.subscribers


def _adaptor(name, husky=False, subscribers=1):
    pub = _Pub(f"/{name}/omniplanner_node/compiled_plan_out", subscribers)
    return NS(
        name=name,
        robot_type="husky" if husky else "spot",
        PLAN_MSG=Path if husky else object,
        plan_pub=pub,
        publish_plan=pub.publish,
    )


@pytest.fixture(autouse=True)
def _no_ros_map_parsing(monkeypatch):
    # The base callback parses a DsgUpdate; these tests set dsg_last directly.
    monkeypatch.setattr(repair.OmniPlannerRos, "dsg_callback", lambda self, msg: None)


def _node(dsg=None, poses=None, adaptors=None):
    """An OmniPlannerRepairRos with its ROS parts faked; logs in node.f_logs."""
    node = N.__new__(N)
    logs = []
    pubs = {}
    timers = []
    clock = {"t": 0.0}

    def publisher(_type, topic, _qos):
        pubs.setdefault(topic, _Pub(topic))
        return pubs[topic]

    def timer(period, cb):
        t = NS(cb=cb, period=period, cancel=lambda: None)
        timers.append(t)
        return t

    log = NS(
        info=lambda m: logs.append(("info", m)),
        warning=lambda m: logs.append(("warning", m)),
        error=lambda m: logs.append(("error", m)),
    )
    node.__dict__.update(
        _constraints_lock=threading.Lock(),
        active_constraints=[],
        _world_lock=threading.Lock(),
        world_tracker=WorldStateTracker(),
        _pending_release={},
        _pending_busy=set(),
        _left_alone=set(),
        _running_plans={},
        _plan_lock=threading.Lock(),
        _plan_deps={},
        _plan_robots=[],
        _pause_pubs={},
        _planning_goal="",
        _active_goal="",
        _pending_changes=[],
        _pending_version=0,
        _pick_failures={},
        _route={},
        _plan_ids={},
        _actions_done={},
        _route_version=None,
        _published_version=None,
        _map_key=None,
        _map_generation=0,
        _pending_epoch="",
        _announced=set(),
        _dispatched={},
        _move_threshold_m=0.5,
        _change_settle_s=1.0,
        _max_pick_retries=2,
        _executor_pause_topic="/{robot}/spot_executor_node/pause",
        _settle_timer=None,
        _announce_grace_s=3.0,
        _progress_timeout_s=60.0,
        _progress_move_m=0.5,
        dsg_lock=threading.Lock(),
        dsg_last=dsg,
        dsg_frame="map",
        robot_adaptors=adaptors
        if adaptors is not None
        else {
            "testbot_a": _adaptor("testbot_a"),
            "testbot_b": _adaptor("testbot_b"),
            "testhusky": _adaptor("testhusky", husky=True),
        },
        f_logs=logs,
        f_pubs=pubs,
        f_timers=timers,
        f_clock=clock,
        f_poses=poses if poses is not None else {},
    )
    node.get_logger = lambda: log
    node.create_publisher = publisher
    node.create_timer = timer
    node.destroy_timer = lambda t: None
    node.get_clock = lambda: NS(
        now=lambda: NS(
            nanoseconds=int(clock["t"] * 1e9),
            to_msg=lambda: NS(sec=int(clock["t"]), nanosec=0),
        )
    )
    node.robot_pose = lambda robot, frame=None: node.f_poses.get(robot)
    node._replan_pub = publisher(None, "~/replan_request", 10)
    node._version_pub = publisher(None, "~/map_version", 1)
    node._progress_pub = publisher(None, "~/progress_alert", 10)
    node._epoch_pub = publisher(None, "~/map_epoch", 1)
    node._path_status_pubs = {}
    return node


def _warnings(node):
    return [m for level, m in node.f_logs if level in ("warning", "error")]


def _path(*pts):
    path = Path()
    for x, y in pts:
        p = PoseStamped()
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        path.poses.append(p)
    return path


def _carry(robot="testbot_a"):
    """testbot_a's pick-and-place of o58: t2 to t4."""
    return ActionSequence(
        "carry",
        robot,
        [
            Follow("map", np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])),
            Pick("map", "box", np.zeros(3), np.array([2.0, 0.2, 0.0]), "o58"),
            Follow("map", np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])),
            Place("map", "box", np.zeros(3), np.array([4.0, 0.0, 0.0]), "o58"),
        ],
    )


def _done(node, robot, action, plan_id="carry", obj="", ok=True):
    node._action_done_callback(
        NS(
            action_type=action,
            success=ok,
            robot_name=robot,
            object_id=obj,
            points=[],
            plan_id=plan_id,
        )
    )


# ---------- P1: a Husky's Path in the plan ----------


def test_finalize_takes_a_path_and_an_action_sequence():
    node = _node(_map())
    plans = {"testhusky": _path((1, 0), (2, 0)), "testbot_a": _carry()}
    out = node.finalize_plans(plans, {})
    assert out is plans
    assert node._plan_ids == {"testhusky": "", "testbot_a": "carry"}
    assert node._plan_robots == ["testbot_a", "testhusky"]
    # only the Spot gets a pause publisher; the Husky has no executor
    assert list(node._pause_pubs) == ["testbot_a"]


def test_put_down_first_skips_a_path_and_takes_the_sequence_id():
    node = _node(_map())
    node._pending_release = {"testbot_a": ["o58"], "testhusky": ["o7"]}
    plans = {"testhusky": _path((1, 0)), "testbot_a": _carry()}
    out = node._put_down_first(plans, {"testbot_a": np.array([2.0, 0.0, 0.0, 0.0])})
    assert isinstance(out["testhusky"], Path)
    assert isinstance(out["testbot_a"].actions[0], Place)
    assert any("testhusky holds" in m for m in _warnings(node))


# ---------- P2 / S1: map epochs ----------


def test_a_reload_moves_the_published_version_on_even_if_heracles_goes_back():
    node = _node(_map("e1", 3))
    for dsg in (_map("e1", 3), _map("e1", 4), _map("e2", 1), _map("e2", 1)):
        node.dsg_last = dsg
        node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    sent = [m.data for m in node.f_pubs["~/map_version"].sent]
    assert sent == [1, 2, 3]  # 3, 4, then the reload's 1: still going up
    assert any("map reloaded (epoch e1 -> e2)" in m for _, m in node.f_logs)
    # The epoch goes out too: found in the local e2e, the goal manager never
    # heard of a reload (the version it watches only goes up).
    assert [m.data for m in node.f_pubs["~/map_epoch"].sent] == ["e1", "e2"]


def test_an_older_heracles_without_an_epoch_still_works():
    node = _node(_map(None, 2))
    assert node._map_state() == ("", 2)


def test_a_reload_rechecks_the_route_and_the_objects():
    node = _node(_map("e1", 4))
    node.finalize_plans({"testbot_a": _carry()}, {"testbot_a": np.zeros(4)})
    node._note_map_state()
    # The reload's map lost t3 and moved o58 3 m: both matter.
    reloaded = _map("e2", 1, places={k: v for k, v in PLACES.items() if k != "t3"})
    node.dsg_last = reloaded
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    (reason,) = [m.data for m in node.f_pubs["~/replan_request"].sent]
    assert "t3" in reason
    assert node.f_pubs["/testbot_a/spot_executor_node/pause"].sent


def test_a_reload_with_moved_objects_replans():
    node = _node(_map("e1", 4))
    node.finalize_plans({"testbot_a": _carry()}, {"testbot_a": np.zeros(4)})
    node._note_map_state()
    node._route = {}  # only the object matters here
    node.dsg_last = _map("e2", 1, o58=(5.0, 3.0))
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    (reason,) = [m.data for m in node.f_pubs["~/replan_request"].sent]
    assert reason.startswith("map reloaded") and "o58" in reason


def test_waiting_for_a_version_ends_when_the_epoch_changes():
    node = _node(_map("e1", 4))
    node._wait_for_map_then_replan(9, "scene changed", epoch="e1")
    assert node.f_pubs["~/replan_request"].sent == []  # 4 < 9: waits
    node.dsg_last = _map("e2", 1)  # reloaded: version 1 is the new map
    node.f_timers[-1].cb()
    assert len(node.f_pubs["~/replan_request"].sent) == 1


def test_every_map_change_is_logged_with_the_route_verdict():
    node = _node(_map("e1", 1))
    node.finalize_plans({"testbot_a": _carry()}, {"testbot_a": np.zeros(4)})
    node._note_map_state()
    node.dsg_last = _map("e1", 2)
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    assert any(
        "map_version 1 -> 2" in m and "route not affected" in m for _, m in node.f_logs
    )


def test_an_unannounced_map_version_is_warned_about():
    node = _node(_map("e1", 1))
    node._note_map_state()
    node.dsg_last = _map("e1", 2)  # e.g. a chat edit removing t5571
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    node.f_timers[-1].cb()
    assert any("no scene change message" in m for m in _warnings(node))


def test_an_announced_map_version_is_not():
    node = _node(_map("e1", 1))
    node._note_map_state()
    node._announced.add(2)
    node.dsg_last = _map("e1", 2)
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))
    node.f_timers[-1].cb()
    assert not any("no scene change message" in m for m in _warnings(node))


# ---------- P4: pausing a Husky ----------


def test_a_husky_is_stopped_with_an_empty_path_and_a_spot_paused():
    node = _node(_map())
    reached = node._pause_robots(["testbot_a", "testhusky"], "scene changed")
    assert reached == ["testbot_a", "testhusky"]
    (stop,) = node.robot_adaptors["testhusky"].plan_pub.sent
    assert isinstance(stop, Path) and stop.poses == []
    assert node.f_pubs["/testbot_a/spot_executor_node/pause"].sent[0].data is True
    assert "/testhusky/spot_executor_node/pause" not in node.f_pubs


def test_only_robots_reached_are_reported_stopped():
    node = _node(
        _map(),
        adaptors={"testhusky_b": _adaptor("testhusky_b", husky=True, subscribers=0)},
    )
    assert node._pause_robots(["testhusky_b"], "map changed") == []
    assert any("could not reach testhusky_b" in m for m in _warnings(node))


# ---------- P5: a Husky's progress ----------


def test_route_places_reads_a_path():
    route = route_places(
        {"testhusky": _path((1, 0), (2, 0), (3, 0))},
        lambda xy: {v: k for k, v in PLACES.items()}.get(xy),
    )
    assert route == {"testhusky": [(0, ["t1", "t2"]), (1, ["t2", "t3"])]}


def test_a_husky_reaching_its_path_visits_places():
    node = _node(_map(), poses={"testhusky": np.array([0.0, 0.0, 0.0, 0.0])})
    node.finalize_plans({"testhusky": _path((1, 0), (2, 0), (3, 0))}, node.f_poses)
    node.f_poses["testhusky"] = np.array([2.0, 0.6, 0.0, 0.0])
    node._progress_tick()
    assert {"t1", "t2"} <= node.world_tracker.visited
    assert node._actions_done["testhusky"] == 1  # past the leg t1 -> t2
    node.f_poses["testhusky"] = np.array([3.0, 0.0, 0.0, 0.0])
    node._progress_tick()
    assert "t3" in node.world_tracker.visited
    assert "testhusky" not in node._dispatched  # at the end of its path
    # ... and the goal manager is told, since a Husky reports no actions.
    sent = node.f_pubs["/testhusky/omniplanner_node/path_status"].sent
    assert [json.loads(m.data)["status"] for m in sent] == ["progress", "done"]


# ---------- P11: a plan that makes no progress ----------


def test_a_robot_that_neither_moves_nor_reports_is_flagged_once():
    node = _node(_map(), poses={"testbot_a": np.array([0.0, 0.0, 0.0, 0.0])})
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    node.f_clock["t"] = 30.0
    node._progress_tick()
    assert node.f_pubs["~/progress_alert"].sent == []
    node.f_clock["t"] = 61.0
    node._progress_tick()
    node._progress_tick()
    (alert,) = [m.data for m in node.f_pubs["~/progress_alert"].sent]
    assert "testbot_a" in alert and "carry" in alert
    assert node.status_warnings() == ["testbot_a stalled on plan carry"]


def test_moving_or_reporting_keeps_the_watchdog_quiet():
    node = _node(_map(), poses={"testbot_a": np.array([0.0, 0.0, 0.0, 0.0])})
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    node.f_clock["t"] = 50.0
    node.f_poses["testbot_a"] = np.array([1.0, 0.0, 0.0, 0.0])  # moved
    node._progress_tick()
    node.f_clock["t"] = 100.0
    _done(node, "testbot_a", "FOLLOW")  # an action result for its plan
    node.f_clock["t"] = 150.0
    node._progress_tick()
    assert node.f_pubs["~/progress_alert"].sent == []


def test_a_moving_spot_is_reported_as_progress():
    """One long follow reports no action until it ends: the goal manager hears
    from here that the Spot is moving, so it does not call it stalled."""
    node = _node(_map(), poses={"testbot_a": np.array([0.0, 0.0, 0.0, 0.0])})
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    node._progress_tick()  # first pose: nothing to compare with yet
    node.f_poses["testbot_a"] = np.array([1.0, 0.0, 0.0, 0.0])
    node._progress_tick()
    node._progress_tick()  # not moved since: nothing new to say
    sent = node.f_pubs["/testbot_a/omniplanner_node/path_status"].sent
    assert [json.loads(m.data) for m in sent] == [
        {"plan_id": "carry", "status": "progress", "reached": 0, "steps": 4}
    ]


# ---------- P13: holding the map has not heard of ----------


def test_a_pick_the_map_missed_is_planned_with(monkeypatch):
    import omniplanner.world_state as ws

    said = []
    monkeypatch.setattr(ws.logger, "warning", lambda *a: said.append(a[0] % a[1:]))
    node = _node(_map(holding={}))  # heracles tracks holding, says none
    _done(node, "testbot_a", "PICK", obj="O58")
    state = node.world_state_for("(object-in-place o58 t4)")
    assert state.holding == {"testbot_a": ["o58"]}
    assert any("nobody holds o58" in m for m in said)
    _done(node, "testbot_a", "PLACE", obj="O58")
    assert node.world_state_for("(object-in-place o58 t4)").holding == {}


def test_once_the_map_agrees_it_is_the_newer_word():
    t = WorldStateTracker()
    t.record("PICK", True, "testbot_a", object_id="o58")
    assert t.snapshot(_map(holding={"testbot_a": ["o58"]})).holding == {
        "testbot_a": ["o58"]
    }
    # the operator emptied the gripper and the map says so
    assert t.snapshot(_map(holding={})).holding == {}


# ---------- P14: busy robots and robots mid-carry ----------


def _busy(*robots):
    return [NS(predicate=BUSY_ROBOTS, symbols=list(robots))]


def _mid_carry():
    """testbot_a has picked o58 and is carrying it; the map missed the pick."""
    node = _node(
        _map(holding={}),
        poses={
            "testbot_a": np.array([2.5, 0.0, 0.0, 0.0]),
            "testhusky": np.array([1.0, 0.0, 0.0, 0.0]),
        },
    )
    node._active_goal = node._planning_goal = "(object-in-place o58 t4)"
    node.finalize_plans({"testbot_a": _carry()}, node.f_poses)
    _done(node, "testbot_a", "FOLLOW")
    _done(node, "testbot_a", "PICK", obj="o58")
    return node


def test_repair_interrupts_a_carrying_robot_and_still_places_the_object():
    node = _mid_carry()
    # a route change: the same goal, replanned (no busy-robots: a repair)
    state = node.world_state_for("(object-in-place o58 t4)")
    assert state.holding == {"testbot_a": ["o58"]}  # planned with o58 in hand
    replanned = ActionSequence(
        "r2",
        "testbot_a",
        [
            Follow("map", np.array([[2.5, 0.0], [4.0, 0.0]])),
            Place("map", "box", np.zeros(3), np.array([4.0, 0.0, 0.0]), "o58"),
        ],
    )
    out = node.finalize_plans({"testbot_a": replanned}, node.f_poses)
    assert out["testbot_a"].actions[-1].object_id == "o58"
    assert any("interrupting testbot_a" in m for m in _warnings(node))


def test_a_new_unrelated_goal_in_repair_puts_the_carried_object_down_first():
    node = _mid_carry()
    node.world_state_for("(at-place testhusky t3)")
    out = node.finalize_plans(
        {
            "testbot_a": ActionSequence("r3", "testbot_a", []),
            "testhusky": _path((3, 0)),
        },
        node.f_poses,
    )
    first = out["testbot_a"].actions[0]
    assert isinstance(first, Place) and first.object_id == "o58"


def test_an_additional_task_plans_only_the_free_robot():
    node = _mid_carry()
    node.world_state_for("(at-place testhusky t3)", constraints=_busy("testbot_a"))
    out = node.finalize_plans(
        {
            "testbot_a": ActionSequence("r4", "testbot_a", []),
            "testhusky": _path((3, 0)),
        },
        node.f_poses,
    )
    assert list(out) == ["testhusky"]  # testbot_a is sent nothing
    # and its carry is still the plan this node watches for it
    assert node._dispatched["testbot_a"]["plan_id"] == "carry"
    assert node._plan_ids["testbot_a"] == "carry"
    assert node._actions_done["testbot_a"] == 2
    assert node._plan_robots == ["testbot_a", "testhusky"]


def test_a_carrying_robot_is_busy_even_if_the_goal_manager_did_not_say():
    node = _mid_carry()
    node.world_state_for("(at-place testhusky t3)", constraints=_busy())
    out = node.finalize_plans(
        {
            "testbot_a": ActionSequence("r5", "testbot_a", []),
            "testhusky": _path((3, 0)),
        },
        node.f_poses,
    )
    assert list(out) == ["testhusky"]


def test_an_additional_task_naming_the_busy_robot_interrupts_it_after_its_put_down():
    node = _mid_carry()
    goal = "(visited-place-by testbot_a t1)"
    node.world_state_for(goal, constraints=_busy("testbot_a"))
    out = node.finalize_plans(
        {
            "testbot_a": ActionSequence(
                "r6", "testbot_a", [Follow("map", np.array([[2.5, 0.0], [1.0, 0.0]]))]
            ),
            "testhusky": _path(),
        },
        node.f_poses,
    )
    kinds = [type(a).__name__ for a in out["testbot_a"].actions]
    assert kinds == ["Place", "Follow"]
    assert out["testbot_a"].actions[0].object_id == "o58"


def test_an_empty_plan_is_never_sent_to_a_carrying_robot():
    node = _mid_carry()
    node.f_poses.pop("testbot_a")  # its pose went stale: no put-down possible
    node.world_state_for("(at-place testhusky t3)")
    out = node.finalize_plans(
        {
            "testbot_a": ActionSequence("r7", "testbot_a", []),
            "testhusky": _path((3, 0)),
        },
        node.f_poses,
    )
    assert "testbot_a" not in out


# ---------- P8: a goal that already holds ----------


def test_all_empty_plans_leave_the_running_plan_alone():
    node = _mid_carry()
    _done(node, "testbot_a", "FOLLOW")
    _done(node, "testbot_a", "PLACE", obj="o58")  # testbot_a is done carrying
    before = dict(node._plan_ids)
    node.world_state_for("(object-in-place o58 t4)")
    out = node.finalize_plans(
        {"testbot_a": ActionSequence("r8", "testbot_a", []), "testhusky": _path()},
        node.f_poses,
    )
    assert all(
        not getattr(p, "actions", getattr(p, "poses", None)) for p in out.values()
    )
    assert node._plan_ids == before  # nothing was committed


def test_a_new_mission_forgets_visits_but_not_holding():
    """P16: a goal starting a new mission is planned from a clean slate, except
    for what the robots physically hold."""
    from omniplanner.world_state import NEW_MISSION

    node = _node(_map())
    node.world_tracker.visited |= {"o1", "t1"}
    node.world_tracker._reported_holding["testbot_a"] = {"o2"}
    state = node.world_state_for("(visited-object o1)", constraints=[])
    assert "o1" in state.visited
    state = node.world_state_for(
        "(visited-object o1)", constraints=[NS(predicate=NEW_MISSION, symbols=[])]
    )
    assert state.visited == set()
    assert node.world_tracker._reported_holding == {"testbot_a": {"o2"}}


# ---------- robot-scoped goals: use-robots leaves the rest of the fleet alone ----------


def _use(*robots):
    return [NS(predicate="use-robots", symbols=list(robots))]


def test_a_use_robots_goal_sends_nothing_to_the_others_and_puts_nothing_down():
    """2026-11-11 (b): a goal for one robot made the carrying robot put its
    object down where it stood."""
    node = _mid_carry()
    state = node.world_state_for(
        "(at-place testhusky t3)", constraints=_use("testhusky")
    )
    assert state.holding == {"testbot_a": ["o58"]}  # still in its hand
    out = node.finalize_plans(
        {
            "testbot_a": ActionSequence("r7", "testbot_a", []),
            "testhusky": _path((3, 0)),
        },
        node.f_poses,
    )
    assert list(out) == ["testhusky"]  # no empty plan, no put-down
    assert node._plan_ids["testbot_a"] == "carry"  # its carry goes on, watched
    assert not any("interrupting" in m for m in _warnings(node))


def test_a_use_robots_robot_holding_what_the_goal_wants_is_planned_with_it():
    node = _mid_carry()
    state = node.world_state_for(
        "(object-in-place o58 t1)", constraints=_use("testbot_a")
    )
    assert state.holding == {"testbot_a": ["o58"]}
    assert node._pending_release == {}
    assert node._pending_busy == {"testbot_b", "testhusky"}
