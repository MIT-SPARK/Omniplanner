"""Plan repair while heracles rebuilds the map (Penn field trial, Nov 20).

The scene change writer rebuilt the map in Neo4j (delete every map node,
reload the prior, replay the changes). The planner got a map read mid-rebuild:
places without their edges. Its route check found "euclid's route from t5770
to t5764 is no longer connected", paused every robot and replanned; the replan
found no plan on the half-built map (0 operators), and the goal manager
stopped euclid, gauss and hamilton -- for edges that were there before and
after.

A map not settled (a rebuild running or failed, the database at another map
than the planner's, or -- without the database -- a map far smaller than the
last) is not judged: it is checked again until it settles. No robot is
stopped for it.
"""

import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
import spark_dsg

from omniplanner.plan_validity import unsettled_map

repair = pytest.importorskip("omniplanner_ros.omniplanner_repair_node")
willow = pytest.importorskip("test_willow_repair_node")
N = repair.OmniPlannerRepairRos
_node, _carry = willow._node, willow._carry


@pytest.fixture(autouse=True)
def _no_ros_map_parsing(monkeypatch):
    monkeypatch.setattr(repair.OmniPlannerRos, "dsg_callback", lambda self, msg: None)


def _map(epoch="e1", version=1, n=30, cut=(), places=None):
    """Places t1..tn, 1 m apart along x, each joined to the next; cut: pairs
    (a, b) not joined; None for no edges at all (a map half loaded)."""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    names = places if places is not None else [f"t{i}" for i in range(1, n + 1)]
    ids = {}
    for name in names:
        a = spark_dsg.PlaceNodeAttributes()
        a.position = np.array([float(name[1:]), 0.0, 0.0])
        nid = spark_dsg.NodeSymbol("t", int(name[1:])).value
        G.add_node(spark_dsg.DsgLayers.MESH_PLACES, nid, a)
        ids[name] = nid
    if cut is not None:
        for a, b in zip(names, names[1:]):
            if int(b[1:]) - int(a[1:]) == 1 and (a, b) not in cut:
                G.insert_edge(ids[a], ids[b])
    o = spark_dsg.KhronosObjectAttributes()
    o.position = np.array([2.0, 0.2, 0.0])  # o58, at t2
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 58).value, o)
    G.metadata.set({"heracles": {"map_version": version, "map_epoch": epoch}})
    return G


class _Db:
    """A fake map database: what heracles' _MapState says."""

    uri = "fake"
    error = ""

    def __init__(self, version=1, epoch="e1", rebuilding=False, failed=""):
        self.state = dict(
            version=version, epoch=epoch, rebuilding=rebuilding, failed=failed
        )

    def read(self):
        return dict(self.state)


def _running(db=None, dsg=None):
    """testbot_a carrying o58 from t2 to t4, on a whole settled map."""
    node = _node(dsg or _map())
    node._map_db = db
    node.finalize_plans({"testbot_a": _carry()}, {"testbot_a": np.zeros(4)})
    node._note_map_state()
    node._map_unsettled()  # the map it planned on, judged settled
    return node


def _new_map(node, dsg):
    node.dsg_last = dsg
    node.dsg_callback(NS(full_update=True, layer_contents=None, header=None))


def _replans(node):
    return [m.data for m in node.f_pubs["~/replan_request"].sent]


def _pauses(node):
    return node.f_pubs.get("/testbot_a/spot_executor_node/pause", NS(sent=[])).sent


def _alerts(node):
    return [m.data for m in node.f_pubs["~/progress_alert"].sent]


# ---------- what a settled map is ----------


def test_a_map_is_not_settled_while_it_is_rebuilt_or_its_rebuild_failed():
    db = {"rebuilding": True, "failed": "", "version": 4, "epoch": "e1"}
    assert "being rebuilt" in unsettled_map(("e1", 4), db, (30, 29), None)
    db = {"rebuilding": False, "failed": "boom", "version": 4, "epoch": "e1"}
    assert "rebuild failed (boom)" in unsettled_map(("e1", 4), db, (30, 29), None)


def test_a_map_is_not_settled_while_the_database_holds_another():
    db = {"rebuilding": False, "failed": "", "version": 5, "epoch": "e1"}
    assert "newer map" in unsettled_map(("e1", 4), db, (30, 29), None)
    db["epoch"] = "e2"
    db["version"] = 4
    assert unsettled_map(("e1", 4), db, (30, 29), None)
    db["epoch"] = "e1"
    assert unsettled_map(("e1", 4), db, (30, 29), None) == ""


def test_the_database_trumps_the_map_size():
    # A real removal of half the map, on a settled database, is the map.
    db = {"rebuilding": False, "failed": "", "version": 4, "epoch": "e1"}
    assert unsettled_map(("e1", 4), db, (10, 2), ("e1", (30, 29))) == ""


def test_without_the_database_a_much_smaller_map_looks_half_loaded():
    settled = ("e1", (30, 29))
    assert "place edges" in unsettled_map(("e1", 5), None, (30, 0), settled)
    assert "places" in unsettled_map(("e1", 5), None, (3, 2), settled)
    assert "no places" in unsettled_map(("e1", 5), None, (0, 0), None)
    assert unsettled_map(("e1", 5), None, (29, 27), settled) == ""
    # another epoch is another map: its size says nothing
    assert unsettled_map(("e2", 1), None, (3, 2), settled) == ""
    assert unsettled_map(("e1", 5), None, None, settled) == ""  # no map yet


# ---------- the route check ----------


def test_a_route_cut_on_a_map_read_mid_rebuild_stops_no_robot():
    """The field trial: the map arrives without the edges on the route while
    heracles still rebuilds; once the rebuild is done they are all there."""
    db = _Db(version=1)
    node = _running(db)
    db.state.update(rebuilding=True)
    _new_map(node, _map(version=2, cut=[("t2", "t3"), ("t3", "t4")]))
    assert _replans(node) == [] and _pauses(node) == []
    assert any("not judging the route" in m for m in willow._warnings(node))
    recheck = node._map_wait()["timer"]
    recheck.cb()  # still rebuilding: still waiting
    assert _replans(node) == [] and _pauses(node) == []
    # The rebuild ends (version bumped) and its map arrives: nothing was cut.
    db.state.update(rebuilding=False, version=3)
    recheck.cb()  # the database is ahead of the planner's map: still waiting
    assert _replans(node) == []
    _new_map(node, _map(version=3))
    recheck.cb()
    assert _replans(node) == [] and _pauses(node) == []
    assert any("route not affected" in m for _, m in node.f_logs)
    assert node._map_wait()["timer"] is None


def test_a_route_cut_for_real_by_the_rebuild_is_replanned_once_it_is_done():
    db = _Db(version=1)
    node = _running(db)
    db.state.update(rebuilding=True)
    _new_map(node, _map(version=2, cut=None))
    assert _replans(node) == []
    # The rebuild retired t3 for good.
    db.state.update(rebuilding=False, version=3)
    _new_map(node, _map(version=3, places=["t1", "t2", "t4", "t5"]))
    (reason,) = _replans(node)
    assert "t3" in reason and _pauses(node)


def test_a_real_cut_on_a_stable_map_is_replanned_as_before():
    db = _Db(version=1)
    node = _running(db)
    db.state.update(version=2)
    _new_map(node, _map(version=2, cut=[("t3", "t4")]))
    (reason,) = _replans(node)
    assert "from t3 to t4 is no longer connected" in reason
    assert _pauses(node)[0].data is True


def test_a_real_cut_without_the_database_is_replanned_as_before():
    node = _running(None)
    _new_map(node, _map(version=2, cut=[("t3", "t4")]))
    (reason,) = _replans(node)
    assert "from t3 to t4 is no longer connected" in reason


def test_a_map_that_changes_during_the_check_is_checked_again():
    db = _Db(version=1)
    node = _running(db)
    db.state.update(version=2)
    mesh_places = N._mesh_places

    def newer_map_lands(dsg):
        # read with the half map, then a newer map replaces it mid-check
        out = mesh_places(dsg)
        node.dsg_last = _map(version=3)
        db.state.update(version=3)
        return out

    node._mesh_places = newer_map_lands
    _new_map(node, _map(version=2, cut=[("t2", "t3")]))
    assert _replans(node) == [] and _pauses(node) == []
    assert node._route_version is None  # owed a check
    node._mesh_places = mesh_places
    node._note_map_state()  # the planner takes in the newer map
    node._map_wait()["timer"].cb()
    assert _replans(node) == [] and _pauses(node) == []
    assert any("route not affected" in m for _, m in node.f_logs)


def test_the_database_moving_on_during_the_check_is_checked_again():
    db = _Db(version=1)
    node = _running(db)
    db.state.update(version=2)
    reads = []

    def read():
        reads.append(1)
        if len(reads) == 2:  # the check after the route was found cut
            db.state.update(rebuilding=True)
        return dict(db.state)

    node._map_db = NS(read=read, error="", uri="fake")
    _new_map(node, _map(version=2, cut=[("t2", "t3")]))
    assert _replans(node) == [] and _pauses(node) == []
    assert node._map_wait()["timer"] is not None


def test_without_the_database_a_half_loaded_map_is_waited_on():
    node = _running(None)
    _new_map(node, _map(version=2, cut=None))  # places, no edges yet
    assert _replans(node) == [] and _pauses(node) == []
    _new_map(node, _map(version=3))  # loaded
    assert _replans(node) == [] and _pauses(node) == []
    assert node._map_wait()["timer"] is None


def test_without_the_database_a_smaller_map_is_taken_after_the_timeout():
    node = _running(None)
    _new_map(node, _map(version=2, n=10))  # 20 places removed for real
    assert _replans(node) == []
    node.f_clock["t"] = 121.0
    node._map_wait()["timer"].cb()
    # judged now: t1..t10 still carry the route
    assert any("taking the map as it is" in m for m in willow._warnings(node))
    assert any("route not affected" in m for _, m in node.f_logs)


def test_a_rebuild_that_does_not_end_is_told_but_stops_no_robot():
    db = _Db(version=1)
    node = _running(db)
    db.state.update(rebuilding=True)
    _new_map(node, _map(version=2, cut=None))
    recheck = node._map_wait()["timer"]
    node.f_clock["t"] = 60.0
    recheck.cb()
    assert _alerts(node) == []
    node.f_clock["t"] = 121.0
    recheck.cb()
    recheck.cb()
    (alert,) = _alerts(node)
    assert "being rebuilt" in alert and "no robot was stopped" in alert
    assert _replans(node) == [] and _pauses(node) == []


def test_a_failed_rebuild_is_told_at_once_but_stops_no_robot():
    db = _Db(version=1)
    node = _running(db)
    db.state.update(failed="rebuild failed after the map was wiped")
    _new_map(node, _map(version=2, cut=None))
    node._map_wait()["timer"].cb()
    (alert,) = _alerts(node)
    assert "rebuild failed" in alert
    assert _replans(node) == [] and _pauses(node) == []


def test_a_reload_read_mid_load_is_judged_once_settled():
    db = _Db(version=4, epoch="e1")
    node = _running(db, _map("e1", 4))
    db.state.update(rebuilding=True, epoch="e2", version=5)
    half = _map("e2", 5, cut=None)
    half.remove_node(spark_dsg.NodeSymbol("O", 58).value)  # not loaded yet
    _new_map(node, half)
    assert _replans(node) == [] and _pauses(node) == []
    db.state.update(rebuilding=False)
    node.dsg_last = _map("e2", 5)  # the same map version, now whole
    node._map_wait()["timer"].cb()
    assert _replans(node) == [] and _pauses(node) == []
    (settled,) = [m for _, m in node.f_logs if m.startswith("map settled")]
    assert "route not affected" in settled
    assert "objects the plan acts on unchanged: ['o58']" in settled


# ---------- a plan that failed on a map not settled ----------


def test_a_plan_failing_mid_rebuild_is_planned_again_after_it():
    db = _Db(version=1)
    node = _running(db)
    db.state.update(rebuilding=True)
    retried = threading.Event()
    exc = ValueError("Fast Downward: the goal is unreachable; 0 operators, 2 facts")
    assert node.defer_plan_failure("plugin", exc, retried.set) is True
    recheck = node._map_wait()["timer"]
    recheck.cb()
    assert not retried.is_set()
    db.state.update(rebuilding=False)
    recheck.cb()
    assert retried.wait(2.0)
    assert node._map_wait()["timer"] is None


def test_a_plan_failing_on_a_settled_map_is_reported():
    node = _running(_Db(version=1))
    assert node.defer_plan_failure("plugin", ValueError("no"), lambda: None) is False


def test_the_planner_holds_back_a_failure_its_subclass_defers():
    """OmniPlannerRos.plan_goal: a deferred failure is not published on
    ~/plan_failed (which the goal manager answers by stopping the robots)."""
    base = repair.OmniPlannerRos
    node = base.__new__(base)
    handlers, failed, deferred = [], [], []
    node.__dict__.update(
        feedback=NS(plugin_feedback_collectors={}),
        goal_count_lock=threading.Lock(),
        goal_count=0,
        planning_lock=threading.Lock(),
        current_planner_lock=threading.Lock(),
        plan_time_start_lock=threading.Lock(),
        current_planner=None,
        plan_time_start=None,
        dsg_last=object(),
        dsg_frame="map",
        goal_group=None,
        plan_failed_pub=NS(publish=failed.append),
        plan_status_pub=NS(publish=lambda m: None),
    )
    node.get_logger = lambda: NS(info=print, warning=print, error=print)
    node.get_robot_poses = lambda frame: {}
    node.create_subscription = lambda *a, **k: handlers.append(a[2])

    def plan(msg, poses):
        raise ValueError("the goal is unreachable")

    plugin = NS(
        get_plan_callback=lambda: (object, "goal", plan),
        get_plugin_feedback=lambda n: None,
    )
    node.defer_plan_failure = lambda name, exc, retry: deferred.append(retry) or True
    base.register_plugin(node, "p", plugin)
    handlers[0](NS())
    assert failed == [] and len(deferred) == 1
    node.defer_plan_failure = lambda name, exc, retry: False
    deferred[0]()  # the retry plans it again, and this time it is reported
    assert [m.data for m in failed] == ["p: the goal is unreachable"]
