"""Reading heracles' _MapState (omniplanner_ros.map_state_db).

The live test needs a THROWAWAY Neo4j (it writes _MapState there), e.g.
``docker run -d --rm -p 127.0.0.1:7799:7687 -e NEO4J_AUTH=neo4j/testpass1
neo4j:5.25.1`` and ``MAP_STATE_TEST_NEO4J=neo4j://127.0.0.1:7799`` (user
neo4j, password testpass1, or MAP_STATE_TEST_NEO4J_AUTH=user/pass). Never
point it at a map in use.
"""

import os
import sys
import time

import pytest

from omniplanner_ros import map_state_db
from omniplanner_ros.map_state_db import MapStateDb


@pytest.fixture
def no_heracles(monkeypatch):
    # the direct read, as where heracles is not installed
    monkeypatch.setitem(sys.modules, "heracles.map_reload", None)


def _db(rows):
    db = MapStateDb("neo4j://nowhere:1", ("neo4j", "x"))
    db.query = lambda q: rows
    return db


def test_the_direct_read(no_heracles):
    now = time.time()
    state = _db([{"t": now, "f": None, "v": 7, "e": "abc"}]).read()
    assert state == {"rebuilding": True, "failed": "", "version": 7, "epoch": "abc"}
    stale = now - map_state_db.REBUILD_STALE_S - 1  # left by a dead writer
    state = _db([{"t": stale, "f": "boom", "v": 7, "e": None}]).read()
    assert state == {"rebuilding": False, "failed": "boom", "version": 7, "epoch": ""}
    assert _db([]).read() == {
        "rebuilding": False,
        "failed": "",
        "version": 0,
        "epoch": "",
    }


def test_an_unreachable_database_reads_as_unknown(no_heracles):
    db = MapStateDb("neo4j://127.0.0.1:1", ("neo4j", "x"), timeout_s=0.5)
    assert db.read() is None and db.error
    # not tried again for a while: no delay on every map
    db.query = lambda q: pytest.fail("queried during the back-off")
    assert db.read() is None
    db.retry_after_s = 0.0
    db.query = lambda q: [{"t": None, "f": None, "v": 2, "e": "e1"}]
    assert db.read()["version"] == 2 and db.error == ""


URI = os.environ.get("MAP_STATE_TEST_NEO4J", "")


@pytest.mark.skipif(not URI, reason="set MAP_STATE_TEST_NEO4J to a throwaway Neo4j")
@pytest.mark.parametrize("heracles", [False, True])
def test_a_live_map_state(monkeypatch, heracles):
    if heracles:
        pytest.importorskip("heracles.map_reload")
    else:
        monkeypatch.setitem(sys.modules, "heracles.map_reload", None)
    auth = tuple(
        os.environ.get("MAP_STATE_TEST_NEO4J_AUTH", "neo4j/testpass1").split("/")
    )
    db = MapStateDb(URI, auth)
    try:
        db.query("MATCH (m:_MapState) DETACH DELETE m")
        db.query(
            "CREATE (:_MapState {version: 4, epoch: 'e1', "
            f"rebuilding_since: {time.time()}}})"
        )
        assert db.read() == {
            "rebuilding": True,
            "failed": "",
            "version": 4,
            "epoch": "e1",
        }
        db.query(
            "MATCH (m:_MapState) SET m.rebuilding_since = null, m.version = 5, "
            "m.rebuild_failed = 'wiped'"
        )
        assert db.read() == {
            "rebuilding": False,
            "failed": "wiped",
            "version": 5,
            "epoch": "e1",
        }
    finally:
        db.close()
