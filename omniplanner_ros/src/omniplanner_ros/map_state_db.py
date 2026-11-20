"""Read heracles' map state (``_MapState``) from the map database.

The planner gets its map from the heracles publisher, which holds the last
map while a rebuild runs -- but a map it read just as a rebuild started is
half built, and nothing in the map itself says so. The ``_MapState`` node
does: ``rebuilding_since`` while a rebuild or reload runs, ``rebuild_failed``
when one failed after the wipe, and the map's ``version`` and ``epoch``.

heracles' own readers (``heracles.map_reload``) are used when heracles is
installed; otherwise the node is read directly, with the same rule for a mark
left by a dead writer (``REBUILD_STALE_S``).
"""

from __future__ import annotations

import time
from typing import Optional

MAP_STATE = "_MapState"
# heracles.map_reload.REBUILD_STALE_S: an older mark was left by a dead writer.
REBUILD_STALE_S = 120.0


class MapStateDb:
    """{rebuilding, failed, version, epoch} of the map, or None if unreadable.

    Connects on first use and again after a failure; every query has a short
    timeout, so an unreachable database delays a check by at most that, and
    after a failure it is not tried again for retry_after_s (None meanwhile).
    """

    def __init__(
        self,
        uri: str,
        auth,
        timeout_s: float = 2.0,
        database="neo4j",
        retry_after_s: float = 30.0,
    ):
        self.uri = uri
        self.auth = auth
        self.timeout_s = float(timeout_s)
        self.database = database
        self.retry_after_s = float(retry_after_s)
        self.driver = None
        self.error = ""  # why the last read failed ("" if it did not)
        self._failed_at = None

    def query(self, query: str) -> list:
        """Rows as dicts: what heracles' readers expect of a db."""
        import neo4j

        if self.driver is None:
            self.driver = neo4j.GraphDatabase.driver(
                self.uri,
                auth=tuple(self.auth),
                connection_timeout=self.timeout_s,
                connection_acquisition_timeout=self.timeout_s,
                max_transaction_retry_time=0.0,
            )
        records, _, _ = self.driver.execute_query(
            neo4j.Query(query, timeout=self.timeout_s), database_=self.database
        )
        return [r.data() for r in records]

    def read(self) -> Optional[dict]:
        now = time.monotonic()
        if self._failed_at is not None and now - self._failed_at < self.retry_after_s:
            return None
        try:
            state = self._read()
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__
            self._failed_at = now
            self.close()
            return None
        self.error = ""
        self._failed_at = None
        return state

    def _read(self) -> dict:
        try:
            from heracles.map_reload import (
                get_map_epoch,
                get_map_version,
                rebuild_state,
            )
        except ImportError:
            return self._read_direct()
        running, failed = rebuild_state(self)
        return {
            "rebuilding": bool(running),
            "failed": failed or "",
            "version": get_map_version(self),
            "epoch": get_map_epoch(self),
        }

    def _read_direct(self) -> dict:
        rows = self.query(
            f"MATCH (m:{MAP_STATE}) RETURN m.rebuilding_since AS t, "
            "m.rebuild_failed AS f, m.version AS v, m.epoch AS e"
        )
        row = rows[0] if rows else {}
        since = row.get("t")
        return {
            "rebuilding": since is not None
            and time.time() - float(since) < REBUILD_STALE_S,
            "failed": row.get("f") or "",
            "version": int(row.get("v") or 0),
            "epoch": row.get("e") or "",
        }

    def close(self) -> None:
        driver, self.driver = self.driver, None
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass
