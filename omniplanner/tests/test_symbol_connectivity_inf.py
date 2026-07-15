"""Regression: symbol_connectivity_to_pddl must tolerate disconnected (inf) pairs.

A live-detected object can snap to a disconnected component of the place graph,
so LayerPlanner.get_external_distance returns np.inf. The old code did
``int(dist)`` unconditionally and crashed with OverflowError. Disconnected pairs
must be omitted (no fabricated ``connected``/``distance`` facts), leaving the POI
reachable via any other finite connection and only truly-isolated objects
unreachable.
"""
import numpy as np

from dsg_pddl.dsg_pddl_grounding import symbol_connectivity_to_pddl


class _Info:
    def __init__(self, symbol):
        self.symbol = symbol


def _pair(a, b, dist):
    return (_Info(a), _Info(b), dist)


def test_finite_pair_emits_connected_and_two_distances():
    out = symbol_connectivity_to_pddl([_pair("o1", "o2", 3.7)])
    assert ("connected", "o1", "o2") in out
    assert ("=", ("distance", "o1", "o2"), 3) in out  # int() truncation preserved
    assert ("=", ("distance", "o2", "o1"), 3) in out


def test_infinite_pair_is_skipped_not_crashing():
    # np.inf (get_shortest_distance's no-path sentinel) must not raise.
    out = symbol_connectivity_to_pddl([_pair("o1", "o2", np.inf)])
    assert out == []


def test_mixed_keeps_finite_drops_infinite():
    out = symbol_connectivity_to_pddl([
        _pair("o1", "o2", 2.0),
        _pair("o1", "o3", float("inf")),
        _pair("o2", "o3", 5.9),
    ])
    connected = {t for t in out if t[0] == "connected"}
    assert ("connected", "o1", "o2") in connected
    assert ("connected", "o2", "o3") in connected
    assert ("connected", "o1", "o3") not in connected  # dropped
