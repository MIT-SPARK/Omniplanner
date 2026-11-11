"""The repair node's subscription callbacks must match the base node's.

The base subscribes ``self.dsg_callback`` to raw DsgUpdate messages; an override
with another signature only fails when the first map arrives, killing the node.
"""

import inspect

import pytest

omniplanner_node = pytest.importorskip("omniplanner_ros.omniplanner_node")
repair_node = pytest.importorskip("omniplanner_ros.omniplanner_repair_node")


def test_dsg_callback_takes_the_message_like_the_base():
    base = inspect.signature(omniplanner_node.OmniPlannerRos.dsg_callback)
    repair = inspect.signature(repair_node.OmniPlannerRepairRos.dsg_callback)
    assert list(repair.parameters) == list(base.parameters)


def test_plan_ids_accept_a_husky_path_plan():
    """A multi-robot plan mixes Spot ActionSequences (plan_id) and husky
    nav_msgs/Path plans (none); finalize_plans used to raise AttributeError."""
    from types import SimpleNamespace as NS

    ids = repair_node.plan_ids({"euclid": NS(plan_id="p1"), "gauss": object()})
    assert ids == {"euclid": "p1", "gauss": ""}
