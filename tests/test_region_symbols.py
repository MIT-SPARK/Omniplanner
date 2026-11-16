"""PDDL region symbols resolve to the graph's own region nodes at planning time.

Hydra's rooms layer names regions R(i); hydra_multi's open-vocabulary regions
(offline fusion) name them r(i). Planning must ground either.
"""

import numpy as np
import spark_dsg
from dsg_pddl.pddl_utils import pddl_symbol_to_node_symbol

from omniplanner.omniplanner import DsgContextProvider


def _graph(region_char):
    G = spark_dsg.DynamicSceneGraph()
    attrs = spark_dsg.RoomNodeAttributes()
    attrs.position = np.array([1.0, 2.0, 0.0])
    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol(region_char, 0), attrs)
    return G


def test_regions_follow_the_graph():
    assert pddl_symbol_to_node_symbol(_graph("R"), "r0") == spark_dsg.NodeSymbol("R", 0)
    assert pddl_symbol_to_node_symbol(_graph("r"), "r0") == spark_dsg.NodeSymbol("r", 0)
    # a region the graph doesn't have keeps the canonical name (the error names R(i))
    assert pddl_symbol_to_node_symbol(_graph("r"), "r7") == spark_dsg.NodeSymbol("R", 7)
    # other layers are unchanged
    assert pddl_symbol_to_node_symbol(_graph("r"), "o3") == spark_dsg.NodeSymbol("O", 3)


def test_plan_context_finds_lowercase_regions():
    ctx = DsgContextProvider(_graph("r"))
    assert "r0" in ctx
    assert np.allclose(ctx["r0"]["position"][:2], [1.0, 2.0])


def _place_graph(place_char):
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    attrs = spark_dsg.PlaceNodeAttributes()
    attrs.position = np.array([1.0, 2.0, 0.0])
    G.add_node(
        spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol(place_char, 5).value, attrs
    )
    return G


def test_mesh_places_follow_the_graph():
    # Hydra's mesh places are T(i): t5 must name T(5), not a t(5) the graph lacks
    assert pddl_symbol_to_node_symbol(_place_graph("T"), "t5") == spark_dsg.NodeSymbol("T", 5)
    # graphs built with lowercase places still resolve
    assert pddl_symbol_to_node_symbol(_place_graph("t"), "t5") == spark_dsg.NodeSymbol("t", 5)
    # a place the graph doesn't have gets the canonical T(i)
    assert pddl_symbol_to_node_symbol(_place_graph("t"), "t9") == spark_dsg.NodeSymbol("T", 9)
