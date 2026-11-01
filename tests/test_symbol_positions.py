"""Symbol positions are found for regions whatever their prefix case.

Hydra's open-vocab regions are named r0, r1, ... while its room finder uses
R0, R1, ...; PDDL lowercases both, so both must map back to their node.
"""

import numpy as np
import spark_dsg
from dsg_pddl.dsg_pddl_grounding import add_symbol_positions
from dsg_pddl.pddl_grounding import PddlSymbol


def graph_with_region(prefix):
    G = spark_dsg.DynamicSceneGraph()
    attrs = spark_dsg.SemanticNodeAttributes()
    attrs.position = np.array([1.0, 2.0, 3.0])
    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol(prefix, 0), attrs)
    return G


def test_open_vocab_region_lowercase_prefix():
    G = graph_with_region("r")
    (s,) = add_symbol_positions(G, [PddlSymbol("r0", "region", [])])
    assert np.allclose(s.position, [1.0, 2.0])


def test_room_uppercase_prefix():
    G = graph_with_region("R")
    (s,) = add_symbol_positions(G, [PddlSymbol("r0", "region", [])])
    assert np.allclose(s.position, [1.0, 2.0])
