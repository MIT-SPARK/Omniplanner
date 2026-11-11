"""Tests for MissingSymbolError in dsg_pddl_grounding."""

import numpy as np
import pytest
import spark_dsg

from dsg_pddl.dsg_pddl_grounding import MissingSymbolError, add_symbol_positions
from dsg_pddl.pddl_grounding import PddlSymbol


@pytest.fixture
def minimal_dsg():
    """Create a minimal DSG with one object node O(1)."""
    G = spark_dsg.DynamicSceneGraph()

    # Create an object node with position
    attrs = spark_dsg.ObjectNodeAttributes()
    attrs.position = np.array([1.0, 2.0, 0.0])
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 1), attrs)

    return G


def test_missing_symbol_error_attributes():
    """Test that MissingSymbolError stores all required attributes."""
    ns = spark_dsg.NodeSymbol("O", 99)
    error = MissingSymbolError(
        symbol=ns,
        pddl_symbol="o99",
        dsg_char="O",
        index=99,
        original=None
    )

    assert error.symbol == ns
    assert error.pddl_symbol == "o99"
    assert error.dsg_char == "O"
    assert error.index == 99
    assert error.original is None


def test_add_symbol_positions_success(minimal_dsg):
    """Test successful position addition for an existing symbol."""
    symbols = [
        PddlSymbol(symbol="o1", layer="object", unary_predicates_to_apply=[])
    ]

    result = add_symbol_positions(minimal_dsg, symbols)

    assert len(result) == 1
    assert result[0].symbol == "o1"
    assert result[0].position is not None
    assert np.allclose(result[0].position, [1.0, 2.0])


def test_add_symbol_positions_missing_symbol(minimal_dsg):
    """Test that MissingSymbolError is raised for non-existent symbol."""
    symbols = [
        PddlSymbol(symbol="o99", layer="object", unary_predicates_to_apply=[])
    ]

    with pytest.raises(MissingSymbolError) as exc_info:
        add_symbol_positions(minimal_dsg, symbols)

    error = exc_info.value
    assert error.dsg_char == "O"
    assert error.index == 99
    assert error.pddl_symbol == "o99"


def test_add_symbol_positions_already_positioned(minimal_dsg):
    """Test that symbols with existing position are skipped."""
    position = np.array([5.0, 6.0])
    symbols = [
        PddlSymbol(
            symbol="o1",
            layer="object",
            unary_predicates_to_apply=[],
            position=position
        )
    ]

    result = add_symbol_positions(minimal_dsg, symbols)

    # Position should remain unchanged
    assert np.array_equal(result[0].position, position)
