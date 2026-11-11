"""omniplanner_node.py must actually take every lock it names in a ``with``.

``with a and b:`` evaluates ``a and b`` (the second lock, since a Lock is
truthy) and enters only that one. The node module cannot be imported without
rclpy, so its source is checked instead.
"""

import ast
from pathlib import Path

NODE = (
    Path(__file__).resolve().parents[1]
    / "omniplanner_ros"
    / "src"
    / "omniplanner_ros"
    / "omniplanner_node.py"
)


def test_no_boolean_expression_as_context_manager():
    tree = ast.parse(NODE.read_text())
    bad = [
        item.context_expr.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.With, ast.AsyncWith))
        for item in node.items
        if isinstance(item.context_expr, ast.BoolOp)
    ]
    assert not bad, f"'with a and b' enters only one lock (lines {bad})"
