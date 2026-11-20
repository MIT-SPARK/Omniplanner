"""Test the compiler's task boundaries without importing the full planning stack.

The production function is loaded from its AST because importing the ROS plugin
registry also imports the DSG/LLM toolchain. Message serialization and action
sequencing are separately exercised by the isolated ROS integration tests.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest

SOURCE = Path(__file__).parents[1] / 'omniplanner_ros/src/omniplanner_ros/pddl_planner_ros.py'


def compiler():
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'compile_phoenix_tasks')
    namespace = {'uuid': uuid, 'logger': logging.getLogger(__name__),
                 'Follow': SimpleNamespace, 'ActionSequence': SimpleNamespace}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['compile_phoenix_tasks']


def test_outbound_and_return_keep_separate_goals():
    outbound = [[0., 0.], [5., 0.], [10., 0.]]
    returning = [[10., 0.], [5., 0.], [0., 0.]]
    context = SimpleNamespace(value=SimpleNamespace(
        symbolic_actions=[('goto-poi', 'turnaround'), ('goto-poi', 'start')],
        parameterized_actions=[outbound, returning]))
    result = compiler()(context, 'gauss', 'map')
    assert result.robot_name == 'gauss' and result.plan_id
    assert len(result.actions) == 2
    assert result.actions[0].path2d == outbound
    assert result.actions[1].path2d == returning
    assert all(action.frame == 'map' for action in result.actions)


def test_repeated_points_do_not_create_extra_tasks():
    path = [[0., 0.], [0., 0.], [1., 0.], [1., 0.]]
    context = SimpleNamespace(value=SimpleNamespace(
        symbolic_actions=[('goto-poi', 'a'), ('inspect', 'a')],
        parameterized_actions=[path, []]))
    result = compiler()(context, 'gauss', 'map')
    assert len(result.actions) == 1
    assert result.actions[0].path2d == path


def test_mismatched_grounding_is_rejected():
    context = SimpleNamespace(value=SimpleNamespace(
        symbolic_actions=[('goto-poi', 'a'), ('goto-poi', 'b')], parameterized_actions=[[]]))
    with pytest.raises(ValueError): compiler()(context, 'gauss', 'map')


def test_no_navigation_tasks_is_rejected():
    context = SimpleNamespace(value=SimpleNamespace(
        symbolic_actions=[('inspect', 'a')], parameterized_actions=[[]]))
    with pytest.raises(ValueError): compiler()(context, 'gauss', 'map')


@pytest.mark.parametrize('enabled', [False, True])
def test_opt_in_never_publishes_tasks_to_legacy_path_topic(enabled):
    from nav_msgs.msg import Path as PathMsg
    from robot_executor_msgs.msg import ActionSequenceMsg
    source = SOURCE.parent / 'omniplanner_node.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'PhoenixPlanningAdaptor')
    class Publisher:
        def __init__(self): self.messages = []
        def publish(self, message): self.messages.append(message)
    class Node:
        def create_publisher(self, *args): return Publisher()
    class Base:
        def __init__(self, config, **kwargs):
            self.name = config.robot_name
            self.plan_pub = Publisher()
        def publish_plan(self, message): self.plan_pub.publish(message)
    scope = {'RobotPlanningAdaptor': Base, 'Path': PathMsg, 'ActionSequenceMsg': ActionSequenceMsg}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), scope)
    adaptor = scope['PhoenixPlanningAdaptor'](
        SimpleNamespace(robot_name='gauss', sequential_tasks=enabled), node=Node())
    if enabled:
        message = ActionSequenceMsg()
        adaptor.publish_plan(message)
        assert adaptor.task_pub.messages == [message]
        assert adaptor.plan_pub.messages == []
        with pytest.raises(TypeError): adaptor.publish_plan(PathMsg())
    else:
        message = PathMsg()
        adaptor.publish_plan(message)
        assert adaptor.plan_pub.messages == [message]
        assert adaptor.task_pub is None
