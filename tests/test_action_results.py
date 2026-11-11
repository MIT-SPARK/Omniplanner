"""Per-action result subscription: optional message, off for Huskies, capped."""

import pytest

node_mod = pytest.importorskip("omniplanner_ros.omniplanner_node")


class _Log:
    def __init__(self):
        self.warnings = []

    def info(self, *a):
        pass

    def warning(self, m):
        self.warnings.append(m)


class _Node:
    def __init__(self):
        self.log = _Log()
        self.subs = []

    def get_logger(self):
        return self.log

    def create_publisher(self, *a):
        return object()

    def create_subscription(self, *a):
        self.subs.append(a)
        return object()


class _Cfg:
    robot_name = "bot"
    robot_type = "spot"
    body_frame = "base"
    action_result_topic = ""


class _Msg:
    action_index = 0
    action_type = "goto"
    status = "ok"
    detail = ""

    def __init__(self, plan_id):
        self.plan_id = plan_id


def _make(cls):
    node = _Node()
    return cls(_Cfg(), node=node), node


def test_cap_keeps_newest_plans():
    ad, _ = _make(node_mod.RobotPlanningAdaptor)
    n = node_mod.MAX_RESULT_PLANS
    assert n == 20
    for i in range(n + 5):
        ad._on_action_result(_Msg(f"p{i}"))
        ad._on_action_result(_Msg(f"p{i}"))
    assert list(ad.action_results) == [f"p{i}" for i in range(5, n + 5)]
    assert all(len(v) == 2 for v in ad.action_results.values())


def test_phoenix_flag_off_and_no_subscription():
    assert node_mod.PhoenixPlanningAdaptor.PUBLISHES_ACTION_RESULTS is False
    ad, node = _make(node_mod.PhoenixPlanningAdaptor)
    assert ad.action_result_sub is None and node.subs == []


def test_missing_message_type_skips_subscription(monkeypatch):
    monkeypatch.setattr(node_mod, "ActionResultMsg", None)
    ad, node = _make(node_mod.RobotPlanningAdaptor)
    assert ad.action_result_sub is None and node.subs == []
    assert any("per-action results are off" in w for w in node.log.warnings)


def test_subscribes_when_available():
    if node_mod.ActionResultMsg is None:
        pytest.skip("spot_tools without ActionResultMsg")
    ad, node = _make(node_mod.RobotPlanningAdaptor)
    assert node.subs[0][1] == "/bot/spot_executor/action_result"


def test_import_is_optional_in_source():
    src = open(node_mod.__file__).read()
    assert "import ActionSequenceMsg\n" in src
    assert "except ImportError" in src and "ActionResultMsg = None" in src
