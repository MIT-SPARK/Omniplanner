"""A real goal manager node, named and configured as on willow, driven by
calling its callbacks: willow is the base station, the robots come from an
omniplanner_plugins.yaml like the deployed one."""

import json
from types import SimpleNamespace as NS

import rclpy
import yaml
from rclpy.parameter import Parameter
from robot_executor_msgs.msg import ActionDoneMsg, ActionMsg, ActionSequenceMsg
from std_msgs.msg import Int64, String

from omniplanner_msgs.msg import ConstrainedPddlGoalMsg, ConstraintFact
from omniplanner_ros.goal_manager_node import GoalManager

SPOTS = ("euclid", "hamilton", "hilbert")
HUSKIES = ("gauss",)


def robots_config(tmp_path, spots=SPOTS, huskies=HUSKIES):
    robots = [
        {"robot_name": r, "type": "robot_executor", "robot_type": "spot"} for r in spots
    ] + [{"robot_name": r, "type": "phoenix", "robot_type": "husky"} for r in huskies]
    path = tmp_path / "omniplanner_plugins.yaml"
    path.write_text(yaml.safe_dump({"robots": robots, "planners": {}}))
    return str(path)


# A ROS domain of its own: a robot stack may be running on this machine, and a
# test must never reach its executors.
TEST_DOMAIN_ID = 97


class Harness:
    def __init__(
        self, tmp_path, record_executor=True, spots=SPOTS, huskies=HUSKIES, **params
    ):
        if not rclpy.ok():
            rclpy.init(domain_id=TEST_DOMAIN_ID)
        self.huskies = set(huskies)
        config = robots_config(tmp_path, spots, huskies)
        overrides = [Parameter("plugin_config_path", value=config)]
        overrides += [Parameter(k, value=v) for k, v in params.items()]
        self.node = GoalManager(namespace="willow", parameter_overrides=overrides)
        n = self.node
        self.t = 1000.0
        n._now = lambda: self.t
        self.goals, self.texts, self.sent, self.logs = [], [], [], []
        self.unreachable = set()
        n._goal_pub = NS(publish=self.goals.append)
        n._display_pub = NS(publish=lambda m: self.texts.append(m.data))
        n._alert_pub = NS(publish=lambda m: None)
        log = NS(
            info=lambda m, **_: self.logs.append(m),
            warning=lambda m, **_: self.logs.append(m),
            error=lambda m, **_: self.logs.append(m),
        )
        n.get_logger = lambda: log
        if record_executor:

            def cmd(robot, verb):
                self.sent.append((robot, verb))
                return robot not in self.unreachable

            n._executor_cmd = cmd

    def close(self):
        self.node.destroy_node()

    # ------------- inputs -------------

    def goal(self, text, robot_id="willow", forbidden=()):
        msg = ConstrainedPddlGoalMsg()
        msg.goal.robot_id, msg.goal.pddl_goal = robot_id, text
        msg.constraints = [
            ConstraintFact(predicate="forbidden-poi", symbols=[s]) for s in forbidden
        ]
        return msg

    def send(self, text, **kw):
        msg = self.goal(text, **kw)
        self.node._commanded_goal_cb(msg)
        return msg

    def plan(self, **robots):
        """The planner's answer: {robot: (plan_id, n_actions, coverage)}, sent
        in the planner's order: sequences, plan_robots, then coverage."""
        for robot, (plan_id, n, _) in robots.items():
            if robot in self.huskies:
                continue
            seq = ActionSequenceMsg(plan_id=plan_id, robot_name=robot)
            seq.actions = [ActionMsg(action_type="FOLLOW") for _ in range(n)]
            self.node._sequence_cb(robot, seq)
        self.node._plan_robots_cb(String(data=json.dumps(sorted(robots))))
        for robot, (_, _, coverage) in robots.items():
            self.node._robot_coverage_cb(robot, String(data=json.dumps(coverage)))

    def done(self, robot, plan_id, success=True, action="FOLLOW", obj=""):
        self.node._action_done_cb(
            ActionDoneMsg(
                plan_id=plan_id,
                robot_name=robot,
                action_type=action,
                success=success,
                object_id=obj,
            )
        )

    def map_version(self, v):
        self.node._map_version_cb(Int64(data=v))

    def failed(self, text):
        self.node._planner_failed_cb(
            String(data=f"multi_robot_pddl_constrained: {text}")
        )

    # ------------- outputs -------------

    def busy(self, msg):
        return [
            list(c.symbols) for c in msg.constraints if c.predicate == "busy-robots"
        ]
