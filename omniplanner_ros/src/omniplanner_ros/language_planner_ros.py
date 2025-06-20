from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib.resources import as_file, files

import dsg_pddl
import nlu_interface.resources
import spark_config as sc
from dsg_pddl.pddl_grounding import PddlDomain
from nlu_interface.llm_interface import OpenAIWrapper
from omniplanner.language_planner import LanguageDomain, LanguageGoal
from omniplanner.omniplanner import PlanRequest
from omniplanner_msgs.msg import LanguageGoalMsg
from ruamel.yaml import YAML
from std_msgs.msg import String

from omniplanner_ros.omniplanner_node import PluginFeedbackCollector

yaml = YAML(typ="safe")


logger = logging.getLogger(__name__)


class LanguagePlannerRos:
    def __init__(self, config: LanguagePlannerConfig):
        self.config = config

        with as_file(
            files(dsg_pddl.domains).joinpath(config.pddl_domain_name + ".pddl")
        ) as path:
            logger.info(f"Loading domain {path}")
            with open(str(path), "r") as fo:
                # Currently, we have a fixed domain. In the future, could make adjustments based on goal message?
                self.domain = PddlDomain(fo.read())

        llm_config_fp = os.path.expandvars(config.llm_config)
        with open(llm_config_fp, "r") as file:
            self.llm_config = yaml.load(file)

        with as_file(
            files(nlu_interface.resources).joinpath(self.llm_config["prompt"] + ".yaml")
        ) as path:
            logger.info(f'Loading prompt from "{path}"')
            with open(str(path), "r") as file:
                prompt = yaml.load(file)
        self.llm_interface = OpenAIWrapper(
            model=self.llm_config["model"],
            mode=self.llm_config["mode"],
            prompt=prompt,
            num_incontext_examples=self.llm_config["num_incontext_examples"],
            temperature=self.llm_config["temperature"],
            api_timeout=self.llm_config["api_timeout"],
            seed=self.llm_config["seed"],
            api_key_env_var=self.llm_config["api_key_env_var"],
            debug=self.llm_config["debug"],
        )

    def get_plan_callback(self):
        return LanguageGoalMsg, "language_goal", self.language_callback

    def get_plugin_feedback(self, node):
        feedback = PluginFeedbackCollector()
        llm_response_pub = node.create_publisher(String, "~/llm_response", 1)

        def publish_llm_response(goal_string):
            feedback_msg = String()
            feedback_msg.data = goal_string
            llm_response_pub.publish(feedback_msg)

        feedback.publish["llm_response"] = publish_llm_response
        return feedback

    def language_callback(self, msg, robot_poses):
        logger.info("In language_callback()")  # TODO: remove after testing on robot
        goal = LanguageGoal(command=msg.command, robot_id=msg.robot_id)
        domain_type = (
            self.config.domain_type if msg.domain_type == "default" else msg.domain_type
        )
        req = PlanRequest(
            domain=LanguageDomain(domain_type, self.domain, self.llm_interface),
            goal=goal,
            robot_states=robot_poses,
        )
        return req


@sc.register_config(
    "omniplanner_pipeline", name="LanguagePlanner", constructor=LanguagePlannerRos
)
@dataclass
class LanguagePlannerConfig(sc.Config):
    domain_type: str = "Pddl"
    pddl_domain_name: str = "GotoObjectDomain"
    llm_config: str = ""
