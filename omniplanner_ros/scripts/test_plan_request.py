import rclpy
from omniplanner_msgs.msg import PlanRequestStringMsg
from rclpy.node import Node


class ReqSender(Node):
    def __init__(self):
        super().__init__("test_plan_request")
        self.plan_req_pub = self.create_publisher(
            PlanRequestStringMsg, "/omniplanner_node/plan_request_in", 1
        )


rclpy.init()

sender = ReqSender()

pr = PlanRequestStringMsg()
pr.goal = "(and (AtPlace p1) (Holding o1))"

sender.plan_req_pub.publish(pr)
