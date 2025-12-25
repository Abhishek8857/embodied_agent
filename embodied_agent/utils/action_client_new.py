import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from agent_action_interface.action import ExecuteMotion


class MotionClient(Node):
    
    def __init__(self):
        super.__init__("execute_motion_client")