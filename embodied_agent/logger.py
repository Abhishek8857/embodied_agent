import rclpy
from rclpy.node import Node

from action_msgs.msg import GoalStatusArray
from moveit_msgs.action import ExecuteTrajectory
from std_msgs.msg import String

class ExecMonitor(Node):
    def __init__(self):
        super().__init__("exec_monitor")

        self.pub = self.create_publisher(String, "moveit_execution_status", 10)

        # Subscribe to feedback
        self.create_subscription(
            ExecuteTrajectory.FeedbackMessage,
            "/execute_trajectory/_action/feedback",
            self.feedback_cb,
            10
        )

        # Subscribe to result
        self.create_subscription(
            ExecuteTrajectory.Result,
            "/execute_trajectory/_action/result",
            self.result_cb,
            10
        )

    def feedback_cb(self, msg):
        self.get_logger().debug("Feedback received.")

    def result_cb(self, msg):
        code = msg.error_code.val
        status = "SUCCESS" if code == 1 else f"ERROR_{code}"
        self.pub.publish(String(data=status))
        self.get_logger().info(f"MoveIt execution result: " + status)


def main():
    rclpy.init()
    node = ExecMonitor()
    rclpy.spin(node)
    rclpy.shutdown()
