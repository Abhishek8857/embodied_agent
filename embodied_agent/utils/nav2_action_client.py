# nav2_action_client.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatus
import threading
import math


class Nav2ActionClient(Node):
    def __init__(self, namespace: str = "kairosAB"):
        super().__init__("nav2_agent_client")
        self._namespace = namespace
        self._action_name = f"/{namespace}/navigate_to_pose"

        self._client = ActionClient(self, NavigateToPose, self._action_name)

        # Subscribe to odometry for get_current_pose()
        self._latest_odom = None
        self.create_subscription(
            Odometry,
            f"/{namespace}/robotnik_base_control/odom",
            self._odom_callback,
            10,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _odom_callback(self, msg: Odometry):
        self._latest_odom = msg

    def _quaternion_to_yaw(self, z: float, w: float) -> float:
        """Convert quaternion (z, w only for planar) to yaw in degrees."""
        yaw_rad = 2.0 * math.atan2(z, w)
        return math.degrees(yaw_rad)

    def get_current_pose(self) -> dict:
        """Return current robot pose from odometry."""
        if self._latest_odom is None:
            return {"success": False, "message": "No odometry received yet"}

        pos = self._latest_odom.pose.pose.position
        ori = self._latest_odom.pose.pose.orientation
        yaw = self._quaternion_to_yaw(ori.z, ori.w)

        return {
            "success": True,
            "x": round(pos.x, 4),
            "y": round(pos.y, 4),
            "yaw_degrees": round(yaw, 2),
        }

    def navigate_to_pose(
    self,
    x: float,
    y: float,
    yaw_degrees: float = 0.0,
    frame_id: str = "map",
    timeout_sec: float = 45.0,
    ) -> dict:
        if not self._client.wait_for_server(timeout_sec=5.0):
            return {
                "success": False,
                "status": "server_unavailable",
                "message": f"Action server {self._action_name} not available",
            }

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        yaw_rad = math.radians(yaw_degrees)
        goal.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)

        self.get_logger().info(f"[Agent] navigate_to_pose → x={x}, y={y}, yaw={yaw_degrees}°")

        # Send goal asynchronously and wait with timeout
        send_goal_future = self._client.send_goal_async(goal)
        if not self._wait_for_future(send_goal_future, timeout_sec=10.0):
            return {"success": False, "status": "timeout", "message": "Timed out waiting for goal acceptance"}

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return {"success": False, "status": "rejected", "message": "Goal rejected by action server"}

        # Wait for result with the main timeout
        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(result_future, timeout_sec=timeout_sec):
            # Cancel the goal so the robot doesn't keep trying
            self.get_logger().warning(f"[Agent] Navigation timed out after {timeout_sec}s — cancelling goal")
            goal_handle.cancel_goal_async()
            return {
                "success": False,
                "status": "timeout",
                "message": f"Navigation timed out after {timeout_sec}s",
            }

        result = result_future.result()
        status_map = {
            GoalStatus.STATUS_SUCCEEDED: ("succeeded", True),
            GoalStatus.STATUS_ABORTED:   ("aborted",   False),
            GoalStatus.STATUS_CANCELED:  ("canceled",  False),
        }
        status_str, success = status_map.get(result.status, (f"unknown({result.status})", False))
        message = (
            f"Reached x={x:.2f}, y={y:.2f}, yaw={yaw_degrees:.1f}°"
            if success else f"Navigation failed: {status_str}"
        )
        return {"success": success, "status": status_str, "message": message}


    def _wait_for_future(self, future, timeout_sec: float) -> bool:
        """Spin until the future completes or timeout is reached. Returns True if completed."""
        import time
        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                return False
            time.sleep(0.05)
        return True
    
    def cancel_navigation(self) -> dict:
        self._client._cancel_goal()
        return {"success": True, "status": "canceled", "message": "Navigation canceled"}

    def shutdown(self):
        self._executor.shutdown()
        self.destroy_node()