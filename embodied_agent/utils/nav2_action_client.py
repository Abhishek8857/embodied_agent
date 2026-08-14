# nav2_action_client.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.duration import Duration
from nav2_msgs.action import NavigateToPose, Spin
from action_msgs.msg import GoalStatus
import tf2_ros
import threading
import math


class Nav2ActionClient(Node):
    def __init__(self, namespace: str = "kairosAB"):
        super().__init__("nav2_agent_client")
        self._namespace = namespace
        self._action_name = f"/{namespace}/navigate_to_pose"
        self._spin_action_name = f"/{namespace}/spin"

        self._client = ActionClient(self, NavigateToPose, self._action_name)
        self._spin_client = ActionClient(self, Spin, self._spin_action_name)

        self._map_frame = "map"
        self._base_frame = f"{namespace}_base_link"

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def _quaternion_to_yaw(self, z: float, w: float) -> float:
        """Convert quaternion (z, w only for planar) to yaw in degrees."""
        yaw_rad = 2.0 * math.atan2(z, w)
        return math.degrees(yaw_rad)

    def get_current_pose(self) -> dict:
        """Return current robot pose in the map frame via TF (map -> base_link)."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._base_frame,
                rclpy.time.Time(),  # latest available transform
                timeout=Duration(seconds=1.0),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            return {"success": False, "message": f"TF lookup failed ({self._map_frame} -> {self._base_frame}): {e}"}

        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = self._quaternion_to_yaw(q.z, q.w)

        return {
            "success": True,
            "x": round(t.x, 4),
            "y": round(t.y, 4),
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

        send_goal_future = self._client.send_goal_async(goal)
        if not self._wait_for_future(send_goal_future, timeout_sec=10.0):
            return {"success": False, "status": "timeout", "message": "Timed out waiting for goal acceptance"}

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return {"success": False, "status": "rejected", "message": "Goal rejected by action server"}

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(result_future, timeout_sec=timeout_sec):
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

    def spin(self, angle_radians: float, time_allowance_sec: float = 15.0, timeout_sec: float = 20.0) -> dict:
        if not self._spin_client.wait_for_server(timeout_sec=5.0):
            return {
                "success": False,
                "status": "server_unavailable",
                "message": f"Action server {self._spin_action_name} not available",
            }

        goal = Spin.Goal()
        goal.target_yaw = angle_radians
        goal.time_allowance = rclpy.duration.Duration(seconds=time_allowance_sec).to_msg()

        self.get_logger().info(f"[Agent] spin → angle={math.degrees(angle_radians):.1f}°")

        send_goal_future = self._spin_client.send_goal_async(goal)
        if not self._wait_for_future(send_goal_future, timeout_sec=10.0):
            return {"success": False, "status": "timeout", "message": "Timed out waiting for goal acceptance"}

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return {"success": False, "status": "rejected", "message": "Spin goal rejected by action server"}

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(result_future, timeout_sec=timeout_sec):
            self.get_logger().warning(f"[Agent] Spin timed out after {timeout_sec}s — cancelling goal")
            goal_handle.cancel_goal_async()
            return {
                "success": False,
                "status": "timeout",
                "message": f"Spin timed out after {timeout_sec}s",
            }

        result = result_future.result()
        status_map = {
            GoalStatus.STATUS_SUCCEEDED: ("succeeded", True),
            GoalStatus.STATUS_ABORTED:   ("aborted",   False),
            GoalStatus.STATUS_CANCELED:  ("canceled",  False),
        }
        status_str, success = status_map.get(result.status, (f"unknown({result.status})", False))

        if success:
            pose = self.get_current_pose()
            message = f"Spin complete, now at x={pose.get('x')}, y={pose.get('y')}, yaw={pose.get('yaw_degrees')}°"
        else:
            message = f"Spin failed: {status_str}"

        return {"success": success, "status": status_str, "message": message}

    def _wait_for_future(self, future, timeout_sec: float) -> bool:
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