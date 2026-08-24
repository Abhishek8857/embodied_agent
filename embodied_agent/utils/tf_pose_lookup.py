import tf2_ros
import time 

from typing import Dict, Any
from rclpy.time import Time
from rclpy.duration import Duration
from .utils import quat_to_rpy


class TfPoseLookup:
    """Standard ROS 2 TF lookup listener using tf2_ros.TransformListener."""
    def __init__(self, node, tf_topic=None, tf_static_topic=None):
        self.node = node
        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        # Standard TransformListener manages subscriptions and static TF caching automatically
        self.tf_listener = tf2_ros.TransformListener(self.buffer, self.node)

    def get_pose(self, base_frame: str, ee_frame: str, timeout_s: float = 5.0) -> Dict[str, Any]:
        deadline = time.time() + float(timeout_s)

        while time.time() < deadline:
            try:
                # Time() (time 0) fetches the latest available transform in the buffer
                tf = self.buffer.lookup_transform(
                    base_frame, ee_frame, Time(), timeout=Duration(seconds=0.2)
                )
                t = tf.transform.translation
                q = tf.transform.rotation
                r, p, y = quat_to_rpy(q.x, q.y, q.z, q.w)

                return {
                    "success": True,
                    "base_frame": base_frame,
                    "ee_frame": ee_frame,
                    "translation": {"x": float(t.x), "y": float(t.y), "z": float(t.z)},
                    "quaternion": {"x": float(q.x), "y": float(q.y), "z": float(q.z), "w": float(q.w)},
                    "rpy_rad": {"r": float(r), "p": float(p), "y": float(y)},
                }
            except Exception:
                time.sleep(0.05)

        return {
            "success": False,
            "error_code": "TF_TIMEOUT",
            "error_description": f"TF not available: {base_frame} <- {ee_frame} (timeout {timeout_s}s)",
        }