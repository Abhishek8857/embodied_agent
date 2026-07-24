import tf2_ros
import time 

from typing import Dict, Any
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, HistoryPolicy, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from .utils import quat_to_rpy


class TfPoseLookup:
    """TF lookup that matches typical /tf QoS (BEST_EFFORT) and stays alive during LLM blocking."""
    def __init__(self, node, tf_topic="/tf", tf_static_topic="/tf_static"):
        self.node = node
        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self._cb_group = ReentrantCallbackGroup()

        qos_tf = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,   
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_tf_static = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL, 
        )

        self._sub_tf = node.create_subscription(
            TFMessage, tf_topic, self._tf_cb, qos_tf, callback_group=self._cb_group
        )
        self._sub_tf_static = node.create_subscription(
            TFMessage, tf_static_topic, self._tf_static_cb, qos_tf_static, callback_group=self._cb_group
        )

    def _tf_cb(self, msg: TFMessage):
        for t in msg.transforms:
            self.buffer.set_transform(t, "tf_listener")

    def _tf_static_cb(self, msg: TFMessage):
        for t in msg.transforms:
            self.buffer.set_transform_static(t, "tf_listener")

    def get_pose(self, base_frame: str, ee_frame: str, timeout_s: float = 5.0) -> Dict[str, Any]:
        deadline = time.time() + float(timeout_s)

        while time.time() < deadline:
            try:
                tf = self.buffer.lookup_transform(
                    base_frame, ee_frame, Time(), timeout=Duration(seconds=0.05)
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
                time.sleep(0.01)

        return {
            "success": False,
            "error_code": "TF_TIMEOUT",
            "error_description": f"TF not available: {base_frame} <- {ee_frame} (timeout {timeout_s}s)",
        }
        