import threading
import time

from typing import Optional, Dict, Any
from sensor_msgs.msg import JointState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy




class JointStateCache:
    """Subscribes to a JointState topic and keeps the latest message."""
    def __init__(self, node, topic: str = "/joint_states", qos: Optional[QoSProfile] = None):
        self.topic = topic
        self._lock = threading.Lock()
        self._last_msg: Optional[JointState] = None
        self._last_time: float = 0.0
        self._cb_group = ReentrantCallbackGroup()

        if qos is None:
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )

        self._sub = node.create_subscription(
            JointState,
            topic,
            self._cb,
            qos,
            callback_group=self._cb_group,
        )

    def _cb(self, msg: JointState):
        with self._lock:
            self._last_msg = msg
            self._last_time = time.time()

    def get_latest(self, max_age_s: float = 1.0) -> Dict[str, Any]:
        with self._lock:
            msg = self._last_msg
            age = time.time() - self._last_time if msg is not None else float("inf")

        if msg is None:
            return {
                "success": False,
                "error_code": "NO_JOINT_STATES",
                "error_description": f"No messages received yet on '{self.topic}'.",
            }

        if age > max_age_s:
            return {
                "success": False,
                "error_code": "STALE_JOINT_STATES",
                "error_description": f"Latest '{self.topic}' is stale (age={age:.3f}s > {max_age_s}s).",
            }

        return {
            "success": True,
            "age_s": float(age),
            "name": list(msg.name),
            "position": list(msg.position),
            "velocity": list(msg.velocity) if msg.velocity else [],
            "effort": list(msg.effort) if msg.effort else [],
        }       
        
        