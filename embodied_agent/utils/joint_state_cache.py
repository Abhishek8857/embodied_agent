import math
import threading
import time

from typing import Optional, Dict, Any, List
from sensor_msgs.msg import JointState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class JointStateCache:
    """Subscribes to /joint_states, sorts arm joints 1-7, and filters out gripper joints."""

    # Explicit order required by MoveIt / Controller
    TARGET_ARM_JOINTS = [
        "joint_1", "joint_2", "joint_3", 
        "joint_4", "joint_5", "joint_6", "joint_7"
    ]

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

    @staticmethod
    def _wrap(a: float) -> float:
        """Wrap a single angle to [-pi, +pi]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

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

        # Create explicit mapping from joint name -> value
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}

        # Extract positions strictly in numerical order joint_1 -> joint_7
        ordered_positions = []
        ordered_velocities = []

        for name in self.TARGET_ARM_JOINTS:
            if name in name_to_pos:
                ordered_positions.append(name_to_pos[name])
                ordered_velocities.append(name_to_vel.get(name, 0.0))
            else:
                return {
                    "success": False,
                    "error_code": "MISSING_JOINT",
                    "error_description": f"Required joint '{name}' not found in /joint_states message.",
                }

        normalized = [self._wrap(p) for p in ordered_positions]

        return {
            "success": True,
            "age_s": float(age),
            "name": self.TARGET_ARM_JOINTS,
            "position": ordered_positions,
            "normalized_position": [round(p, 4) for p in normalized],
            "velocity": ordered_velocities,
        }