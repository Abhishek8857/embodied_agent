import threading
import time
from typing import Optional, Dict, Any

from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class GraspPoseCache:
    """
    Keeps the latest PoseStamped from /grasp_pose.

    IMPORTANT behavior (for your request):
    - default_require_new=True: get_latest() will WAIT for the NEXT message
      (won't reuse latched/old message).
    - default_wait_timeout_s: max time it will wait for a new message.
    - returns dict shaped exactly for tools.py expectations:
        {
          "success": True,
          "pose": {"frame_id": ..., "position": {...}, "orientation": {...}},
          "timestamp": <wall time float>,
          "age_s": <float>,
        }
    """

    def __init__(
        self,
        node,
        topic: str = "/grasp_pose",
        qos: Optional[QoSProfile] = None,
        default_wait_timeout_s: float = 15.0,
        default_require_new: bool = True,
    ):
        self.topic = topic
        self.default_wait_timeout_s = float(default_wait_timeout_s)
        self.default_require_new = bool(default_require_new)

        self._lock = threading.Lock()
        self._last_msg: Optional[PoseStamped] = None
        self._last_rx_time: float = 0.0
        self._msg_count: int = 0

        self._event = threading.Event()
        self._cb_group = ReentrantCallbackGroup()

        if qos is None:
            # Matches your publisher (RELIABLE + TRANSIENT_LOCAL) well.
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,  # subscriber can still receive latched messages
            )

        self._sub = node.create_subscription(
            PoseStamped,
            topic,
            self._cb,
            qos,
            callback_group=self._cb_group,
        )

    def _cb(self, msg: PoseStamped):
        with self._lock:
            self._last_msg = msg
            self._last_rx_time = time.time()
            self._msg_count += 1
            self._event.set()

    def get_latest(
        self,
        max_age_s: float = 5.0,
        wait_timeout_s: Optional[float] = None,
        require_new: Optional[bool] = None,
    ) -> Dict[str, Any]:
        max_age_s = float(max_age_s)

        if wait_timeout_s is None:
            wait_timeout_s = self.default_wait_timeout_s
        wait_timeout_s = float(wait_timeout_s)

        if require_new is None:
            require_new = self.default_require_new
        require_new = bool(require_new)

        # Snapshot counter at call-time: require_new means "must receive a new msg after this call"
        with self._lock:
            start_count = self._msg_count

        deadline = time.monotonic() + max(0.0, wait_timeout_s)

        while True:
            with self._lock:
                msg = self._last_msg
                rx_time = self._last_rx_time
                count = self._msg_count

            if msg is not None:
                age = time.time() - rx_time
                is_fresh = age <= max_age_s
                is_new = (count > start_count) if require_new else True

                if is_fresh and is_new:
                    return {
                        "success": True,
                        "pose": {
                            "frame_id": msg.header.frame_id,
                            "position": {
                                "x": float(msg.pose.position.x),
                                "y": float(msg.pose.position.y),
                                "z": float(msg.pose.position.z),
                            },
                            "orientation": {
                                "x": float(msg.pose.orientation.x),
                                "y": float(msg.pose.orientation.y),
                                "z": float(msg.pose.orientation.z),
                                "w": float(msg.pose.orientation.w),
                            },
                        },
                        "timestamp": float(rx_time),
                        "age_s": float(age),
                    }

            # timeout / no-wait
            if wait_timeout_s <= 0.0 or time.monotonic() >= deadline:
                if msg is None:
                    return {
                        "success": False,
                        "error_code": "NO_GRASP_POSE",
                        "error_description": f"No messages received yet on '{self.topic}'.",
                    }

                age = time.time() - rx_time
                if require_new and count == start_count:
                    return {
                        "success": False,
                        "error_code": "NO_NEW_GRASP_POSE",
                        "error_description": (
                            f"No NEW grasp pose arrived on '{self.topic}' within {wait_timeout_s:.1f}s."
                        ),
                        "age_s": float(age),
                    }

                return {
                    "success": False,
                    "error_code": "STALE_GRASP_POSE",
                    "error_description": f"Latest '{self.topic}' is stale (age={age:.3f}s > {max_age_s}s).",
                    "age_s": float(age),
                }

            # wait for new callback
            self._event.clear()
            remaining = max(0.0, deadline - time.monotonic())
            self._event.wait(timeout=min(0.2, remaining))
