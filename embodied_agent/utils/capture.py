# embodied_agent/utils/capture.py

import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def _timestamp_name(prefix: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}{ext}"


class OneShotImageSubscriber:
    """Subscribe to an Image topic, capture exactly one message, then stop."""
    def __init__(self, node, topic: str):
        self.node = node
        self.topic = topic
        self.bridge = CvBridge()

        self._event = threading.Event()
        self._msg: Optional[Image] = None

        self._cb_group = ReentrantCallbackGroup()
        self._qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,  # common for camera streams
            durability=DurabilityPolicy.VOLATILE,
        )

        self._sub = node.create_subscription(
            Image,
            topic,
            self._cb,
            self._qos,
            callback_group=self._cb_group,
        )

    def _cb(self, msg: Image):
        self._msg = msg
        self._event.set()

    def capture(self, timeout_s: float = 2.0) -> Optional[Image]:
        ok = self._event.wait(timeout=float(timeout_s))
        # cleanup subscription immediately
        try:
            self.node.destroy_subscription(self._sub)
        except Exception:
            pass
        return self._msg if ok else None


def capture_rgb_image(node,
                  save_dir: str = "captures/rgb",
                  topic: str = "/camera/color/image_raw",
                  filename: Optional[str] = None,
                  timeout_s: float = 2.0) -> str:
    """
    Capture one RGB image and save it as JPG.
    Returns the saved filepath.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = OneShotImageSubscriber(node, topic)
    msg = sub.capture(timeout_s=timeout_s)
    if msg is None:
        raise TimeoutError(f"No RGB image received on '{topic}' within {timeout_s}s")

    cv_img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")

    if filename is None:
        filename = "rgb.jpg"

    out_path = out_dir / filename
    if out_path.exists():
        out_path.unlink()
        
    ok = cv2.imwrite(str(out_path), cv_img)
    if not ok:
        raise RuntimeError(f"Failed to write image to '{out_path}'")

    return str(out_path)


def capture_raw_depth_image(node,
                        save_dir: str = "captures/depth",
                        topic: str = "/camera/depth/image_rect_raw",
                        filename_prefix: str = "depth",
                        timeout_s: float = 2.0) -> Tuple[str, Optional[str]]:
    """
    Capture one depth image and save:
      - always: raw depth as .npy (lossless)
      - optionally: .png if depth is uint16 (16UC1)
    Returns: (npy_path, png_path_or_None)
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = OneShotImageSubscriber(node, topic)
    msg = sub.capture(timeout_s=timeout_s)  
    if msg is None:
        raise TimeoutError(f"No depth image received on '{topic}' within {timeout_s}s")

    depth = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
    depth_np = np.array(depth)

    npy_path = out_dir / f"{filename_prefix}.npy"
    
    if npy_path.exists():
        npy_path.unlink()
        
    np.save(str(npy_path), depth_np)

    png_path = None
    fixed_png_path = out_dir / f"{filename_prefix}.png"

    # Save 16-bit PNG only if uint16 (typical for 16UC1)
    if depth_np.dtype == np.uint16:
        if fixed_png_path.exists():
            fixed_png_path.unlink()
        ok = cv2.imwrite(str(fixed_png_path), depth_np)
        png_path = str(fixed_png_path) if ok else None
    else:
        if fixed_png_path.exists():
            fixed_png_path.unlink()
            
    return str(npy_path), (str(png_path) if png_path else None)
