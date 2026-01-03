#!/usr/bin/env python3
"""
Blocking (synchronous) ROS 2 Action client for `agent_action_interface/action/ExecuteMotion`.

Designed to be called from an AI agent tool layer:

- Tool builds `data: List[float]` where:
    data[0] = 0.0 -> joint target   (data[1:] = joint values)
    data[0] = 1.0 -> pose target    (data[1:] = x,y,z,qx,qy,qz,qw)
    data[0] = 2.0 -> gripper cmd    (data[1]  = command value)

- Client sends the goal, blocks until result (or timeout),
  and returns a JSON-serializable dict with:
    success/failure, error_code, error_description, final status, and feedback trace.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from action_msgs.msg import GoalStatus

from agent_action_interface.action import ExecuteMotion


_STATUS_MAP = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


@dataclass
class MotionResult:
    ok: bool
    goal_accepted: bool
    final_status: str
    error_code: str
    error_description: str
    feedback: List[Dict[str, Any]]
    sent_data: List[float]
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.ok),
            "goal_accepted": bool(self.goal_accepted),
            "final_status": self.final_status,
            "error_code": self.error_code,
            "error_description": self.error_description,
            "feedback": self.feedback,
            "sent_data": self.sent_data,
            "elapsed_s": self.elapsed_s,
        }


class ExecuteMotionClient:
    """
    Thread-spun ROS 2 action client that offers a blocking .send() API.
    Safe to call from synchronous AI-agent tool functions.
    """

    def __init__(
        self,
        action_name: str = "execute_motion",
        node_name: str = "execute_motion_client",
        server_wait_timeout_s: float = 10.0,
        expected_pose_len: int = 7,
        expected_gripper_len: int = 1,
        expected_joint_len: Optional[int] = None,
        logger_level: Optional[int] = None,
    ) -> None:
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self._node = Node(node_name)
        if logger_level is not None:
            self._node.get_logger().set_level(logger_level)

        self._expected_pose_len = int(expected_pose_len)
        self._expected_gripper_len = int(expected_gripper_len)
        self._expected_joint_len = expected_joint_len if expected_joint_len is None else int(expected_joint_len)

        self._action_client = ActionClient(self._node, ExecuteMotion, action_name)

        # Spin in background so futures + feedback callbacks complete.
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_stop_evt = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        if not self._action_client.wait_for_server(timeout_sec=server_wait_timeout_s):
            self._node.get_logger().error(
                f"Action server '{action_name}' not available after {server_wait_timeout_s:.1f}s"
            )

        self._closed = False
        self._send_lock = threading.Lock()

    def close(self) -> None:
        """Stop background spin and shutdown rclpy if we initialized it."""
        if self._closed:
            return
        self._closed = True

        try:
            self._spin_stop_evt.set()
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=2.0)
        except Exception:
            pass

        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass

        try:
            self._node.destroy_node()
        except Exception:
            pass

        if self._owns_rclpy:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def send(
        self,
        data: Sequence[float],
        *,
        timeout_s: float = 120.0,
        wait_for_server_s: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Send a goal (data array) and block until a result (or timeout).
        Returns a JSON-serializable dict.
        """
        if self._closed:
            return MotionResult(
                ok=False,
                goal_accepted=False,
                final_status="CLIENT_CLOSED",
                error_code="CLIENT_CLOSED",
                error_description="ExecuteMotionClient.close() has been called.",
                feedback=[],
                sent_data=list(data),
                elapsed_s=0.0,
            ).to_dict()

        with self._send_lock:

            data_list = [float(x) for x in data]
            if len(data_list) == 0:
                return MotionResult(
                    ok=False,
                    goal_accepted=False,
                    final_status="INVALID_GOAL",
                    error_code="INVALID_GOAL",
                    error_description="Goal data array is empty.",
                    feedback=[],
                    sent_data=data_list,
                    elapsed_s=0.0,
                ).to_dict()

            if not self._action_client.wait_for_server(timeout_sec=wait_for_server_s):
                return MotionResult(
                    ok=False,
                    goal_accepted=False,
                    final_status="SERVER_UNAVAILABLE",
                    error_code="SERVER_UNAVAILABLE",
                    error_description=f"Action server not available after {wait_for_server_s:.1f}s.",
                    feedback=[],
                    sent_data=data_list,
                    elapsed_s=0.0,
                ).to_dict()

            self._validate_payload(data_list)

            goal_msg = ExecuteMotion.Goal()
            goal_msg.data = data_list

            feedback_trace: List[Dict[str, Any]] = []
            fb_lock = threading.Lock()

            def feedback_cb(feedback_msg: ExecuteMotion.FeedbackMessage) -> None:
                entry = {
                    "t": time.time(),
                    "state": str(feedback_msg.feedback.state),
                    "progress": float(feedback_msg.feedback.progress),
                }
                with fb_lock:
                    feedback_trace.append(entry)

            t0 = time.monotonic()

            # 1) Send goal
            goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=feedback_cb)
            goal_evt = threading.Event()
            goal_future.add_done_callback(lambda _f: goal_evt.set())

            if not goal_evt.wait(timeout=timeout_s):
                return MotionResult(
                    ok=False,
                    goal_accepted=False,
                    final_status="TIMEOUT",
                    error_code="TIMEOUT",
                    error_description=f"Timed out waiting for goal acceptance after {timeout_s:.1f}s.",
                    feedback=self._safe_copy_feedback(feedback_trace, fb_lock),
                    sent_data=data_list,
                    elapsed_s=time.monotonic() - t0,
                ).to_dict()

            goal_handle = goal_future.result()
            if goal_handle is None:
                return MotionResult(
                    ok=False,
                    goal_accepted=False,
                    final_status="GOAL_ERROR",
                    error_code="GOAL_ERROR",
                    error_description="Goal future returned None (unexpected).",
                    feedback=self._safe_copy_feedback(feedback_trace, fb_lock),
                    sent_data=data_list,
                    elapsed_s=time.monotonic() - t0,
                ).to_dict()

            if not goal_handle.accepted:
                return MotionResult(
                    ok=False,
                    goal_accepted=False,
                    final_status="REJECTED",
                    error_code="REJECTED",
                    error_description="Goal was rejected by the server.",
                    feedback=self._safe_copy_feedback(feedback_trace, fb_lock),
                    sent_data=data_list,
                    elapsed_s=time.monotonic() - t0,
                ).to_dict()

            # 2) Wait for result
            remaining = max(0.0, timeout_s - (time.monotonic() - t0))
            result_future = goal_handle.get_result_async()
            res_evt = threading.Event()
            result_future.add_done_callback(lambda _f: res_evt.set())

            if not res_evt.wait(timeout=remaining):
                return MotionResult(
                    ok=False,
                    goal_accepted=True,
                    final_status="TIMEOUT",
                    error_code="TIMEOUT",
                    error_description=f"Timed out waiting for result after {timeout_s:.1f}s.",
                    feedback=self._safe_copy_feedback(feedback_trace, fb_lock),
                    sent_data=data_list,
                    elapsed_s=time.monotonic() - t0,
                ).to_dict()

            wrapped = result_future.result()
            status_str = _STATUS_MAP.get(int(wrapped.status), str(int(wrapped.status)))
            res: ExecuteMotion.Result = wrapped.result

            ok = bool(res.success) and status_str == "SUCCEEDED"
            return MotionResult(
                ok=ok,
                goal_accepted=True,
                final_status=status_str,
                error_code=str(res.error_code),
                error_description=str(res.error_description),
                feedback=self._safe_copy_feedback(feedback_trace, fb_lock),
                sent_data=data_list,
                elapsed_s=time.monotonic() - t0,
            ).to_dict()

    # Convenience wrappers (optional)
    def send_joint_target(self, joints: Sequence[float], *, timeout_s: float = 120.0) -> Dict[str, Any]:
        return self.send([0.0] + [float(x) for x in joints], timeout_s=timeout_s)

    def send_pose_target(
        self,
        x: float, y: float, z: float,
        qx: float, qy: float, qz: float, qw: float,
        *,
        timeout_s: float = 120.0
    ) -> Dict[str, Any]:
        return self.send([1.0, x, y, z, qx, qy, qz, qw], timeout_s=timeout_s)

    def send_gripper_command(self, value: float, *, timeout_s: float = 60.0) -> Dict[str, Any]:
        return self.send([2.0, float(value)], timeout_s=timeout_s)

    def _spin(self) -> None:
        while not self._spin_stop_evt.is_set() and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    @staticmethod
    def _safe_copy_feedback(trace: List[Dict[str, Any]], lock: threading.Lock) -> List[Dict[str, Any]]:
        with lock:
            return list(trace)

    def _validate_payload(self, data_list: List[float]) -> None:
        flag = int(data_list[0])
        n = len(data_list) - 1
        if flag not in (0, 1, 2):
            self._node.get_logger().warn(
                f"Unknown command flag {data_list[0]} (expected 0,1,2). Sending anyway."
            )
            return

        if flag == 0 and self._expected_joint_len is not None and n != self._expected_joint_len:
            self._node.get_logger().warn(
                f"Joint target has {n} values, expected {self._expected_joint_len}. Sending anyway."
            )
        if flag == 1 and n != self._expected_pose_len:
            self._node.get_logger().warn(
                f"Pose target has {n} values, expected {self._expected_pose_len} (x,y,z,qx,qy,qz,qw). Sending anyway."
            )
        if flag == 2 and n != self._expected_gripper_len:
            self._node.get_logger().warn(
                f"Gripper command has {n} values, expected {self._expected_gripper_len}. Sending anyway."
            )




def main() -> None:
    """Optional CLI sanity check (requires running action server)."""
    motion = ExecuteMotionClient(expected_joint_len=None)
    try:
            res = motion.send([0.0, 0.0, -0.7650, -3.15, -2.13, 0.006, -1.2, 1.55], timeout_s=120.0)
            print(res)
    finally:
        motion.close()


if __name__ == "__main__":
    main()



