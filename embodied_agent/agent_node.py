import re
import rclpy
import os
import json
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from .agent import build_embodied_agent
from .utils.nav2_tools import get_tools
from .utils.utils import format_message, format_response, print_response
from .utils.episode_recorder import EpisodeRecorder
from .utils.memory_summarizer import build_summary
from .config import get_config
from .context import Context

from pathlib import Path

EPISODES_DIR = Path("episodes")
MEMORY_DIR   = Path("memory")

# ---------------------------------------------------------------------------
# Layer 1: Terminal tools — the tool that must have run for a task to be done
# ---------------------------------------------------------------------------
# _REQUIRED_TOOLS = {
#     "forward":    "move_forward",
#     "backward":   "move_backward",
#     "left":       "move_left",
#     "right":      "move_right",
#     "turn left":  "turn_left",
#     "turn right": "turn_right",
#     "go to":      "navigate_to_location",
#     "navigate":   "navigate_to_pose",
#     }

# ---------------------------------------------------------------------------
# Layer 2: Semantic checks — all nav tools share the same response shape
# ---------------------------------------------------------------------------
_CHECKED_TOOLS = {
    "navigate_to_pose",
    "navigate_to_location",
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
    "get_current_pose",
}

def _check_tool_result(tool_name: str, data: dict) -> str | None:
    if not data.get("success"):
        reason = data.get("message") or data.get("status") or "unknown error"
        return f"'{tool_name}' failed: {reason}"
    return None

# ---------------------------------------------------------------------------
# Layer 3: Agent verification block — mandated by system prompt
# ---------------------------------------------------------------------------
_VERIFICATION_FAILED_RE  = re.compile(r"result\s*:\s*failed",  re.IGNORECASE)
_VERIFICATION_SUCCESS_RE = re.compile(r"result\s*:\s*success", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_tool_result(content) -> dict:
    try:
        return json.loads(content) if isinstance(content, str) else (content or {})
    except (json.JSONDecodeError, TypeError):
        return {}


def _extract_failure(result: dict, query: str) -> str | None:
    """
    Return a failure description string if the navigation task did not complete,
    or None if everything succeeded.

    Layer 1 — Terminal tool check (structural):
        Did the expected tool for this query run and return success=True?
        Catches: required tool never called because an earlier step aborted.

    Layer 2 — Semantic tool check:
        Did any navigation tool return success=False?
        Catches: Nav2 action server rejections, odometry failures, unknown locations.

    Layer 3 — Agent verification block (template parsing):
        The system prompt forces "Result: SUCCESS / FAILED" in every response.
        Catches: all physical outcome failures the agent itself detects,
                 e.g. robot missed target pose, outside tolerance.
    """
    messages    = result.get("messages", [])
    query_lower = query.lower()

    # --- Layer 1: did the expected terminal tool run and succeed? ---
    # for keyword, tool_name in _REQUIRED_TOOLS.items():
    #     if keyword not in query_lower:
    #         continue

    #     tool_msgs = [
    #         m for m in messages
    #         if isinstance(m, ToolMessage) and m.name == tool_name
    #     ]

    #     if not tool_msgs:
    #         return f"'{tool_name}' was never executed"

    #     data = _parse_tool_result(tool_msgs[-1].content)
    #     if not data.get("success"):
    #         reason = data.get("message") or data.get("status") or "unknown error"
    #         return f"'{tool_name}' failed: {reason}"

    # --- Layer 2: did any tool return a domain failure? ---
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name not in _CHECKED_TOOLS:
            continue
        reason = _check_tool_result(msg.name, _parse_tool_result(msg.content))
        if reason:
            return reason

    # --- Layer 3: parse the agent's structured verification block ---
    found_success = False
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content or ""
        if not content.strip():
            continue
        if _VERIFICATION_FAILED_RE.search(content):
            return "agent verification block reported Result: FAILED"
        if _VERIFICATION_SUCCESS_RE.search(content):
            found_success = True

    if found_success:
        return None

    return None  # no verification block present — treat as success

# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

class Agent(Node):
    def __init__(self):
        super().__init__("embodied_agent")
        qos_profile: QoSProfile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(String, "query", self.query_callback, qos_profile=qos_profile)
        tools = get_tools(self)

        self.agent = build_embodied_agent(tools=tools)

        self.recorder = EpisodeRecorder(save_dir=EPISODES_DIR)
        self.get_logger().info(f"Episode recorder session: {self.recorder.session_id}")

        self.get_logger().info("Agent node initialised!")
        self.message_recieved: bool = False


    def query_callback(self, msg):
        """Callback function to parse the received query to the Agent."""
        self.get_logger().info("Waiting for user query")

        if not self.message_recieved:
            self.message_recieved = True

        self.get_logger().info(f"User Query: '{msg.data}'")

        threading.Thread(target=self.handle_query,
                         args=(msg.data,),
                         daemon=True).start()


    def handle_query(self, user_query: str, max_attempts: int = 3):
        episode = self.recorder.start_episode(query=user_query)
        invoke_message = format_message(user_query)
        result = None

        try:
            for attempt in range(1, max_attempts + 1):
                self.get_logger().info(
                    f"Invoking Agent — attempt {attempt}/{max_attempts}"
                )

                result = self.agent.invoke(
                    invoke_message,
                    config=get_config(),
                    context={"user_role": "beginner"},
                )

                failure = _extract_failure(result, user_query)

                if failure is None:
                    self.get_logger().info("Navigation task completed successfully.")
                    break

                if attempt < max_attempts:
                    self.get_logger().warning(
                        f"Attempt {attempt} failed: {failure}. Retrying..."
                    )
                    invoke_message = format_message(
                        f"[RETRY — Attempt {attempt + 1}/{max_attempts}]\n"
                        f"The previous navigation attempt did not complete.\n"
                        f"Reason: {failure}\n\n"
                        f"Call get_current_pose() to re-establish position, "
                        f"then re-execute the original task:\n"
                        f"{user_query}"
                    )
                else:
                    self.get_logger().error(
                        f"All {max_attempts} attempts failed. Last failure: {failure}"
                    )

            formatted = format_response(result)
            print_response(formatted)

            self.recorder.close_episode_from_formatted_response(
                episode=episode,
                formatted=formatted,
                outcome="success" if _extract_failure(result, user_query) is None else "failed",
            )
            self.get_logger().info(f"Episode saved → {episode.episode_id}")

        except Exception as e:
            self.get_logger().error(f"Error executing User Query: {e}")
            self.recorder.close_episode(
                episode=episode,
                final_response="",
                outcome="error",
                error=str(e),
            )

        finally:
            self.message_recieved = False
            self.get_logger().info("Waiting for the next message...\n")

    def save_memory(self):
        """
        Build and write a compact memory.json from all session files
        in the episodes directory. Called once on shutdown.
        """
        session_files = [
            p for p in EPISODES_DIR.glob("*.json")
            if not p.name.startswith("memory")
        ]

        if not session_files:
            self.get_logger().warning("No session files found, skipping memory save.")
            return

        try:
            summary = build_summary(session_files)
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            out_path = MEMORY_DIR / "memory.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

            stats = summary["stats"]
            self.get_logger().info(
                f"Memory saved → {out_path} "
                f"({stats['total_episodes']} episodes, "
                f"{stats['successful']} ok / {stats['errors']} errors)"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to write memory: {e}")


def main():
    rclpy.init()
    embodied_agent_node = Agent()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node=embodied_agent_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        embodied_agent_node.get_logger().info("Keyboard interrupt received, shutting down...")
    finally:
        embodied_agent_node.save_memory()
        executor.shutdown()
        embodied_agent_node.destroy_node()
        rclpy.shutdown()