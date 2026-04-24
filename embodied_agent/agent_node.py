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
from .tools import get_tools
from .utils.utils import format_message, format_response, print_response
from .utils.episode_recorder import EpisodeRecorder
from .utils.recovery_advisor import RecoveryAdvisor
from .utils.episode_summarizer import build_summary
from .config import get_config
from .llm import get_qwen_llm
from .context import Context

from pathlib import Path

EPISODES_DIR = Path("episodes")
MEMORY_DIR   = Path("memory")


def _extract_failure(result: dict, query: str) -> str | None:
    structured = result.get("structured_response")

    # DEBUG:
    print(f"Structured outcome is {structured.outcome}")

    if structured is None:
        return None  # no structured output — treat as success

    if structured.task_type == "query":
        print(f"Task type {query} detected")
        return None  # informational, never a failure

    if structured.outcome == "failed":
        return structured.failure_reason or "agent reported failure"

    return None


def _extract_completed_steps(result: dict) -> list[str]:
    """
    Walk the message history from the previous attempt and collect
    the names of tool calls that returned successfully.
    Skips any tool whose result contains 'error' or 'failed'.
    """
    completed = []
    messages = result.get("messages", [])

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None)
            content = msg.content or ""

            if tool_name and "error" not in content.lower() and "failed" not in content.lower():
                completed.append(tool_name)

    return completed


class Agent(Node):
    def __init__(self):
        super().__init__("embodied_agent")
        qos_profile: QoSProfile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(String, "query", self.query_callback, qos_profile=qos_profile)
        tools = get_tools(self)

        self.agent = build_embodied_agent(tools=tools)

        self.recorder = EpisodeRecorder(save_dir=EPISODES_DIR)
        self.get_logger().info(f"Episode recorder session: {self.recorder.session_id}")

        # Toggle: set to False to disable the RecoveryAdvisor and retry with plain context only
        self.use_recovery_advisor: bool = False

        self.recovery_advisor = RecoveryAdvisor(
            recorder=self.recorder,
            llm=get_qwen_llm(),
            memory_path=MEMORY_DIR / "memory.json",
        ) if self.use_recovery_advisor else None

        self.get_logger().info(
            f"Agent node initialised! (recovery_advisor={'enabled' if self.use_recovery_advisor else 'disabled'})"
        )
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


    def handle_query(self, user_query: str, max_attempts: int = 2):
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
                    self.get_logger().info("Task completed successfully.")
                    break

                if attempt < max_attempts:
                    # Resolve hint only when the advisor is enabled
                    if self.use_recovery_advisor and self.recovery_advisor is not None:
                        hint = self.recovery_advisor.get_hint(
                            failure_reason=failure,
                            query=user_query,
                        )
                    else:
                        hint = None

                    episode.record_retry(
                        attempt=attempt,
                        failure_reason=failure,
                        hint_used=hint,
                    )

                    if hint:
                        self.get_logger().warning(
                            f"Attempt {attempt} failed: {failure}. Retrying with hint: {hint}"
                        )
                        retry_body = (
                            f"[RETRY — Attempt {attempt + 1}/{max_attempts}]\n"
                            f"The previous attempt did not complete the task. Return to Home pose.\n"
                            f"Reason: {failure}\n\n"
                            f"{hint}\n\n"
                            f"ALWAYS return to home pose first, then re-execute the original task:\n"
                            f"{user_query}"
                        )
                    else:
                        self.get_logger().warning(
                            f"Attempt {attempt} failed: {failure}. Retrying with execution context."
                        )

                        # Build context of what was already accomplished
                        completed_steps = _extract_completed_steps(result)

                        if completed_steps:
                            completed_summary = "\n".join(
                                f"  - {step}" for step in completed_steps
                            )
                            context_block = (
                                f"WHAT WAS ALREADY COMPLETED IN THE PREVIOUS ATTEMPT:\n"
                                f"{completed_summary}\n\n"
                                f"DO NOT repeat these steps. Continue from the point of failure.\n"
                                f"If pick_up_object appears above, the robot is currently holding "
                                f"the object — do NOT re-pick it.\n"
                                f"If move_to_named_pose(home) appears above, the robot is at home.\n"
                            )
                        else:
                            context_block = (
                                f"No steps were completed before the failure. "
                                f"Re-execute the full task from scratch with fresh perception "
                                f"and segmentation.\n"
                            )

                        retry_body = (
                            f"[RETRY — Attempt {attempt + 1}/{max_attempts}]\n"
                            f"The previous attempt failed at the following step:\n"
                            f"Reason: {failure}\n\n"
                            f"{context_block}"
                            f"(check completed steps above). Then continue the original task:\n"
                            f"{user_query}"
                        )

                    invoke_message = format_message(retry_body)

                else:
                    self.get_logger().error(
                        f"All {max_attempts} attempts failed. Last failure: {failure}"
                    )

            formatted = format_response(result)
            print_response(formatted)

            self.recorder.close_episode_from_formatted_response(
                episode=episode,
                formatted=formatted,
                outcome="success",
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
        session_files = [
            p for p in EPISODES_DIR.glob("*.json")
            if not p.name.startswith("memory")
        ]

        if not session_files:
            self.get_logger().warning("No session files found, skipping memory save.")
            return

        try:
            summary = build_summary(session_files)

            # build_summary returns {} when all session files are empty
            if not summary:
                self.get_logger().info("No episodes recorded this session, skipping memory save.")
                return

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