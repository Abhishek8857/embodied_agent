import rclpy
import os
import threading
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String 
from .agent import build_embodied_agent
from .tools import get_tools
from .utils.utils import format_message, format_response, print_response
from .utils.episode_recorder import EpisodeRecorder
from .utils.memory_summarizer import build_summary
from .config import get_config
from .context import Context

import json
from pathlib import Path

EPISODES_DIR = Path("episodes")
MEMORY_DIR   = Path("memory")


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
        """Callback function to parse the recieved query to the Agent"""
        self.get_logger().info("Waiting of user query")
        
        if not self.message_recieved:
            self.message_recieved = True
        
        self.get_logger().info(f"User Query: '{msg.data}'")
                
        threading.Thread(target=self.handle_query, 
                            args=(msg.data,), 
                            daemon=True).start()


    def handle_query(self, user_query: str):
        episode = self.recorder.start_episode(query=user_query)

        try:
            self.get_logger().info(f"Invoking Agent with User Query: {user_query}")

            response = self.agent.invoke(
                format_message(user_query),
                config=get_config(),
                context={"user_role": "beginner"},
            )

            formatted = format_response(response)
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
        embodied_agent_node.get_logger().info("Keyboard interrupt recieved, shutting down...")
    finally:
        embodied_agent_node.save_memory()
        executor.shutdown()
        embodied_agent_node.destroy_node()
        rclpy.shutdown()