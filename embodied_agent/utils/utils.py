import os
import math
import rclpy
import tf2_ros
import threading
import time

from typing import Dict, Optional, Any
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from .nodes import AgentPublisher
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str
            
        
def format_message (msg: str) -> dict:
    return {"messages": [{"role" : "user" , "content": msg}]}


def format_response(msg: dict) -> dict:
    """
    Function to format and print the Agents response to a Readable format in terminal

    Args:
        msg (dict): The output of the Agent in a dictionary format.

    Returns:
        dict: Formatted response in a dictionary format with human_messages, ai_messages, final_response as keys.
    """
    data = msg["messages"]
    human_messages, ai_messages, tool_calls, final_response = [], [], [], []

    for obj in data:
        if isinstance(obj, HumanMessage):
            human_messages.append({"content": obj.content})

        elif isinstance(obj, AIMessage):
            # final text
            if obj.content and obj.content.strip():
                final_response.append({"content": obj.content})
            # tool calls (can be multiple)
            if getattr(obj, "tool_calls", None):
                ai_messages.extend(obj.tool_calls)

        elif isinstance(obj, ToolMessage):
            tool_calls.append({
                "tool": obj.name,
                "output": obj.content,
                "tool_call_id": obj.tool_call_id
            })

    return {
        "human_messages": human_messages,
        "tool_calls": tool_calls,
        "ai_messages": ai_messages,
        "final_response": final_response
    }

def print_response(data: dict):
    """    
    Print only the latest full exchange (Human → AI → Tool(s) → Final AI Response).
    Handles multiple tool calls made during one request.
    Clears the screen before printing for a live-display effect.

    Args:
        data (dict): _description_
    """
    human_msgs = data.get("human_messages", [])
    ai_msgs = data.get("ai_messages", [])
    tool_msgs = data.get("tool_calls", [])
    final_responses = data.get("final_response", [])

    # Clear terminal for live dashboard style output
    os.system("clear")

    width = 80
    def print_centered_header(title: str):
        print(f" {title} ".center(width, "="))

    # Get the latest human message
    h_msg = human_msgs[-1] if human_msgs else None

    # Determine which AI and tool messages belong to this exchange
    last_human_index = len(human_msgs) - 1
    ai_after_human = ai_msgs[last_human_index:] if ai_msgs else []
    final = final_responses[-1] if final_responses else None

    # Create a lookup map for tool call outputs
    tool_map = {t['tool_call_id']: t for t in tool_msgs}

    # === Print Human Message ===
    if h_msg:
        print_centered_header("Human Message")
        print(h_msg["content"])
        print()

    # === Print All AI Messages & Tools ===
    for ai_msg in ai_after_human:
        print_centered_header("AI Message")

        if "name" in ai_msg and "args" in ai_msg:
            print("Tool Calls:")
            print(f"  {ai_msg['name']} (Call ID: {ai_msg['id']})")
            print("  Args:")
            for k, v in ai_msg["args"].items():
                print(f"    {k}: {v}")

            # Find and print corresponding tool message(s)
            t_msg = tool_map.get(ai_msg['id'])
            if t_msg:
                print()
                print_centered_header("Tool Message")
                print(f"Name: {t_msg['tool']}")
                print(f"Output: {t_msg['output']}")
                print(f"Tool Call ID: {t_msg['tool_call_id']}")
        print()

    # === Print Final Response ===
    if final:
        print_centered_header("Final AI Response")
        print(final["content"])
        print()


def quat_to_rpy(x: float, y: float, z: float, w: float):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
    

def publish_to(type_name, topic_name: str, coordinates: list = None, msg: bool = None) -> None:
    """
    Publishes Coordinates to MoveIt

    Args:
        coordinates (list) = None :  Desired coordinates to publish
        msg (bool) = None : Desired Bool Message to publish
    """
    try:
        # Initialise the ROS2 Node to publish the coordinates
        publisher_node = AgentPublisher(type=type_name, topic=topic_name)
        if type_name == Float64MultiArray:
            publisher_node.publish_callback(coordinates)
        elif type_name == Bool:
            publisher_node.publish_callback(msg)
        
        # Spin the Node to keep it publishing
        rclpy.spin_once(publisher_node, timeout_sec=0.1)

        # Destroy Node
        publisher_node.destroy_node()
    except Exception as e:
        print(f"Error Publishing Coordinates to Robot: {e}")
        return
    finally:
        publisher_node.destroy_node()
    

def get_openai_api_key() -> str: 
    """
    Returns the OpenAI API key to connect with the Open Router server

    Raises:
        RuntimeError: FileNotFoundError (the api config file is missing) 

    Returns:
        str: API Key
    """
    try:
        with open("openai_api_key.config", "r") as f:
            api_key = os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("Missing openai_api_key.config file")
    
    return api_key


def get_langsmith_api_key() -> str:
    """
    Returns the Langsmith API key to connect with the langsmith server to monitor agent statistics

    Raises:
        RuntimeError: FileNotFoundError (the api config file is missing) 

    Returns:
        str: API Key
    """
    try:
        with open("langsmith_api_key.config", "r") as f:
            api_key = os.environ["LANGSMITH_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("Missing langsmith_api_key.config file")
    
    return api_key

def get_gemini_api_key() -> str:
    """
    Returns Gemini API Key required to invoke gemini for image segmentation

    Raises:
        RuntimeError: FileNotFoundError (api config file is missing)
        
    Returns:
        str: API Key
    """
    
    try:
        with open("gemini_api_key.config", "r") as f:
            api_key = os.environ["GEMINI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("Missing geminie_api_key.config file")
    
    return api_key