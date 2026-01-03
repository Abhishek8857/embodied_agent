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
from .config import get_config
from .context import Context

class Agent(Node):
    def __init__(self):
        super().__init__("embodied_agent")
        qos_profile: QoSProfile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(String, "query", self.query_callback, qos_profile=qos_profile)
        tools = get_tools(self)
        
        self.agent = build_embodied_agent(tools=tools)
        self.get_logger().info("Agent node initialised!")
        self.message_recieved: bool = False
    
    
    def query_callback (self, msg):
        """Callback function to parse the recieved query to the Agent"""
        self.get_logger().info("Waiting of user query")
        
        # Wait for the User query
        if not self.message_recieved:
            self.message_recieved = True
        
        # Confirm message recieved from the user    
        self.get_logger().info(f"User Query: '{msg.data}'")
                
        threading.Thread(target=self.handle_query, 
                            args=(msg.data,), 
                            daemon=True).start()


    def handle_query(self, user_query: str):
        try:
            self.get_logger().info(f"Invoking Agent with User Query: {user_query}")

            response = self.agent.invoke(
                format_message(user_query),
                config=get_config(),
                context={"user_role": "beginner"},
            )

            print_response(format_response(response))

        except Exception as e:
            self.get_logger().error(f"Error executing User Query: {e}")

        finally:
            self.message_recieved = False
            self.get_logger().info("Waiting for the next message...\n")
        
        
def main ():
    rclpy.init()
    embodied_agent_node = Agent()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node=embodied_agent_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        embodied_agent_node.get_logger().info("Keyboard interrupt recieved, shutting down...")
    finally:
        executor.shutdown()
        embodied_agent_node.destroy_node()
        rclpy.shutdown()
    