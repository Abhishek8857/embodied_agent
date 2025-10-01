import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String 
from .agent import embodied_agent
from .utils import format_message

class Agent(Node):
    def __init__(self):
        super().__init__("Embodied_Agent")
        qos_profile: QoSProfile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(String, "query", self.query_callback, qos_profile=qos_profile)
        self.agent = embodied_agent
        
        self.get_logger().info("Agent node initialised!")
        
        self.message_recieved: bool = False
    
    
    
    def query_callback (self, msg):
        """Callback function to parse the recieved query to the Agent"""
        self.get_logger().info("Waiting of user query")
        
        # Wait for the User query
        if not self.message_recieved:
            self.message_recieved = True
        
        # Confirm message recieved from the user    
        self.get_logger().info(f"User Query: {msg.data}")
        
        # Invoke the Agent with user query and handle failure
        try:
            self.get_logger().info(f"Invoking Agent with User Query")
            self.agent.invoke(msg.data)
        except Exception as e:
            self.get_logger().info(f"Error executing user Query: {e}")
        finally:
            self.message_recieved = False
            self.get_logger().info("Waiting for the next message...\n")
        
def main ():
    rclpy.init()
    embodied_agent_node = Agent()
    try:
        rclpy.spin(embodied_agent_node)
    except KeyboardInterrupt:
        embodied_agent_node.get_logger().info("Keyboard interrupt received, shutting down...")
    finally:
        embodied_agent_node.destroy_node()
        rclpy.shutdown()
    