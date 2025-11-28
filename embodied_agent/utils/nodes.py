from rclpy.node import Node

class AgentPublisher(Node):
    def __init__(self, topic: str, type):
        super().__init__("Agent_Publisher")
        self.topic = topic
        self.type = type
        self.publisher = self.create_publisher(self.type, self.topic, 100)
        self.get_logger().info("Agent Publisher initialised...")
        
    def publish_callback (self, data):
        msg = self.type()
        msg.data = data
        self.publisher.publish(msg)
        self.get_logger().info(f"Published data: {msg.data}")


