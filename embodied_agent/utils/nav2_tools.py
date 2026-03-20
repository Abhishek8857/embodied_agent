# nav_tools.py
import rclpy
from langchain_core.tools import tool
from .nav2_action_client import Nav2ActionClient

def get_tools(node) -> list:
    nav2_client = Nav2ActionClient(namespace="kairosAB")

    @tool
    def navigate_to_home() -> dict:
        """
        Navigate the robot to the home/origin position (x=0, y=0, yaw=0).
        Use this when the user asks to go home, return to start, or reset position.
        """
        return nav2_client.navigate_to_pose(x=0.0, y=0.0, yaw_degrees=-0.0)

    @tool
    def navigate_to_pose(x: float, y: float, yaw_degrees: float = 0.0) -> dict:
        """
        Navigate the robot to an absolute (x, y) position on the map with a given heading.

        Args:
            x:           Target X coordinate in meters (map frame, +X = left)
            y:           Target Y coordinate in meters (map frame, +Y = forward)
            yaw_degrees: Target heading in degrees. 0 = facing +X, 90 = facing -Y (backward),
                         180 = facing -X, -90 = facing +Y (forward)

        Returns:
            dict with success (bool), status (str), message (str)
        """
        return nav2_client.navigate_to_pose(x=x, y=y, yaw_degrees=yaw_degrees)

    @tool
    def get_current_pose() -> dict:
        """
        Get the robot's current pose on the map (x, y, yaw).
        Use this before relative moves or to verify the robot's position.

        Returns:
            dict with x (float), y (float), yaw_degrees (float)
        """
        return nav2_client.get_current_pose()

    return [
        navigate_to_home,
        navigate_to_pose,
        get_current_pose,
    ]