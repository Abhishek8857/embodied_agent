# nav2_tools.py
import rclpy
import math
import json
import os
from pathlib import Path
from langchain_core.tools import tool
from .nav2_action_client import Nav2ActionClient
from .utils import relative_move, quat_to_rpy

LOCATIONS_PATH = Path(__file__).parent.parent / "locations" / "saved_locations.json"

def _load_locations() -> dict:
    with open(LOCATIONS_PATH, "r") as f:
        return json.load(f)

def get_tools(node) -> list:
    nav2_client = Nav2ActionClient(namespace="kairosAB")

    @tool
    def navigate_to_pose(x: float, y: float, yaw_degrees: float = 0.0) -> dict:
        """
        Navigate the robot to an absolute (x, y) position on the map with a given heading.

        Args:
            x:           Target X coordinate in meters (map frame, +X = forward)
            y:           Target Y coordinate in meters (map frame, +Y = left)
            yaw_degrees: Target heading in degrees. 0° = facing +X (forward),
                         90° = facing +Y (left), -90° = facing -Y (right),
                         180° = facing -X (backward).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        return nav2_client.navigate_to_pose(x=x, y=y, yaw_degrees=yaw_degrees)

    @tool
    def navigate_to_location(location_name: str) -> dict:
        """
        Navigate the robot to a named saved location (e.g. "table A", "table B", "home").
        Use this when the user refers to a place by name rather than coordinates.

        Args:
            location_name: Name of the saved location (case-insensitive).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        locations = _load_locations()

        # Case-insensitive lookup
        key = next((k for k in locations if k.lower() == location_name.strip().lower()), None)
        if key is None:
            available = ", ".join(f'"{k}"' for k in locations)
            return {
                "success": False,
                "status":  "not_found",
                "message": f'Location "{location_name}" not found. Available: {available}',
            }

        entry = locations[key]
        pos   = entry["position"]
        ori   = entry["orientation"]

        x = pos["x"]
        y = pos["y"]

        # Convert full quaternion → yaw using existing util
        _, _, yaw_rad = quat_to_rpy(ori["x"], ori["y"], ori["z"], ori["w"])
        yaw_degrees   = math.degrees(yaw_rad)

        return nav2_client.navigate_to_pose(x=x, y=y, yaw_degrees=yaw_degrees)

    @tool
    def save_location(location_name: str) -> dict:
        """
        Save the robot's current pose as a named location in saved_locations.json.
        Use this when the user says "save this location", "remember this spot",
        "add this as <name>", or similar.

        Args:
            location_name: Name to save the location under (e.g. "table G", "charging dock").

        Returns:
            dict with success (bool), status (str), message (str)
        """
        pose = nav2_client.get_current_pose()
        if "x" not in pose:
            return pose

        # Convert yaw back to quaternion (planar: x=0, y=0)
        yaw_rad = math.radians(pose["yaw_degrees"])
        entry = {
            "position": {
                "x": pose["x"],
                "y": pose["y"],
                "z": 0.0,
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(yaw_rad / 2.0),
                "w": math.cos(yaw_rad / 2.0),
            },
        }

        locations = _load_locations()
        already_exists = location_name.strip().lower() in [k.lower() for k in locations]
        locations[location_name.strip()] = entry

        with open(LOCATIONS_PATH, "w") as f:
            json.dump(locations, f, indent=2)

        action = "updated" if already_exists else "saved"
        return {
            "success": True,
            "status":  action,
            "message": f'Location "{location_name}" {action} at x={pose["x"]:.4f}, y={pose["y"]:.4f}, yaw={pose["yaw_degrees"]:.2f}°',
        }
        
    @tool
    def delete_location(location_name: str) -> dict:
        """
        Delete a named location from saved_locations.json.
        Use this when the user says "delete location", "remove <name>", "forget <name>".

        Args:
            location_name: Name of the location to delete (case-insensitive).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        locations = _load_locations()

        key = next((k for k in locations if k.lower() == location_name.strip().lower()), None)
        if key is None:
            available = ", ".join(f'"{k}"' for k in locations)
            return {
                "success": False,
                "status":  "not_found",
                "message": f'Location "{location_name}" not found. Available: {available}',
            }

        del locations[key]

        with open(LOCATIONS_PATH, "w") as f:
            json.dump(locations, f, indent=2)

        return {
            "success": True,
            "status":  "deleted",
            "message": f'Location "{key}" has been deleted.',
        }
        
    @tool
    def list_locations() -> dict:
        """
        Return all saved location names the robot can navigate to.
        Use this when the user asks "where can you go?" or "what locations do you know?".

        Returns:
            dict with success (bool), locations (list of str)
        """
        locations = _load_locations()
        return {"success": True, "locations": list(locations.keys())}

    @tool
    def get_current_pose() -> dict:
        """
        Get the robot's current pose on the map (x, y, yaw).
        Use this before relative moves or to verify the robot's position.

        Returns:
            dict with x (float), y (float), yaw_degrees (float)
        """
        return nav2_client.get_current_pose()

    @tool
    def move_forward(distance: float = 1.0) -> dict:
        """
        Move the robot forward by a given distance while keeping its current heading.

        Args:
            distance: Distance to move in metres (must be positive).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        pose = nav2_client.get_current_pose()
        if "x" not in pose:
            return pose
        target = relative_move(pose, distance_x=distance, distance_y=0.0)
        return nav2_client.navigate_to_pose(**target)

    @tool
    def move_backward(distance: float = 1.0) -> dict:
        """
        Move the robot backward by a given distance while keeping its current heading.

        Args:
            distance: Distance to move in metres (must be positive).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        pose = nav2_client.get_current_pose()
        if "x" not in pose:
            return pose
        target = relative_move(pose, distance_x=-distance, distance_y=0.0)
        return nav2_client.navigate_to_pose(**target)

    @tool
    def move_left(distance: float = 1.0) -> dict:
        """
        Strafe the robot to the left by a given distance while keeping its current heading.

        Args:
            distance: Distance to move in metres (must be positive).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        pose = nav2_client.get_current_pose()
        if "x" not in pose:
            return pose
        target = relative_move(pose, distance_x=0.0, distance_y=-distance)
        return nav2_client.navigate_to_pose(**target)

    @tool
    def move_right(distance: float = 1.0) -> dict:
        """
        Strafe the robot to the right by a given distance while keeping its current heading.

        Args:
            distance: Distance to move in metres (must be positive).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        pose = nav2_client.get_current_pose()
        if "x" not in pose:
            return pose
        target = relative_move(pose, distance_x=0.0, distance_y=distance)
        return nav2_client.navigate_to_pose(**target)

    @tool
    def turn_left(degrees: float = 90.0) -> dict:
        """
        Rotate the robot counter-clockwise (left) by a given angle in place.

        Args:
            degrees: Angle to rotate in degrees (must be positive).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        pose = nav2_client.get_current_pose()
        if "yaw_degrees" not in pose:
            return pose
        new_yaw = (pose["yaw_degrees"] + degrees) % 360
        return nav2_client.navigate_to_pose(x=pose["x"], y=pose["y"], yaw_degrees=new_yaw)

    @tool
    def turn_right(degrees: float = 90.0) -> dict:
        """
        Rotate the robot clockwise (right) by a given angle in place.

        Args:
            degrees: Angle to rotate in degrees (must be positive).

        Returns:
            dict with success (bool), status (str), message (str)
        """
        pose = nav2_client.get_current_pose()
        if "yaw_degrees" not in pose:
            return pose
        new_yaw = (pose["yaw_degrees"] - degrees) % 360
        return nav2_client.navigate_to_pose(x=pose["x"], y=pose["y"], yaw_degrees=new_yaw)

    return [
        navigate_to_pose,
        navigate_to_location,
        list_locations,
        save_location,
        delete_location,
        get_current_pose,
        move_forward,
        move_backward,
        move_left,
        move_right,
        turn_left,
        turn_right,
    ]