import base64

from langchain_core.tools import tool
from .llm import get_llm
from langchain.messages import HumanMessage
from .utils.action_client import ExecuteMotionClient
from .utils.tf_pose_lookup import TfPoseLookup
from .utils.joint_state_cache import JointStateCache
from .utils.capture import *
from .utils.grasp_pose_cache import GraspPoseCache
from .utils.gemini_segmentor import GeminiSegmentor
from .utils.calculate_place_pose import get_placement_pose
from .utils.save_graspnet_image import save_graspnet_image
from .utils.register_poses import RegisterPoses


def get_tools(node):
    motion = ExecuteMotionClient(action_name="/execute_motion", expected_joint_len=7)
    tf_lookup = TfPoseLookup(node)
    joint_state_lookup = JointStateCache(node, topic="/joint_states")
    grasp_pose_lookup = GraspPoseCache(node, topic="/grasp_pose")
    segmentor = GeminiSegmentor()
    pose_registry = RegisterPoses(path="poses/poses.json")

    # Shared state to pass data between tools
    _state = {
        "last_segmentation": None,
    }
    
    
    

    # ── Pose registry tools ────────────────────────────────────────────────────

    @tool(name_or_callable="save_current_pose",
      description="Save a named pose from joint states returned by get_current_joint_states. "
                  "Always call get_current_joint_states first, then pass the result here.")
    def save_current_pose(name: str, positions: list[float], names: list[str], description: str = ""):
        """
        Args:
            name:        Short name for the pose (e.g. 'home', 'above_table')
            positions:   The 'position' list from get_current_joint_states output
            names:       The 'name' list from get_current_joint_states output
            description: Optional note about what this pose is for
        """
        # Filter to arm joints only, exclude gripper finger joints
        arm_joints = [
            pos for joint_name, pos in zip(names, positions)
            if joint_name.startswith("joint_")
        ]

        if len(arm_joints) != 7:
            return {"success": False, "error": f"Expected 7 arm joints, got {len(arm_joints)}. Check names list."}

        return pose_registry.save_pose(name, arm_joints, description)
        
    @tool(name_or_callable="move_to_named_pose",
          description="Move the robot to a previously saved named pose. "
                      "Use list_saved_poses first to see what poses are available.")
    def move_to_named_pose(name: str):
        """
        Args:
            name: Name of the saved pose (e.g. 'home', 'retract', 'above_table')
        """
        joints = pose_registry.get_joints(name)
        if joints is None:
            available = list(pose_registry.list_poses().keys())
            return {
                "success": False,
                "error": f"Pose '{name}' not found.",
                "available_poses": available,
            }
        data = [0.0] + joints
        return motion.send(data)

    @tool(name_or_callable="list_saved_poses",
          description="List all saved named poses with their descriptions.")
    def list_saved_poses():
        poses = pose_registry.list_poses()
        if not poses:
            return {"success": True, "poses": {}, "message": "No poses saved yet. Use save_current_pose to save one."}
        return {"success": True, "poses": poses}

    @tool(name_or_callable="delete_saved_pose",
          description="Delete a named pose from the registry.")
    def delete_saved_pose(name: str):
        """
        Args:
            name: Name of the pose to delete
        """
        return pose_registry.delete_pose(name)

    @tool(name_or_callable="rename_saved_pose",
          description="Rename an existing saved pose.")
    def rename_saved_pose(old_name: str, new_name: str):
        """
        Args:
            old_name: Current name of the pose
            new_name: New name for the pose
        """
        return pose_registry.rename_pose(old_name, new_name)


    @tool(name_or_callable="move_to_pose",
          description="Send a pose target (x,y,z,qx,qy,qz,qw). Returns success/failure + feedback trace.")
    def move_to_pose(x: float, y: float, z: float,
                     qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x, y, z, qx, qy, qz, qw]
        return motion.send(data)

    @tool(name_or_callable="close_the_gripper", description="Closes the Gripper")
    def close_the_gripper():
        data = [2.0, 0.8]
        return motion.send(data)

    @tool(name_or_callable="open_the_gripper", description="Opens the Gripper")
    def open_the_gripper():
        data = [2.0, -0.0]
        return motion.send(data)


    @tool(name_or_callable="get_current_pose",
          description="Get the current pose of the robot between two frames.")
    def get_current_pose(base_frame: str, ee_frame: str, timeout_s: float):
        """
        Args:
            base_frame: Reference frame like 'base_link'
            ee_frame:   End-effector frame like 'end_effector_link'
            timeout_s:  Lookup timeout in seconds
        """
        base_frame = base_frame or "base_link"
        ee_frame   = ee_frame   or "end_effector_link"
        timeout_s  = timeout_s  or 3.0
        return tf_lookup.get_pose(base_frame, ee_frame, timeout_s)

    @tool(name_or_callable="get_current_joint_states",
          description="Get the current joint states of the robot.")
    def get_current_joint_states(max_age_s: float):
        max_age_s = float(max_age_s or 1.0)
        return joint_state_lookup.get_latest(max_age_s)

    @tool(name_or_callable="capture_only_rgb_image",
          description="Captures the RGB image, saves locally and returns path.")
    def capture_only_rgb_image():
        return {"path": capture_rgb_image(node,
                                          topic="/camera/color/image_raw",
                                          save_dir="captures/rgb")}

    @tool(name_or_callable="capture_only_depth_image",
          description="Captures depth image, saves it locally and returns path.")
    def capture_only_depth_image():
        return {"path": capture_raw_depth_image(node,
                                                topic="/camera/depth_registered/image_rect",
                                                save_dir="captures/depth")}

    @tool(name_or_callable="capture_rgbd",
          description="Capture RGB + depth + K and save to captures/rgbd/rgbd_image.npz.")
    def capture_rgbd():
        return capture_rgbd_npz(
            node,
            save_dir="captures/rgbd",
            filename="rgbd_image.npz",
            rgb_topic="/camera/color/image_raw",
            depth_topic="/camera/depth_registered/image_rect",
            camera_info_topic="/camera/depth_registered/camera_info",
            timeout_s=2.0,
        )

    @tool(name_or_callable="get_latest_grasp_pose",
          description="Get the latest grasp pose published on /grasp_pose topic. Returns pose (x,y,z,qx,qy,qz,qw) or error.")
    def get_latest_grasp_pose(max_age_s: float = 20.0):
        return grasp_pose_lookup.get_latest(max_age_s=float(max_age_s or 5.0))


    @tool(name_or_callable="pick_up_object",
          description="Picks up object at a specified pose (x, y, z, qx, qy, qz, qw).")
    def pick_up_object(x: float, y: float, z: float,
                       qx: float, qy: float, qz: float, qw: float,
                       pre_grasp_offset: float = 0.15,
                       lift_height: float = 0.15):
        """
        Args:
            x, y, z:           Grasp position in metres
            qx, qy, qz, qw:    Grasp orientation as quaternion
            pre_grasp_offset:  Distance to offset before approaching (metres)
            lift_height:       How high to lift after grasping (metres)
        """
        data = [3.0, x, y, z, qx, qy, qz, qw, pre_grasp_offset, lift_height]
        return motion.send(data)

    @tool(name_or_callable="place_object",
          description="Place held object at specified pose (x,y,z,qx,qy,qz,qw).")
    def place_object(x: float, y: float, z: float,
                     qx: float, qy: float, qz: float, qw: float,
                     retreat_distance: float = 0.15):
        """
        Args:
            x, y, z:           Place position in metres
            qx, qy, qz, qw:    Place orientation as quaternion
            retreat_distance:  How far to retreat after placing (metres)
        """
        data = [4.0, x, y, z, qx, qy, qz, qw, retreat_distance]
        return motion.send(data)

    @tool(
        name_or_callable="save_segmentation_for_graspnet",
        description="Save segmentation results in Contact-GraspNet format. "
                    "Call this after segment_objects, then run Contact-GraspNet externally."
    )
    def save_for_graspnet():
        if _state["last_segmentation"] is None:
            return {
                "success": False,
                "error": "No segmentation results found, call segment_objects() first."
            }

        result = save_graspnet_image(
            _state["last_segmentation"],
            rgbd_path="captures/rgbd/rgbd_image.npz",
            output_path="/ros-ai-agent/captures/segmentation/rgbd_sgmtd/rgbd_sgmtd.npz",
        )

        if result.get("success", False):
            _state["last_segmentation"] = None

        return result

    @tool(name_or_callable="get_place_pose",
          description="Get placement pose on top of a segmented object. "
                      "Use after segment_objects. Returns x,y,z,qx,qy,qz,qw ready for place_object.")
    def get_place_pose(timeout_s: float,
                       target_object_label: str = None,
                       height_offset: float = 0.23):
        """
        Args:
            target_object_label: Which object to place on (e.g. 'red cube'). If None, uses first.
            height_offset:       Clearance above surface in metres
        """
        if _state["last_segmentation"] is None:
            return {"success": False, "error": "No segmentation results found. Call segment_objects first."}

        tf_transform = tf_lookup.get_pose("base_link", "camera_depth_frame", timeout_s or 1.0)
        return get_placement_pose(
            _state["last_segmentation"],
            target_object_label=target_object_label,
            height_offset=0.23,
            tf_transform=tf_transform,
            apply_tf=True,
        )

    @tool(name_or_callable="segment_objects",
          description="Segment objects in the scene by natural language query "
                      "(e.g. 'blue objects', 'the red cup'). "
                      "Requires capture_rgbd to have been called first.")
    def segment_objects(query: str):
        """
        Args:
            query: What to segment, e.g. 'blue objects', 'the red cup'
        """
        result = segmentor.segment("captures/rgbd/rgbd_image.npz", query)
        _state["last_segmentation"] = result
        if _state["last_segmentation"] is None:
            return {"success": False, "error": "Segmentation failed. Results not cached."}

        return result

    @tool(name_or_callable="describe_environment",
          description="Use after capture_only_rgb_image() — passes the saved image to the VLM for a description.")
    def describe_environment(query: str):
        """
        Args:
            query: What specific aspect of the image to describe
        """
        image_path = "captures/rgb/rgb.jpg"
        model = get_llm()

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ]
        )
        return model.invoke([message]).content


    return [
        # Pose registry
        save_current_pose,
        move_to_named_pose,
        list_saved_poses,
        delete_saved_pose,
        rename_saved_pose,
        # Motion
        move_to_pose,
        close_the_gripper,
        open_the_gripper,
        # Sensing
        get_current_pose,
        get_current_joint_states,
        capture_only_rgb_image,
        capture_only_depth_image,
        capture_rgbd,
        get_latest_grasp_pose,
        # Manipulation
        pick_up_object,
        place_object,
        save_for_graspnet,
        get_place_pose,
        segment_objects,
        describe_environment,
    ]