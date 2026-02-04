import base64

from langchain_core.tools import tool
from .llm import get_llm
from langchain.messages import HumanMessage
from .utils.action_client import ExecuteMotionClient
from .utils.tf_pose import TfPoseLookup
from .utils.joint_state_cache import JointStateCache
from .utils.capture import *
from .utils.grasp_pose_cache import GraspPoseCache
from .utils.gemini_segmentor import GeminiSegmentor



def get_tools(node):
    motion = ExecuteMotionClient(action_name="/execute_motion", expected_joint_len=7)
    tf_lookup = TfPoseLookup(node)
    joint_state_lookup = JointStateCache(node, topic="/isaac_joint_states")
    grasp_pose_lookup = GraspPoseCache(node, topic="/grasp_pose")
    segmentor = GeminiSegmentor()
    
    @tool(name_or_callable="move_to_home_pose", description="Send a joint target. Expects 7 joint values. Returns success/failure + feedback trace.")
    def move_to_home_pose():
        data = [0.0, 0.049, -0.4882, 3.1227, -2.0745, 0.0112, -0.9870, 1.55]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_to_retract_pose", description="Send a joint target for retract pose. Expects 7 values. Returns success/failure * feedback trace.")
    def move_to_retract_pose():
        data = [0.0, 0.0, -0.0, 3.1227, -1.5, 0.0, -1.6, 1.55]
        return motion.send(data)


    @tool(name_or_callable="move_to_pose", description="Send a pose target (x,y,z,qx,qy,qz,qw). Returns success/failure + feedback trace.")
    def move_to_pose(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
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


    @tool(name_or_callable="move_forward", description="Moves the robot forward by a specified amount")
    def move_forward(distance: float, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x or 0, y or 0, z or distance, qx or 0, qy or 0, qz or 0, qw or 1]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_backward", description="Moves the robot backward by a specified amount")
    def move_backward(distance: float, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x or 0, y or 0, z or distance, qx or 0, qy or 0, qz or 0, qw or 1]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_left", description="Moves the robot left by a specified amount")
    def move_left(distance: float, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x or 0, y or 0, z or distance, qx or 0, qy or 0, qz or 0, qw or 1]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_right", description="Moves the robot right by a specified amount")
    def move_right(distance: float, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x or 0, y or 0, z or distance, qx or 0, qy or 0, qz or 0, qw or 1]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_upward", description="Moves the robot upward by a specified amount")
    def move_upward(distance: float, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x or 0, y or 0, z or distance, qx or 0, qy or 0, qz or 0, qw or 1]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_downwards", description="Moves the robot downward by a specified amount")
    def move_downward(distance: float, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x or 0, y or 0, z or distance, qx or 0, qy or 0, qz or 0, qw or 1]
        return motion.send(data)
    
        
    @tool(name_or_callable="get_current_pose", description="Get the current pose of the robot between two frames.")
    def get_current_pose(base_frame: str, ee_frame: str, timeout_s: float):
        """Get the current pose from base_frame to ee_frame.
        
        Args:
            base_frame: Reference frame like 'base_link' 
            ee_frame: End-effector frame like 'end_effector_link' or 'tool0'
            timeout_s: Lookup timeout in seconds (default: 1.0)
        """
        base_frame = base_frame or "base_link"      
        ee_frame = ee_frame or "end_effector_link"    
        timeout_s = timeout_s or 1.0                
        
        return tf_lookup.get_pose(base_frame, ee_frame, timeout_s)
    
    @tool(name_or_callable="get_current_joint_states", description="Get the current joint states of the robot")
    def get_current_joint_states(max_age_s: float):
        max_age_s = float(max_age_s or 1.0)
        return joint_state_lookup.get_latest(max_age_s)
    
    
    @tool(name_or_callable="capture_only_rgb_image", description="Captures the RGB image and saves locally and returns path")
    def capture_only_rgb_image ():
        return {"path": capture_rgb_image(node, 
                                          topic="/front_stereo_camera/rgb/image_raw", 
                                          save_dir="captures/rgb")}
   
    
    @tool(name_or_callable="capture_only_depth_image", description="Captures depth image, saves it locally and returns path")
    def capture_only_depth_image():
        return {"path": capture_raw_depth_image(node, 
                                                topic="/front_stereo_camera/depth/image_rect_raw",    
                                                save_dir="captures/depth")}
    
    
    @tool(name_or_callable="get_latest_grasp_pose", description="Get the latest grasp pose published on /grasp_pose topic. Returns pose (x,y,z,qx,qy,qz,qw) or error.")
    def get_latest_grasp_pose(max_age_s: float = 20.0):
        return grasp_pose_lookup.get_latest(max_age_s=float(max_age_s or 5.0))
        
    
    @tool(name_or_callable="pick_up_object", description="Picks up object at a specified pose (x, y, z, qx, qy, qz, qw)")
    def pick_up_object(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float, pre_grasp_offset: float = 0.15, lift_height: float = 0.15):
        """        
        Execute complete pick/grasp sequence at specified pose with automatic pre-grasp planning.
        
        Args:
            x, y, z: Grasp position in meters
            qx, qy, qz, qw: Grasp orientation as quaternion
            pre_grasp_offset: Distance to offset before approaching (meters)
            lift_height: How high to lift after grasping (meters)
        
        Returns:
            Success/failure with execution trace
        """
        data = [3.0, x, y, z, qx, qy, qz, qw, pre_grasp_offset, lift_height]
        return motion.send(data)    
    
    
    @tool(name_or_callable="place_object", description="Place held object at specified pose (x,y,z,qx,qy,qz,qw). ")
    def place_object(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float,retreat_distance: float = 0.15):
        """
        Execute complete place sequence at specified pose.
        
        Args:
            x, y, z: Place position in meters
            qx, qy, qz, qw: Place orientation as quaternion
            retreat_distance: How far to retreat after placing (meters)
        
        Returns:
            Success/failure with execution trace
        """
        data = [4.0, x, y, z, qx, qy, qz, qw, retreat_distance]
        return motion.send(data)
    
    
    @tool(name_or_callable="capture_rgbd", description="Capture RGB + depth + K and save to captures/rgbd/rgbd_image.npz")
    def capture_rgbd():
        return capture_rgbd_npz(
            node,
            save_dir="captures/rgbd",
            filename="rgbd_image.npz",
            rgb_topic="/front_stereo_camera/rgb/image_raw",
            depth_topic="/front_stereo_camera/depth/image_rect_raw",
            camera_info_topic="/front_stereo_camera/rgb/camera_info",
            timeout_s=2.0,
        )

    @tool(name_or_callable="segment_objects", description="Segment objects in the scene by natural language query (e.g. 'blue objects', 'the red cup')." 
                                                          "Requires capture_rgbd to have been called first. Returns label, bounding box, and 3D grasp center for each detected object.")
    def segment_objects(query: str):
        """
        Args:
            query: What to segment, e.g. "blue objects", "the red cup"
        """
        return segmentor.segment("captures/rgbd/rgbd_image.npz", query)
    

    @tool(name_or_callable="describe_what_you_see", description="Takes an Image and parses it to the VLM for a description")
    def describe_what_you_see():
        """
        Capture an image and parse it to the language model to be described
        """
        image_path = "captures/rgb/rgb.jpg"
        model = get_llm()

        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the image in detail "},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )

        ai_msg = model.invoke([message])
        return ai_msg.content
    


    return [move_to_home_pose,
            move_to_retract_pose,
            move_to_pose, 
            move_forward,
            move_backward,
            move_left,
            move_right,
            move_upward,
            move_downward,
            close_the_gripper, 
            open_the_gripper,
            get_current_pose,
            segment_objects,
            get_current_joint_states,
            describe_what_you_see,
            capture_only_rgb_image,
            capture_only_depth_image,
            capture_rgbd, 
            get_latest_grasp_pose,
            pick_up_object,
            place_object]
