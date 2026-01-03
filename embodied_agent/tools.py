import base64

from langchain_core.tools import tool
from .llm import get_llm
from langchain.messages import HumanMessage
from .utils.action_client import ExecuteMotionClient
from .utils.tf_pose import TfPoseLookup
from .utils.joint_state_cache import JointStateCache
from .utils.capture import capture_rgb_image, capture_raw_depth_image



def get_tools(node):
    motion = ExecuteMotionClient(action_name="/execute_motion", expected_joint_len=7)
    tf_lookup = TfPoseLookup(node)
    joint_state_lookup = JointStateCache(node, topic="/isaac_joint_states")

   
    @tool(name_or_callable="move_to_home_pose", description="Send a joint target. Expects 7 joint values. Returns success/failure + feedback trace.")
    def move_to_home_pose():
        data = [0.0, 0.049, -0.4882, 3.1227, -2.0745, 0.0112, -0.9870, 1.55]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_to_retract_pose", description="Send a joint target for retract pose. Expects 7 values. Returns success/failure * feedback trace.")
    def move_to_retract_pose():
        data = [0.0, 0.0, 0.0, 3.1227, -1.5, 0.0, -1.6, 1.55]
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
    def move_forward(distance: float | int, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x, y, z, qx, qy, qz, qw]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_backward", description="Moves the robot backward by a specified amount")
    def move_backward(distance: float | int, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x, y, z, qx, qy, qz, qw]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_left", description="Moves the robot left by a specified amount")
    def move_left(distance: float | int, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x, y, z, qx, qy, qz, qw]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_right", description="Moves the robot right by a specified amount")
    def move_right(distance: float | int, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x, y, z, qx, qy, qz, qw]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_upward", description="Moves the robot upward by a specified amount")
    def move_upward(distance: float | int, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x, y, z, qx, qy, qz, qw]
        return motion.send(data)
    
    
    @tool(name_or_callable="move_downwards", description="Moves the robot downward by a specified amount")
    def move_downward(distance: float | int, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float):
        data = [1.0, x, y, z, qx, qy, qz, qw]
        return motion.send(data)
    
    
    @tool(name_or_callable="get_current_pose", description="Get the current pose of the robot")
    def get_current_pose(base_frame: str = "base_link", ee_frame: str = "end_effector_link", timeout_s: float = 1.0):
        return tf_lookup.get_pose(base_frame, ee_frame, timeout_s)
    
    
    @tool(name_or_callable="get_current_joint_states", description="Get the current joint states of the robot")
    def get_current_joint_states(max_age_s: float = 1.0):
        return joint_state_lookup.get_latest(max_age_s)
    
    
    @tool(name_or_callable="capture_image", description="Captures the RGB image and saves locally and returns path")
    def capture_image ():
        return {"path": capture_rgb_image(node, save_dir="captures/rgb")}
   
    
    @tool(name_or_callable="capture_depth_image", description="Captures depth image, saves it locally and returns path")
    def capture_depth_image():
        return {"path": capture_raw_depth_image(node, save_dir="captures/depth")}
    
    
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
            get_current_joint_states,
            describe_what_you_see,
            capture_image,
            capture_depth_image]
