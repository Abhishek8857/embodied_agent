import base64

from langchain_core.tools import tool
from .llm import get_llm
from langchain.messages import HumanMessage
from .utils.action_client import ExecuteMotionClient
from .utils.utils import TfPoseLookup



def get_tools(node):
    motion = ExecuteMotionClient(action_name="/execute_motion", expected_joint_len=7)
    tf_lookup = TfPoseLookup(node)

   
    @tool(name_or_callable="move_to_home_pose", description="Send a joint target. Expects 6 joint values. Returns success/failure + feedback trace.")
    def move_to_home_pose():
        data = [0.0, 0.0, -0.7650, -3.15, -2.13, 0.006, -1.2, 1.55]
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

    
    @tool(name_or_callable="get_current_pose", description="Get the current pose of the robot")
    def get_current_pose(base_frame: str = "base_link", ee_frame: str = "end_effector_link", timeout_s: float = 1.0):
        return tf_lookup.get_pose(base_frame, ee_frame, timeout_s)
    
    
    @tool(name_or_callable="describe_what_you_see", description="Takes an Image and parses it to the VLM for a description")
    def describe_what_you_see():
        """
        Capture an image and parse it to the language model to be described
        """
        image_path = "test_image.jpg"
        model = get_llm()

        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            
        message = HumanMessage(
            content=[
                {"type": "text", "text": "describe the "},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )

        ai_msg = model.invoke([message])
        return ai_msg.content
    


    return [move_to_home_pose,
            move_to_pose, 
            move_forward,
            close_the_gripper, 
            open_the_gripper,
            get_current_pose,
            describe_what_you_see]
