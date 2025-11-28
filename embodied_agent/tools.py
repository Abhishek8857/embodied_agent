import base64
from langchain.messages import HumanMessage
from std_msgs.msg import Float64MultiArray
from langchain_core.tools import tool
from .llm import get_llm
from .utils.utils import publish_to

COORDINATE_TYPE = Float64MultiArray
COORDINATE_TOPIC = "published_coordinates"

@tool(name_or_callable="multiply", description="Tool used to multiply numbers")
def multiply(a: float | int, b: float | int) -> float | int:
    """Multiply a and b.

    Args:
        a: first number
        b: second number
    """
    return a * b

@tool(name_or_callable="add", description="Tool used to add numbers")
def add(a: float | int, b: float | int) -> float | int:
    """Adds a and b.

    Args:
        a: first number
        b: second number
    """
    return a + b


@tool(name_or_callable="divide", description="Tool used to divide numbers")
def divide(a: float | int, b: float | int) -> float | int:
    """Divide a and b.

    Args:
        a: first number
        b: second number
    """
    return a / b


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
    
        
@tool(name_or_callable="go_to_home pose", description="Move the robot to a predefined position as a default pose")
def go_to_home_pose():
    """
    Move the Arm to a predefined home position
    """
    home_pose_coordinates = [0.0, 0.0, -0.7650, -3.15, -2.13, 0.006, -1.2, 1.55]
    # home_pose_coordinates = [1.0, 0.28, -0.2, 0.5, 0.0, 1.0, 0.0, 0.0]

    publish_to(type_name=COORDINATE_TYPE, topic_name=COORDINATE_TOPIC, coordinates=home_pose_coordinates)


@tool(name_or_callable="open_gripper", description="opens the gripper")
def open_gripper():
    """
    Opens the Gripper
    """
    open_coordinates = [2.0, -0.0, 0.0]
    publish_to(type_name=COORDINATE_TYPE, topic_name=COORDINATE_TOPIC, coordinates=open_coordinates)


@tool(name_or_callable="close_gripper", description="closes the gripper")
def close_gripper():
    """
    Closes the Gripper
    """
    close_coordinates = [2.0, -0.8, 0.8]
    publish_to(type_name=COORDINATE_TYPE, topic_name=COORDINATE_TOPIC, coordinates=close_coordinates)
    

def get_tools():
    tools=[             
        describe_what_you_see,
        go_to_home_pose,
        open_gripper,
        close_gripper,
        add,
        multiply,
        divide
        ]
    
    return tools
    

