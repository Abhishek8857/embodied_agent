    
import base64
import httpx
from langchain.messages import HumanMessage
from openai import OpenAI
from langchain_core.tools import tool
from langsmith import traceable
from langgraph.runtime import get_runtime
from langgraph.store.memory import InMemoryStore
from langchain.messages import HumanMessage
from .context import Context
from .llm import get_llm

@tool(name_or_callable="multiply",description="Tool used to multiple numbers")
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


@tool
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
    
        


def get_tools():
    tools=[multiply,
            divide,
            add,
            describe_what_you_see]
    
    return tools
    

