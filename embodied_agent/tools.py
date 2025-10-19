from langchain_core.tools import tool
from langchain.tools import ToolNode
from langsmith import traceable
from langgraph.runtime import get_runtime
from langgraph.store.memory import InMemoryStore
from .context import Context


# @tool
# @traceable
# def save_user_info(user_info: UserInfo):
#     """Saves the user information

#     Args:
#         user_info (UserInfo): name

#     Returns:
#         str : Confirmation of succesful execution
#     """
#     runtime = get_runtime(Context)
#     store = runtime.store 
#     user_id = runtime.context.user_id 
#     # Store data in the store (namespace, key, data)
#     store.put(("users",), user_id, user_info) 
    
#     return "Successfully saved user info."


@tool(name_or_callable="multiply",description="Tool used to multiple numbers")
@traceable
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
@traceable
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


def get_tools() -> ToolNode:
    tool_node = ToolNode(
        tools=[multiply,
               divide,
               add,
               ],
        handle_tool_errors="Please check your input and try again"
    )
    
    return tool_node
    

