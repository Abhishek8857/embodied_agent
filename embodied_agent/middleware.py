from langchain.agents.middleware.types import AgentState, ModelRequest, wrap_model_call, wrap_tool_call
from langgraph.runtime import Runtime
from langchain_core.messages import ToolMessage
from .context import Context


def dynamic_system_prompt(request: ModelRequest, state: AgentState, runtime: Runtime[Context]) -> ModelRequest:
    user_role = runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        prompt = f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        prompt = f"{base_prompt} Explain concepts simply and avoid jargon."
    else:
        prompt = base_prompt

    request.system_prompt = prompt
    return request

@wrap_model_call
def dynamic_model_selection():
    pass



@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
                        content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )