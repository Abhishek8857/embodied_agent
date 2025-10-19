from langchain.agents.middleware.types import modify_model_request, AgentState, ModelRequest
from langgraph.runtime import Runtime
from .context import Context


@modify_model_request
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

