import os
from langchain.agents import create_agent
from pathlib import Path
from dotenv import load_dotenv
from .llm import get_llm, get_qwen_llm
from .utils.nav2_prompts import get_nav_prompt
from .memory import get_memory, get_memory_storage
from .context import Context
from .middleware import dynamic_system_prompt, handle_tool_errors, dynamic_model_selection
from .utils.utils import get_langsmith_api_key
from .utils.memory_context import build_memory_context
from .response_format import ResponseFormat

def build_embodied_agent(tools, memory_summary_path: str = "memory/memory.json"):

    load_dotenv(dotenv_path=".env")
    os.environ["LANGSMITH_API_KEY"] = get_langsmith_api_key()

    store = get_memory_storage()

    # Build Qwen-distilled memory context for the system prompt 
    memory_context = build_memory_context(
        summary_path=memory_summary_path,
        llm=get_qwen_llm(),
        fallback_to_manual=True,
    )

    system_prompt = get_nav_prompt() + memory_context

    embodied_agent = create_agent(
        model=get_llm(),
        tools=tools,
        system_prompt=system_prompt,
        middleware=[handle_tool_errors],
        checkpointer=get_memory(),
        store=store,
        response_format=ResponseFormat
    )

    return embodied_agent