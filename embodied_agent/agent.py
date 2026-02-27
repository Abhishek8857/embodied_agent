import os
from langchain.agents import create_agent
from pathlib import Path
from dotenv import load_dotenv
from .llm import get_llm
from .prompts import get_prompts
from .memory import get_memory, get_memory_storage
from .context import Context
from .middleware import dynamic_system_prompt, handle_tool_errors, dynamic_model_selection
from .utils.utils import get_langsmith_api_key


def build_embodied_agent(tools):

    load_dotenv(dotenv_path=".env")
    os.environ["LANGSMITH_API_KEY"] = get_langsmith_api_key()


    embodied_agent = create_agent(model=get_llm(), 
                        tools=tools, 
                        system_prompt=get_prompts(),
                        middleware=[handle_tool_errors],
                        checkpointer=get_memory(),
                        store=get_memory_storage())

    return embodied_agent

