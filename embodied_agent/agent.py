import os
from langchain.agents import create_agent
from pathlib import Path
from dotenv import load_dotenv
from .llm import get_llm
from .tools import get_tools
from .prompts import get_prompts
from .memory import get_memory
from .context import Context
from .middleware import dynamic_system_prompt, handle_tool_errors, dynamic_model_selection
from .utils.utils import get_langsmith_api_key


load_dotenv(dotenv_path=".env")
os.environ["LANGSMITH_API_KEY"] = get_langsmith_api_key()


embodied_agent = create_agent(model=get_llm(), 
                     tools=get_tools(), 
                     system_prompt=get_prompts(),
                     context_schema=Context,
                     middleware=[handle_tool_errors],
                     checkpointer=get_memory())

