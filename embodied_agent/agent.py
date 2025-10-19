import os
from langchain.agents import create_agent
from .llm import get_llm
from .tools import get_tools
from .prompts import get_prompts
from .memory import get_memory
from .context import Context
from .middleware import dynamic_system_prompt
from .utils import get_langsmith_api_key

# Langsmith setup for Tracing
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]="embodied_agent"
os.environ["LANGMSMITH_API_KEY"]=get_langsmith_api_key()


embodied_agent = create_agent(model=get_llm(), 
                     tools=get_tools(), 
                     prompt=get_prompts(),
                     context_schema=Context,
                     middleware=[dynamic_system_prompt],
                     checkpointer=get_memory())

