from langchain.agents import create_agent
from .llm import get_llm
from .tools import get_tools
from .prompts import get_prompts
from .memory import get_memory
from .utils import Context

embodied_agent = create_agent(model=get_llm(), 
                     tools=get_tools(), 
                     prompt=get_prompts())

