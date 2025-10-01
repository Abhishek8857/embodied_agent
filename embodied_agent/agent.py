import os
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from .llm import get_model
from .post_model_hook import get_post_model_hook
from .pre_model_hook import get_pre_model_hook
from .response_format import get_response_format
from .tools import get_tools
from .prompts import get_prompts

embodied_agent = create_react_agent(model=get_model(),
                           tools=get_tools())


png_data = embodied_agent.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)