import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
from .utils.utils import get_openai_api_key, get_mistral_api_key


temperature: float = 0.1
agent_model: str = "gpt-4.1"
distiller_model = "qwen/qwen3-32b" # "qwen/qwen3-32b" "qwen/qwen3-8b" "qwen/qwen3-vl-30b-a3b-thinking"
base_url: str =  "https://openrouter.ai/api/v1"


def get_llm():
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model=agent_model,
        temperature=temperature,
        base_url=base_url,
        default_headers={
            "Authorization": f"Bearer {get_openai_api_key()}"
        }
    )
    

def get_qwen_llm():
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model=distiller_model,
        temperature=temperature,
        base_url=base_url,
        default_headers={
            "Authorization": f"Bearer {get_openai_api_key()}"
        }
    )