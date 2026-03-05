import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
from .utils.utils import get_openai_api_key, get_mistral_api_key


temperature: float = 0.8
model: str = "gpt-4.1"
qwen_model = "qwen/qwen3-vl-30b-a3b-thinking"
base_url: str =  "https://openrouter.ai/api/v1"


def get_llm():
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model=model,
        temperature=temperature,
        base_url=base_url,
        default_headers={
            "Authorization": f"Bearer {get_openai_api_key()}"
        }
    )
    

def get_qwen_llm():
    return ChatOpenAI(
        api_key=get_openai_api_key(),
        model=qwen_model,
        temperature=temperature,
        base_url=base_url,
        default_headers={
            "Authorization": f"Bearer {get_openai_api_key()}"
        }
    )