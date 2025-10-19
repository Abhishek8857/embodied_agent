import os
from langchain_openai import ChatOpenAI
from .utils import get_openai_api_key


temperature: float = 0.8
model: str = "gpt-4.1"
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

