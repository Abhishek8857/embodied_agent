# import os
# from langchain.chat_models import init_chat_model

# try:
#     with open("openai_api_key.config", "r") as f:
#         os.environ["OPENAI_API_KEY"] = f.read().strip()
# except FileNotFoundError:
#     raise RuntimeError("Missing openai_api_key.config file")

# def get_llm():
#     return init_chat_model("openai:gpt-4.1")


from langchain_ollama import ChatOllama
from .tools import get_tools

temperature: float = 0.8
model: str = "llama3.1" 

def get_llm():
    return ChatOllama(model=model, temperature=temperature)
