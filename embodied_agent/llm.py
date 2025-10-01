from langchain_ollama import ChatOllama
from .tools import get_tools

language_model : str = "llama3.2:3b"
model_temperature : float = 0.8

def get_model ():
    return ChatOllama(model=language_model, temperature=model_temperature)  