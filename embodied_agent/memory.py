from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store


def get_memory() -> InMemorySaver:
    return InMemorySaver()

