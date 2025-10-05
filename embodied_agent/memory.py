from langgraph.checkpoint.memory import InMemorySaver

def get_memory() -> InMemorySaver:
    return InMemorySaver()