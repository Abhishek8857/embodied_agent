import json
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver    

MEMORY_PATH = Path("memory/memory.json")


def get_memory() -> InMemorySaver:
    return InMemorySaver()

def get_memory_storage():
    store = InMemoryStore()
    if MEMORY_PATH.exists():
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        namespace = ("agent",)
        store.put(namespace, "memory", data)

    return store




