import json
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver    

MEMORY_PATH = Path("memory/memory.json")


def get_memory() -> InMemorySaver:
    return InMemorySaver()

def get_memory_storage():
    store = InMemoryStore()  # persistent store alternative: PostgresStore.from_conn_string(...)
    if MEMORY_PATH.exists():
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        # choose a namespace & key convention (example: ("agent",) / "memory")
        namespace = ("agent",)
        # store the whole JSON under one key so your tools/agent can read it
        store.put(namespace, "memory", data)
        # optionally populate more granular keys (preferences, episodes, etc.)
        # store.put(namespace, "preferences", data.get("world_state", {}).get("preferences"))
    return store




