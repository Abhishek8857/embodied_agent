from dataclasses import dataclass


@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


def format_message (msg: str) -> dict:
    return {"messages": [{"role" : "user" , "content": msg}]}