from dataclasses import dataclass
from typing_extensions import TypedDict

@dataclass
class UserInfo(TypedDict):
    name: str

def get_pre_model_hook():
    pass