from pydantic import BaseModel
from typing import List, Union


class ToolResult(BaseModel):
    """Response format for the Tool Call"""
    name: str
    args: dict
    result: Union[int, float, str]
    

class ResponseFormat(BaseModel):
    """Response format for the Agent"""
    response: str
    answer: Union[int, float, str] | None = None
    tools: Union[List[ToolResult]] | None = None
    
    class Config:
        extra = "forbid"