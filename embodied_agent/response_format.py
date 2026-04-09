from pydantic import BaseModel
from typing import List, Union, Literal


class ToolResult(BaseModel):
    name: str
    args: dict
    result: Union[int, float, str]


class ResponseFormat(BaseModel):
    response: str
    answer: Union[int, float, str] | None = None
    tools: Union[List[ToolResult]] | None = None
    task_type: Literal["action", "query"] = "query"
    outcome: Literal["success", "failed"] = "success"
    failure_reason: str | None = None

    class Config:
        extra = "forbid"