from pydantic import BaseModel, field_validator
from typing import List, Union, Literal, Any
import json


class ToolResult(BaseModel):
    name: str
    args: dict
    result: Any = None

    @field_validator("result", mode="before")
    @classmethod
    def coerce_complex_to_str(cls, v):
        if isinstance(v, (list, dict)):
            return json.dumps(v)
        return v


class ResponseFormat(BaseModel):
    response: str
    answer: Union[int, float, str] | None = None
    tools: Union[List[ToolResult]] | None = None
    task_type: Literal["action", "query"] = "query"
    outcome: Literal["success", "failed"] = "success"
    failure_reason: str | None = None

    class Config:
        extra = "forbid"