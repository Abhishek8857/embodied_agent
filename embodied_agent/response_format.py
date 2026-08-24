from typing import Any, List, Literal, Union
from pydantic import BaseModel, ConfigDict, field_validator
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
    model_config = ConfigDict(extra="forbid")

    response: str
    # Accepts scalar primitives, lists, dictionaries, or None
    answer: Union[int, float, str, List[Any], dict, None] = None
    tools: List[ToolResult] | None = None
    task_type: Literal["action", "query"] = "query"
    outcome: Literal["success", "failed"] = "success"
    failure_reason: str | None = None