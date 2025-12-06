from typing import Literal, TypedDict, NotRequired, Optional, Union

from openai.types.chat import ChatCompletionMessageToolCallUnion
import torch

class UserMessage(TypedDict):
    role: Literal["user"]
    content: str

class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str
    tool_calls: Optional[list[ChatCompletionMessageToolCallUnion]] = None

class ToolMessage(TypedDict):
    role: Literal["tool"]
    content: str
    tool_call_id: str

class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str

APIMessage = Union[UserMessage, AssistantMessage, ToolMessage, SystemMessage]
