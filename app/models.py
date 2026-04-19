from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    max_new_tokens: Optional[int] = Field(None, ge=1, le=2048)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stream: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What are the key risk factors in a high debt-to-equity ratio?"
                        }
                    ],
                    "temperature": 0.3,
                    "stream": False
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    content: str
    model: str
    finish_reason: str = "stop"
    usage: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model: str
    adapter_loaded: bool
    device: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
