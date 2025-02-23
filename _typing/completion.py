from typing import List

from langchain.schema import Document
from pydantic import BaseModel, Field, validator


class CompletionInput(BaseModel):
    message: str = Field(...)


class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: str
    message: str
    type: str

    @validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["bot", "you"]:
            raise ValueError("sender must be bot or you")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in ["start", "stream", "end", "error", "info", "source"]:
            raise ValueError("type must be start, stream or end")
        return v
