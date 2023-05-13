from typing import List

from langchain.schema import Document
from pydantic import BaseModel, Field


class CompletionInput(BaseModel):
    message: str = Field(...)


class CompletionOutput(BaseModel):
    answer: str
    docs: List[Document]
