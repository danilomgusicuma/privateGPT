from typing import Any

from fastapi import WebSocket

from langchain.callbacks.base import AsyncCallbackHandler

from _typing.completion import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket: WebSocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.websocket.send_json(ChatResponse(sender="bot", message=token, type="stream").dict())
