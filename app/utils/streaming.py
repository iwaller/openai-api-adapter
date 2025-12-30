import json
import time
import uuid
from collections.abc import AsyncIterator

from app.models.common import ChatRequest, StreamChunk
from app.providers.base import Provider


async def stream_generator(
    provider: Provider,
    request: ChatRequest,
    api_key: str,
) -> AsyncIterator[str]:
    """
    Convert provider stream chunks to OpenAI SSE format.

    Yields SSE-formatted strings for streaming responses.
    """
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(time.time())
    model = request.model

    async for chunk in provider.chat_stream(request, api_key):
        if chunk.type == "start":
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "system_fingerprint": None,
            }
            yield f"data: {json.dumps(data)}\n\n"

        elif chunk.type == "delta":
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk.content},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                "system_fingerprint": None,
            }
            yield f"data: {json.dumps(data)}\n\n"

        elif chunk.type == "stop":
            data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "system_fingerprint": None,
            }
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
