import json
import time
import uuid
from collections.abc import AsyncIterator

from openai_api_adapter.config import settings
from openai_api_adapter.models.common import ChatRequest, StreamChunk
from openai_api_adapter.providers.base import Provider
from openai_api_adapter.utils.logger import log_response, log_stream_chunk, log_stream_end, log_stream_start

# Max content size to accumulate for logging (to prevent unbounded memory growth)
MAX_LOG_CONTENT_SIZE = 50000  # 50KB


async def stream_generator(
    provider: Provider,
    request: ChatRequest,
    api_key: str,
    request_id: str = "",
) -> AsyncIterator[str]:
    """
    Convert provider stream chunks to OpenAI SSE format.

    Supports both text content and tool calls streaming.
    Yields SSE-formatted strings for streaming responses.
    """
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(time.time())
    model = request.model

    # Collect content for logging with size limit to prevent memory issues
    full_content: list[str] = []
    full_content_size = 0
    content_truncated = False

    # Use dict for tool calls to handle out-of-order indices
    tool_calls_log: dict[int, dict] = {}
    finish_reason = "stop"

    try:
        async for chunk in provider.chat_stream(request, api_key):
            if chunk.type == "start":
                log_stream_start(request_id, model)
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
                # Text content delta - accumulate with size limit
                if not content_truncated:
                    if full_content_size + len(chunk.content) <= MAX_LOG_CONTENT_SIZE:
                        full_content.append(chunk.content)
                        full_content_size += len(chunk.content)
                    else:
                        content_truncated = True
                log_stream_chunk(request_id, chunk.content)
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

            elif chunk.type == "tool_call_start":
                # Tool call start - send id, type, and function name
                if chunk.tool_call:
                    # Track tool call for logging using dict with index as key
                    tool_calls_log[chunk.tool_call.index] = {
                        "id": chunk.tool_call.id,
                        "name": chunk.tool_call.name,
                        "arguments": "",
                    }

                    data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": timestamp,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": chunk.tool_call.index,
                                            "id": chunk.tool_call.id,
                                            "type": "function",
                                            "function": {
                                                "name": chunk.tool_call.name,
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                },
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                        "system_fingerprint": None,
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            elif chunk.type == "tool_call_delta":
                # Tool call arguments delta
                if chunk.tool_call:
                    # Append to tool call arguments for logging (using dict lookup)
                    if chunk.tool_call.index in tool_calls_log:
                        tool_calls_log[chunk.tool_call.index]["arguments"] += chunk.tool_call.arguments_delta

                    data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": timestamp,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": chunk.tool_call.index,
                                            "function": {
                                                "arguments": chunk.tool_call.arguments_delta,
                                            },
                                        }
                                    ]
                                },
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                        "system_fingerprint": None,
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            elif chunk.type == "stop":
                log_stream_end(request_id)
                finish_reason = chunk.finish_reason or "stop"
                input_tokens = chunk.input_tokens or 0
                output_tokens = chunk.output_tokens or 0

                # Log complete response with content and/or tool calls
                log_content = "".join(full_content) if full_content else None
                if tool_calls_log:
                    # Include tool calls in log
                    tool_calls_str = json.dumps(tool_calls_log, ensure_ascii=False)
                    if log_content:
                        log_content = f"{log_content}\n[Tool Calls: {tool_calls_str}]"
                    else:
                        log_content = f"[Tool Calls: {tool_calls_str}]"

                log_response(
                    request_id=request_id,
                    content=log_content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    finish_reason=finish_reason,
                )

                # Send final chunk with finish_reason
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
                            "finish_reason": finish_reason,
                        }
                    ],
                    "system_fingerprint": None,
                }
                yield f"data: {json.dumps(data)}\n\n"

                # Always send usage chunk (some clients expect it even without stream_options)
                # Send even if tokens are 0 to ensure override values are reported
                if input_tokens > 0 or output_tokens > 0 or settings.override_usage:
                    usage_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": timestamp,
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                        "system_fingerprint": None,
                    }
                    yield f"data: {json.dumps(usage_data)}\n\n"

                yield "data: [DONE]\n\n"

    except Exception as e:
        log_response(request_id=request_id, error=str(e))
        # Send error in SSE format before connection drops
        error_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": None,
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"
