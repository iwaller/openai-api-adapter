from typing import Literal

from pydantic import BaseModel


class ImageSource(BaseModel):
    """Image source for multimodal content."""

    type: Literal["base64", "url"]
    media_type: str
    data: str


class ToolUse(BaseModel):
    """Tool use block for assistant messages."""

    id: str
    name: str
    input: dict  # Tool input as parsed JSON


class ToolResult(BaseModel):
    """Tool result block for user messages."""

    tool_use_id: str
    content: str


class ContentBlock(BaseModel):
    """Content block that can be text, image, tool_use, or tool_result."""

    type: Literal["text", "image", "tool_use", "tool_result"]
    text: str | None = None
    source: ImageSource | None = None
    tool_use: ToolUse | None = None
    tool_result: ToolResult | None = None


class Message(BaseModel):
    """Chat message with role and content."""

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentBlock]


class ToolDefinition(BaseModel):
    """Tool definition for function calling."""

    name: str
    description: str | None = None
    input_schema: dict  # JSON Schema


class ChatRequest(BaseModel):
    """Internal chat request format."""

    model: str
    messages: list[Message]
    max_tokens: int = 65536
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None  # Stop sequences
    stream: bool = False
    stream_include_usage: bool = False  # OpenAI stream_options.include_usage
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "any", "none", or {"type": "tool", "name": "..."}


class ChatResponse(BaseModel):
    """Internal chat response format."""

    id: str
    model: str
    content: str | None = None
    tool_calls: list[ToolUse] | None = None
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"


class StreamToolCall(BaseModel):
    """Tool call information for streaming."""

    index: int
    id: str | None = None  # Only in first chunk
    name: str | None = None  # Only in first chunk
    arguments_delta: str = ""  # Incremental JSON


class StreamChunk(BaseModel):
    """Streaming chunk for SSE responses."""

    type: Literal["start", "delta", "tool_call_start", "tool_call_delta", "stop"]
    content: str = ""
    model: str | None = None
    tool_call: StreamToolCall | None = None
    finish_reason: str | None = None  # "stop" or "tool_calls"
    input_tokens: int | None = None  # For stop chunk
    output_tokens: int | None = None  # For stop chunk


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    owned_by: str = "provider"
