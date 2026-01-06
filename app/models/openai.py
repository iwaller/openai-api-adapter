from typing import Any, Literal

from pydantic import BaseModel, Field


# Request models


class OpenAIImageUrl(BaseModel):
    """OpenAI image URL content."""

    url: str
    detail: str | None = None


class OpenAIInputAudio(BaseModel):
    """OpenAI audio input content."""

    data: str  # base64 encoded audio
    format: str  # e.g., "wav", "mp3"


class OpenAIContentPart(BaseModel):
    """OpenAI content part - flexible to support various content types.

    Standard OpenAI types: text, image_url, input_audio
    Cursor/Claude types: tool_use, tool_result (passed through directly)
    """

    type: str  # Flexible type to support Cursor's Claude-style content
    text: str | None = None
    image_url: OpenAIImageUrl | None = None
    input_audio: OpenAIInputAudio | None = None  # Will be stripped for Claude
    # Cursor/Claude-specific fields (passed through)
    id: str | None = None  # tool_use id
    name: str | None = None  # tool name
    input: Any | None = None  # tool input
    tool_use_id: str | None = None  # tool_result reference
    content: Any | None = None  # tool_result content
    cache_control: Any | None = None  # Cursor cache control


class OpenAIFunctionCall(BaseModel):
    """OpenAI function call in assistant message."""

    name: str
    arguments: str  # JSON string


class OpenAIToolCall(BaseModel):
    """OpenAI tool call in assistant message."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    """OpenAI chat message format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[OpenAIContentPart] | None = None
    # For assistant messages with tool calls
    tool_calls: list[OpenAIToolCall] | None = None
    # For tool response messages
    tool_call_id: str | None = None
    name: str | None = None  # Function name for tool responses


class OpenAIStreamOptions(BaseModel):
    """OpenAI stream options."""

    include_usage: bool = False


class OpenAIChatRequest(BaseModel):
    """OpenAI chat completion request format."""

    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    stream_options: OpenAIStreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None  # Newer OpenAI parameter
    # These are ignored but accepted for compatibility
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None
    # Function calling (ignored per Claude API behavior note)
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    functions: list[Any] | None = None
    function_call: Any | None = None


# Response models


class OpenAIMessageResponse(BaseModel):
    """OpenAI response message."""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAIChoice(BaseModel):
    """OpenAI response choice."""

    index: int = 0
    message: OpenAIMessageResponse
    logprobs: Any | None = None
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    """OpenAI token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    """OpenAI chat completion response format."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage
    system_fingerprint: str | None = None


# Streaming response models


class OpenAIDelta(BaseModel):
    """OpenAI streaming delta content."""

    role: str | None = None
    content: str | None = None


class OpenAIStreamChoice(BaseModel):
    """OpenAI streaming choice."""

    index: int = 0
    delta: OpenAIDelta
    logprobs: Any | None = None
    finish_reason: str | None = None


class OpenAIStreamChunk(BaseModel):
    """OpenAI streaming chunk format."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIStreamChoice]
    system_fingerprint: str | None = None


# Model listing


class OpenAIModel(BaseModel):
    """OpenAI model info."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "anthropic"


class OpenAIModelsResponse(BaseModel):
    """OpenAI models list response."""

    object: str = "list"
    data: list[OpenAIModel]


# Error response


class OpenAIErrorDetail(BaseModel):
    """OpenAI error detail."""

    type: str
    message: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI error response format."""

    error: OpenAIErrorDetail
