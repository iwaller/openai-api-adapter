import json
import time
import uuid

from app.config import settings
from app.models.common import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    ImageSource,
    Message,
    ToolDefinition,
    ToolResult,
    ToolUse,
)
from app.models.openai import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAIFunctionCall,
    OpenAIMessageResponse,
    OpenAIToolCall,
    OpenAIUsage,
)


def convert_openai_to_common(request: OpenAIChatRequest, model: str) -> ChatRequest:
    """
    Convert OpenAI request format to common internal format.

    Note: Audio input is stripped as Claude does not support it.
    Note: Tool strict parameter is ignored as Claude does not guarantee schema conformance.

    Args:
        request: OpenAI-format chat request.
        model: Actual model name (after prefix parsing).

    Returns:
        Common ChatRequest format.
    """
    messages: list[Message] = []

    # Collect consecutive tool results to merge into single user message
    pending_tool_results: list[ContentBlock] = []

    def flush_tool_results() -> None:
        """Flush accumulated tool results into a single user message."""
        nonlocal pending_tool_results
        if pending_tool_results:
            messages.append(Message(role="user", content=pending_tool_results))
            pending_tool_results = []

    for openai_msg in request.messages:
        # Handle tool response messages - accumulate for merging
        if openai_msg.role == "tool":
            if openai_msg.tool_call_id and openai_msg.content:
                content_str = (
                    openai_msg.content
                    if isinstance(openai_msg.content, str)
                    else str(openai_msg.content)
                )
                pending_tool_results.append(
                    ContentBlock(
                        type="tool_result",
                        tool_result=ToolResult(
                            tool_use_id=openai_msg.tool_call_id,
                            content=content_str,
                        ),
                    )
                )
            continue

        # Flush any pending tool results before processing non-tool message
        flush_tool_results()

        # Handle assistant messages with tool_calls
        if openai_msg.role == "assistant" and openai_msg.tool_calls:
            content_blocks: list[ContentBlock] = []

            # Add text content if present
            if openai_msg.content:
                if isinstance(openai_msg.content, str):
                    content_blocks.append(
                        ContentBlock(type="text", text=openai_msg.content)
                    )

            # Add tool use blocks
            for tool_call in openai_msg.tool_calls:
                try:
                    input_data = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    input_data = {"raw": tool_call.function.arguments}

                content_blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use=ToolUse(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=input_data,
                        ),
                    )
                )

            if content_blocks:
                messages.append(Message(role="assistant", content=content_blocks))
            continue

        # Handle regular messages
        if isinstance(openai_msg.content, str):
            messages.append(
                Message(role=openai_msg.role, content=openai_msg.content)
            )
        elif openai_msg.content:
            # Convert content parts, stripping audio
            content_blocks = []
            for part in openai_msg.content:
                if part.type == "text":
                    content_blocks.append(
                        ContentBlock(type="text", text=part.text or "")
                    )
                elif part.type == "image_url" and part.image_url:
                    url = part.image_url.url
                    # Handle base64 data URLs
                    if url.startswith("data:image/"):
                        # Parse data URL: data:image/png;base64,xxxxx
                        parts = url.split(",", 1)
                        if len(parts) == 2:
                            media_info = parts[0]  # data:image/png;base64
                            data = parts[1]
                            # Extract media type
                            media_type = media_info.replace("data:", "").replace(
                                ";base64", ""
                            )
                            content_blocks.append(
                                ContentBlock(
                                    type="image",
                                    source=ImageSource(
                                        type="base64",
                                        media_type=media_type,
                                        data=data,
                                    ),
                                )
                            )
                    else:
                        # HTTP URL
                        content_blocks.append(
                            ContentBlock(
                                type="image",
                                source=ImageSource(
                                    type="url",
                                    media_type="image/jpeg",  # Default
                                    data=url,
                                ),
                            )
                        )
                elif part.type == "input_audio":
                    # Audio input is stripped (not supported by Claude)
                    pass

            if content_blocks:
                messages.append(Message(role=openai_msg.role, content=content_blocks))

    # Flush any remaining tool results at the end
    flush_tool_results()

    # Convert tools if present (strict parameter is ignored)
    tools: list[ToolDefinition] | None = None
    if request.tools:
        tools = []
        for tool in request.tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                func = tool.get("function", {})
                tools.append(
                    ToolDefinition(
                        name=func.get("name", ""),
                        description=func.get("description"),
                        input_schema=func.get("parameters", {}),
                    )
                )

    # Check if stream_options.include_usage is set
    include_usage = False
    if request.stream_options and request.stream_options.include_usage:
        include_usage = True

    # Use max_completion_tokens if provided, fallback to max_tokens, then default
    max_tokens = (
        request.max_completion_tokens
        or request.max_tokens
        or settings.default_max_tokens
    )

    # Convert stop sequences (filter whitespace-only sequences as per Claude docs)
    stop_sequences: list[str] | None = None
    if request.stop:
        if isinstance(request.stop, str):
            if request.stop.strip():
                stop_sequences = [request.stop]
        else:
            stop_sequences = [s for s in request.stop if s.strip()]
        if not stop_sequences:
            stop_sequences = None

    # Convert tool_choice to Claude format
    # OpenAI: "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}
    # Claude: "auto", "any", or {"type": "tool", "name": "..."}
    tool_choice = None
    if request.tool_choice:
        if isinstance(request.tool_choice, str):
            if request.tool_choice == "required":
                tool_choice = "any"  # Claude equivalent
            elif request.tool_choice in ("auto", "none"):
                tool_choice = request.tool_choice
        elif isinstance(request.tool_choice, dict):
            # {"type": "function", "function": {"name": "..."}} -> {"type": "tool", "name": "..."}
            func = request.tool_choice.get("function", {})
            if func.get("name"):
                tool_choice = {"type": "tool", "name": func["name"]}

    return ChatRequest(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=stop_sequences,
        stream=request.stream,
        stream_include_usage=include_usage,
        tools=tools,
        tool_choice=tool_choice,
    )


def convert_common_to_openai(response: ChatResponse) -> OpenAIChatResponse:
    """
    Convert common response format to OpenAI format.

    Args:
        response: Common ChatResponse format.

    Returns:
        OpenAI-format chat response.
    """
    # Convert tool calls if present
    tool_calls: list[OpenAIToolCall] | None = None
    if response.tool_calls:
        tool_calls = [
            OpenAIToolCall(
                id=tc.id,
                type="function",
                function=OpenAIFunctionCall(
                    name=tc.name,
                    arguments=json.dumps(tc.input),
                ),
            )
            for tc in response.tool_calls
        ]

    # Use OpenAI-style ID format (chatcmpl-xxx) instead of Claude's msg_xxx
    chat_id = f"chatcmpl-{uuid.uuid4()}"

    return OpenAIChatResponse(
        id=chat_id,
        object="chat.completion",
        created=int(time.time()),
        model=response.model,
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIMessageResponse(
                    role="assistant",
                    content=response.content,
                    tool_calls=tool_calls,
                ),
                logprobs=None,
                finish_reason=response.finish_reason,
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=response.input_tokens,
            completion_tokens=response.output_tokens,
            total_tokens=response.input_tokens + response.output_tokens,
        ),
    )
