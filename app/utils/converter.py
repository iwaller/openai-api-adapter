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
            content_blocks: list[ContentBlock] = []
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
                elif part.type == "tool_use":
                    # Cursor sends Claude-style tool_use directly
                    content_blocks.append(
                        ContentBlock(
                            type="tool_use",
                            tool_use=ToolUse(
                                id=part.id or "",
                                name=part.name or "",
                                input=part.input or {},
                            ),
                        )
                    )
                elif part.type == "tool_result":
                    # Cursor sends Claude-style tool_result directly
                    # Extract text content from nested content
                    result_content = ""
                    if isinstance(part.content, str):
                        result_content = part.content
                    elif isinstance(part.content, list):
                        # Content is a list of blocks, extract text
                        text_parts = []
                        for block in part.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        result_content = "\n".join(text_parts)

                    content_blocks.append(
                        ContentBlock(
                            type="tool_result",
                            tool_result=ToolResult(
                                tool_use_id=part.tool_use_id or "",
                                content=result_content,
                            ),
                        )
                    )

            if content_blocks:
                messages.append(Message(role=openai_msg.role, content=content_blocks))

    # Flush any remaining tool results at the end
    flush_tool_results()

    # Convert tools if present (strict parameter is ignored)
    # Supports both OpenAI format and Cursor's direct format
    from app.utils.logger import logger
    tools: list[ToolDefinition] | None = None
    if request.tools:
        logger.info(f"Request has {len(request.tools)} tools, first tool type: {type(request.tools[0])}")
        if request.tools:
            first_tool = request.tools[0]
            logger.debug(f"First tool content: {first_tool}")
        tools = []
        for tool in request.tools:
            # Handle both dict and Pydantic model
            if isinstance(tool, dict):
                tool_dict = tool
            else:
                # Pydantic model - convert to dict first
                tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)

            # Check for standard OpenAI format: {"type": "function", "function": {...}}
            if tool_dict.get("type") == "function" and tool_dict.get("function"):
                func = tool_dict["function"]
                tools.append(
                    ToolDefinition(
                        name=func.get("name", ""),
                        description=func.get("description"),
                        input_schema=func.get("parameters", {}),
                    )
                )
            # Check for Cursor's direct format: {"name": ..., "input_schema": ...}
            elif tool_dict.get("name") and tool_dict.get("input_schema"):
                tools.append(
                    ToolDefinition(
                        name=tool_dict["name"],
                        description=tool_dict.get("description"),
                        input_schema=tool_dict["input_schema"],
                    )
                )
        logger.info(f"Converted {len(tools)} tools to ToolDefinition")

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

    # Convert tool_choice to Claude format (use dict format for consistency)
    # OpenAI: "auto", "none", "required", or {"type": "function", "function": {"name": "..."}}
    # Claude: {"type": "auto"}, {"type": "any"}, {"type": "none"}, or {"type": "tool", "name": "..."}
    tool_choice = None
    if request.tool_choice:
        if isinstance(request.tool_choice, str):
            if request.tool_choice == "required":
                tool_choice = {"type": "any"}  # Claude equivalent
            elif request.tool_choice in ("auto", "none"):
                tool_choice = {"type": request.tool_choice}
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
