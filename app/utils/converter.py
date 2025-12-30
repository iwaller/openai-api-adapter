import json
import time

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

    for openai_msg in request.messages:
        # Handle tool response messages
        if openai_msg.role == "tool":
            if openai_msg.tool_call_id and openai_msg.content:
                content_str = (
                    openai_msg.content
                    if isinstance(openai_msg.content, str)
                    else str(openai_msg.content)
                )
                messages.append(
                    Message(
                        role="user",
                        content=[
                            ContentBlock(
                                type="tool_result",
                                tool_result=ToolResult(
                                    tool_use_id=openai_msg.tool_call_id,
                                    content=content_str,
                                ),
                            )
                        ],
                    )
                )
            continue

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

    return ChatRequest(
        model=model,
        messages=messages,
        max_tokens=request.max_tokens or 4096,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        tools=tools,
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

    return OpenAIChatResponse(
        id=response.id,
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
