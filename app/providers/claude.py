from collections.abc import AsyncIterator
from typing import Any

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError as AnthropicAuthError,
    RateLimitError as AnthropicRateLimitError,
)

from app.config import settings
from app.exceptions import (
    AuthenticationError,
    ConnectionError,
    ProviderAPIError,
    RateLimitError,
)
from app.models.common import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    ImageSource,
    Message,
    ModelInfo,
    StreamChunk,
    StreamToolCall,
    ToolUse,
)
from app.providers.base import Provider


def _map_finish_reason(claude_reason: str | None) -> str:
    """Map Claude stop_reason to OpenAI finish_reason."""
    mapping = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
    }
    return mapping.get(claude_reason or "", "stop")


class ClaudeProvider(Provider):
    """Claude provider using Anthropic SDK."""

    @property
    def name(self) -> str:
        return "claude"

    def _get_client(self, api_key: str) -> AsyncAnthropic:
        """Create Anthropic client with optional custom base URL."""
        kwargs: dict[str, Any] = {"api_key": api_key}
        if settings.claude_base_url:
            kwargs["base_url"] = settings.claude_base_url
        return AsyncAnthropic(**kwargs)

    def _extract_system(
        self, messages: list[Message]
    ) -> tuple[list[Message], list[dict[str, Any]] | None]:
        """
        Extract and concatenate system messages.

        Claude only supports a single system message at the beginning,
        so we concatenate all system messages together.

        Returns system as a list of content blocks with cache_control
        for prompt caching support.
        """
        system_parts: list[str] = []
        other_messages: list[Message] = []

        for msg in messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_parts.append(msg.content)
                else:
                    # Extract text from content blocks
                    for block in msg.content:
                        if block.type == "text" and block.text:
                            system_parts.append(block.text)
            else:
                other_messages.append(msg)

        if not system_parts:
            return other_messages, None

        # Return as list of content blocks with cache_control on the last block
        # This enables Claude's prompt caching feature
        system_text = "\n".join(system_parts)
        system_blocks = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        return other_messages, system_blocks

    def _convert_content_block(self, block: ContentBlock) -> dict[str, Any]:
        """Convert a ContentBlock to Anthropic format."""
        if block.type == "text":
            return {"type": "text", "text": block.text or ""}
        elif block.type == "image" and block.source:
            if block.source.type == "base64":
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.source.media_type,
                        "data": block.source.data,
                    },
                }
            else:  # url
                return {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": block.source.data,
                    },
                }
        elif block.type == "tool_use" and block.tool_use:
            return {
                "type": "tool_use",
                "id": block.tool_use.id,
                "name": block.tool_use.name,
                "input": block.tool_use.input,
            }
        elif block.type == "tool_result" and block.tool_result:
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_result.tool_use_id,
                "content": block.tool_result.content,
            }
        return {"type": "text", "text": ""}

    def _convert_messages(
        self, messages: list[Message], enable_caching: bool = True
    ) -> list[dict[str, Any]]:
        """Convert common Message format to Anthropic SDK format.

        Optionally adds cache_control to strategic messages for prompt caching.
        Strategy: Cache the second-to-last user message to cache conversation history.
        """
        result: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                content_blocks = [
                    self._convert_content_block(block) for block in msg.content
                ]
                result.append({"role": msg.role, "content": content_blocks})

        # Add cache_control to strategic messages for conversation caching
        if enable_caching and len(result) >= 3:
            # Find the second-to-last user message to cache conversation history
            user_msg_indices = [i for i, m in enumerate(result) if m["role"] == "user"]
            if len(user_msg_indices) >= 2:
                # Cache up to the second-to-last user message
                cache_idx = user_msg_indices[-2]
                msg_to_cache = result[cache_idx]

                # Add cache_control to the last content block
                if isinstance(msg_to_cache["content"], list) and msg_to_cache["content"]:
                    msg_to_cache["content"][-1]["cache_control"] = {"type": "ephemeral"}
                elif isinstance(msg_to_cache["content"], str):
                    # Convert string to content block with cache_control
                    msg_to_cache["content"] = [
                        {
                            "type": "text",
                            "text": msg_to_cache["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]

        return result

    def _convert_tools(self, request: ChatRequest) -> list[dict[str, Any]] | None:
        """Convert tools to Anthropic format with caching support.

        Adds cache_control to the last tool for prompt caching.
        This is effective because Cursor typically sends many tools.
        """
        from app.utils.logger import logger

        if not request.tools:
            logger.debug("No tools in request")
            return None

        tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
            for tool in request.tools
        ]

        # Add cache_control to the last tool for prompt caching
        if tools:
            tools[-1]["cache_control"] = {"type": "ephemeral"}

        logger.info(f"Converted {len(tools)} tools for Claude API (with caching)")
        if tools:
            logger.debug(f"First tool: name={tools[0]['name']}, schema_keys={list(tools[0]['input_schema'].keys())}")

        return tools

    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        """Non-streaming chat completion."""
        client = self._get_client(api_key)
        messages, system = self._extract_system(request.messages)

        try:
            kwargs: dict[str, Any] = {
                "model": request.model,
                "max_tokens": request.max_tokens,
                "messages": self._convert_messages(messages),
            }

            if system:
                kwargs["system"] = system
            if request.temperature is not None:
                # Cap temperature at 1.0 as per Claude docs
                kwargs["temperature"] = min(request.temperature, 1.0)
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p
            if request.stop:
                kwargs["stop_sequences"] = request.stop

            # Add tools if present
            tools = self._convert_tools(request)
            if tools:
                kwargs["tools"] = tools
                # Add tool_choice if specified (default to auto if tools present)
                if request.tool_choice:
                    kwargs["tool_choice"] = request.tool_choice
                else:
                    kwargs["tool_choice"] = {"type": "auto"}

            # Log kwargs being sent to Claude (excluding messages for brevity)
            from app.utils.logger import logger
            log_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
            log_kwargs["tools_count"] = len(tools) if tools else 0
            log_kwargs["messages_count"] = len(kwargs.get("messages", []))
            logger.info(f"Claude API kwargs: {log_kwargs}")

            response = await client.messages.create(**kwargs)

            # Log cache statistics if available
            usage = response.usage
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            if cache_creation > 0 or cache_read > 0:
                logger.info(
                    f"Cache stats: created={cache_creation}, read={cache_read}, "
                    f"total_input={usage.input_tokens}"
                )

            # Extract content and tool calls from response
            content = ""
            tool_calls: list[ToolUse] = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolUse(
                            id=block.id,
                            name=block.name,
                            input=block.input,
                        )
                    )

            # Map Claude stop_reason to OpenAI finish_reason
            finish_reason = _map_finish_reason(response.stop_reason)

            return ChatResponse(
                id=response.id,
                model=response.model,
                content=content if content else None,
                tool_calls=tool_calls if tool_calls else None,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                finish_reason=finish_reason,
            )

        except AnthropicAuthError as e:
            raise AuthenticationError(str(e))
        except AnthropicRateLimitError as e:
            raise RateLimitError(str(e))
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to Claude API: {e}")
        except APIStatusError as e:
            from app.utils.logger import logger
            logger.error(f"Claude API error: status={e.status_code}, message={e.message}, body={e.body}")
            raise ProviderAPIError(e.status_code, str(e))

    async def chat_stream(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion with tool call support."""
        client = self._get_client(api_key)
        messages, system = self._extract_system(request.messages)

        try:
            kwargs: dict[str, Any] = {
                "model": request.model,
                "max_tokens": request.max_tokens,
                "messages": self._convert_messages(messages),
            }

            if system:
                kwargs["system"] = system
            if request.temperature is not None:
                # Cap temperature at 1.0 as per Claude docs
                kwargs["temperature"] = min(request.temperature, 1.0)
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p
            if request.stop:
                kwargs["stop_sequences"] = request.stop

            # Add tools if present
            tools = self._convert_tools(request)
            if tools:
                kwargs["tools"] = tools
                # Add tool_choice if specified (default to auto if tools present)
                if request.tool_choice:
                    kwargs["tool_choice"] = request.tool_choice
                else:
                    kwargs["tool_choice"] = {"type": "auto"}

            # Log kwargs being sent to Claude (excluding messages for brevity)
            from app.utils.logger import logger
            log_kwargs = {k: v for k, v in kwargs.items() if k != "messages"}
            log_kwargs["tools_count"] = len(tools) if tools else 0
            log_kwargs["messages_count"] = len(kwargs.get("messages", []))
            logger.info(f"Claude API stream kwargs: {log_kwargs}")

            async with client.messages.stream(**kwargs) as stream:
                # Send start chunk
                yield StreamChunk(type="start", model=request.model)

                # Track tool calls by index (content block index)
                tool_call_index_map: dict[int, int] = {}  # block_index -> tool_call_index
                current_tool_index = 0
                finish_reason = "stop"
                input_tokens = 0
                output_tokens = 0

                async for event in stream:
                    if event.type == "message_start":
                        # Capture input tokens and cache stats from message_start
                        if hasattr(event.message, "usage"):
                            usage = event.message.usage
                            input_tokens = usage.input_tokens
                            # Log cache statistics if available
                            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
                            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                            if cache_creation > 0 or cache_read > 0:
                                logger.info(
                                    f"Cache stats: created={cache_creation}, read={cache_read}, "
                                    f"total_input={input_tokens}"
                                )

                    elif event.type == "content_block_start":
                        # Check if this is a tool use block
                        if hasattr(event.content_block, "type"):
                            if event.content_block.type == "tool_use":
                                # Map block index to tool call index
                                tool_call_index_map[event.index] = current_tool_index
                                yield StreamChunk(
                                    type="tool_call_start",
                                    tool_call=StreamToolCall(
                                        index=current_tool_index,
                                        id=event.content_block.id,
                                        name=event.content_block.name,
                                    ),
                                )
                                current_tool_index += 1

                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            # Text content delta
                            yield StreamChunk(type="delta", content=event.delta.text)
                        elif hasattr(event.delta, "partial_json"):
                            # Tool input JSON delta
                            tool_idx = tool_call_index_map.get(event.index, 0)
                            yield StreamChunk(
                                type="tool_call_delta",
                                tool_call=StreamToolCall(
                                    index=tool_idx,
                                    arguments_delta=event.delta.partial_json,
                                ),
                            )

                    elif event.type == "message_delta":
                        # Get finish reason and output tokens from message delta
                        if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                            finish_reason = _map_finish_reason(event.delta.stop_reason)
                        if hasattr(event, "usage") and event.usage:
                            output_tokens = event.usage.output_tokens

                # Send stop chunk with finish reason and usage
                yield StreamChunk(
                    type="stop",
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

        except AnthropicAuthError as e:
            raise AuthenticationError(str(e))
        except AnthropicRateLimitError as e:
            raise RateLimitError(str(e))
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to Claude API: {e}")
        except APIStatusError as e:
            from app.utils.logger import logger
            logger.error(f"Claude API error: status={e.status_code}, message={e.message}, body={e.body}")
            raise ProviderAPIError(e.status_code, str(e))

    def list_models(self) -> list[ModelInfo]:
        """Return available Claude models."""
        return [
            ModelInfo(id=model_id, owned_by="anthropic")
            for model_id in settings.claude_allowed_models
        ]
