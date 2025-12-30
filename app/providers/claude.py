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
    ToolUse,
)
from app.providers.base import Provider


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
    ) -> tuple[list[Message], str | None]:
        """
        Extract and concatenate system messages.

        Claude only supports a single system message at the beginning,
        so we concatenate all system messages together.
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

        system = "\n".join(system_parts) if system_parts else None
        return other_messages, system

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

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert common Message format to Anthropic SDK format."""
        result: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                content_blocks = [
                    self._convert_content_block(block) for block in msg.content
                ]
                result.append({"role": msg.role, "content": content_blocks})

        return result

    def _convert_tools(self, request: ChatRequest) -> list[dict[str, Any]] | None:
        """Convert tools to Anthropic format."""
        if not request.tools:
            return None

        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
            for tool in request.tools
        ]

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
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p

            # Add tools if present
            tools = self._convert_tools(request)
            if tools:
                kwargs["tools"] = tools

            response = await client.messages.create(**kwargs)

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

            # Determine finish reason
            finish_reason = "stop"
            if response.stop_reason == "tool_use":
                finish_reason = "tool_calls"
            elif response.stop_reason:
                finish_reason = response.stop_reason

            return ChatResponse(
                id=response.id,
                model=response.model,
                content=content if content else None,
                tool_calls=tool_calls if tool_calls else None,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                finish_reason=finish_reason,
            )

        except AnthropicAuthError as e:
            raise AuthenticationError(str(e))
        except AnthropicRateLimitError as e:
            raise RateLimitError(str(e))
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to Claude API: {e}")
        except APIStatusError as e:
            raise ProviderAPIError(e.status_code, str(e))

    async def chat_stream(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion."""
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
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p

            # Add tools if present
            tools = self._convert_tools(request)
            if tools:
                kwargs["tools"] = tools

            async with client.messages.stream(**kwargs) as stream:
                # Send start chunk
                yield StreamChunk(type="start", model=request.model)

                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield StreamChunk(type="delta", content=event.delta.text)

                # Send stop chunk
                yield StreamChunk(type="stop")

        except AnthropicAuthError as e:
            raise AuthenticationError(str(e))
        except AnthropicRateLimitError as e:
            raise RateLimitError(str(e))
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to Claude API: {e}")
        except APIStatusError as e:
            raise ProviderAPIError(e.status_code, str(e))

    def list_models(self) -> list[ModelInfo]:
        """Return available Claude models."""
        return [
            ModelInfo(id=model_id, owned_by="anthropic")
            for model_id in settings.claude_allowed_models
        ]
