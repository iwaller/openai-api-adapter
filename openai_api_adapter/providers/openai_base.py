"""Base provider for OpenAI-compatible APIs."""

import json
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthError,
)

from openai_api_adapter.exceptions import (
    AuthenticationError,
    ConnectionError,
    ProviderAPIError,
    RateLimitError,
)
from openai_api_adapter.models.common import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelInfo,
    StreamChunk,
    StreamToolCall,
    ToolDefinition,
    ToolUse,
)
from openai_api_adapter.providers.base import Provider
from openai_api_adapter.utils.logger import logger


def _safe_json_loads(s: str | None) -> dict:
    """Safely parse JSON string, returning empty dict on failure."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in tool arguments: {s[:100]}...")
        return {}


def _map_finish_reason(openai_reason: str | None) -> str:
    """Map OpenAI finish_reason to internal format."""
    mapping = {
        "stop": "stop",
        "tool_calls": "tool_calls",
        "length": "length",
        "content_filter": "content_filter",
    }
    return mapping.get(openai_reason or "", "stop")


class OpenAIBaseProvider(Provider):
    """Base provider for OpenAI-compatible APIs.

    Subclasses should implement:
    - name: Provider identifier
    - _get_base_url(): Return the API base URL
    - _get_allowed_models(): Return list of allowed model names
    - _get_default_model(): Return default model name

    Optional overrides for customization:
    - _filter_request_kwargs(): Filter/modify request parameters before sending
    """

    # Client cache: {(base_url, api_key): AsyncOpenAI}
    _client_cache: dict[tuple[str | None, str], AsyncOpenAI] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'aiberm', 'openai')."""
        pass

    @abstractmethod
    def _get_base_url(self) -> str | None:
        """Return the API base URL, or None for default OpenAI."""
        pass

    @abstractmethod
    def _get_allowed_models(self) -> list[str]:
        """Return list of allowed model names."""
        pass

    @abstractmethod
    def _get_default_model(self) -> str:
        """Return default model name."""
        pass

    def normalize_model_name(self, model_name: str) -> str:
        """Normalize model name, use default if not in allowed list."""
        if model_name not in self._get_allowed_models():
            return self._get_default_model()
        return model_name

    def _get_client(self, api_key: str) -> AsyncOpenAI:
        """Get or create OpenAI client with connection reuse."""
        base_url = self._get_base_url()
        cache_key = (base_url, api_key)

        if cache_key not in self._client_cache:
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client_cache[cache_key] = AsyncOpenAI(**kwargs)

        return self._client_cache[cache_key]

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal message format to OpenAI format."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                # Handle content blocks
                content_parts: list[dict[str, Any]] = []
                tool_calls: list[dict[str, Any]] = []

                for block in msg.content:
                    if block.type == "text" and block.text:
                        content_parts.append({"type": "text", "text": block.text})
                    elif block.type == "image" and block.source:
                        if block.source.type == "base64":
                            content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{block.source.media_type};base64,{block.source.data}"
                                    },
                                }
                            )
                        else:
                            content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": block.source.data},
                                }
                            )
                    elif block.type == "tool_use" and block.tool_use:
                        tool_calls.append(
                            {
                                "id": block.tool_use.id,
                                "type": "function",
                                "function": {
                                    "name": block.tool_use.name,
                                    "arguments": json.dumps(block.tool_use.input),
                                },
                            }
                        )
                    elif block.type == "tool_result" and block.tool_result:
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.tool_result.tool_use_id,
                                "content": block.tool_result.content,
                            }
                        )
                        continue

                if tool_calls:
                    # Assistant message with tool calls
                    msg_dict: dict[str, Any] = {"role": msg.role}
                    if content_parts:
                        msg_dict["content"] = content_parts
                    msg_dict["tool_calls"] = tool_calls
                    result.append(msg_dict)
                elif content_parts:
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        result.append(
                            {"role": msg.role, "content": content_parts[0]["text"]}
                        )
                    else:
                        result.append({"role": msg.role, "content": content_parts})

        return result

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert internal tool format to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def _convert_tool_choice(self, tool_choice: str | dict | None) -> str | dict | None:
        """Convert internal tool_choice format to OpenAI format.

        Override this method to customize tool_choice handling.
        """
        if tool_choice is None:
            return None

        if isinstance(tool_choice, str):
            return tool_choice

        # Convert from internal format to OpenAI format
        tool_choice_type = tool_choice.get("type")
        if tool_choice_type == "tool":
            return {"type": "function", "function": {"name": tool_choice.get("name")}}
        if tool_choice_type == "any":
            return "required"
        if tool_choice_type in {"auto", "none", "required"}:
            return tool_choice_type
        return tool_choice

    def _build_base_request_kwargs(self, request: ChatRequest) -> dict:
        """Build base request kwargs for OpenAI API.

        This builds the common parameters. Subclasses can override
        _filter_request_kwargs() to modify these before sending.
        """
        model = self.normalize_model_name(request.model)
        kwargs = {
            "model": model,
            "messages": self._convert_messages(request.messages),
            "stream": request.stream,
        }

        # Token limits - include both for maximum compatibility
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
            kwargs["max_completion_tokens"] = request.max_tokens

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.stop:
            kwargs["stop"] = request.stop

        if request.tools:
            kwargs["tools"] = self._convert_tools(request.tools)

        if request.tool_choice:
            converted = self._convert_tool_choice(request.tool_choice)
            if converted:
                kwargs["tool_choice"] = converted

        if request.stream and request.stream_include_usage:
            kwargs["stream_options"] = {"include_usage": True}

        return kwargs

    def _filter_request_kwargs(self, kwargs: dict) -> dict:
        """Filter or modify request kwargs before sending.

        Override this method in subclasses to:
        - Remove unsupported parameters
        - Add provider-specific parameters
        - Modify parameter values

        Args:
            kwargs: The request kwargs built by _build_base_request_kwargs()

        Returns:
            Modified kwargs dict
        """
        return kwargs

    def _build_request_kwargs(self, request: ChatRequest) -> dict:
        """Build final request kwargs by applying filters."""
        kwargs = self._build_base_request_kwargs(request)
        return self._filter_request_kwargs(kwargs)

    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        """Non-streaming chat completion."""
        client = self._get_client(api_key)

        try:
            kwargs = self._build_request_kwargs(request)
            logger.debug(
                f"{self.name} request: model={request.model}, params={list(kwargs.keys())}"
            )

            response = await client.chat.completions.create(**kwargs)

            # Extract response content
            choice = response.choices[0]
            message = choice.message

            content = message.content
            tool_calls = None

            if message.tool_calls:
                tool_calls = [
                    ToolUse(
                        id=tc.id,
                        name=tc.function.name,
                        input=_safe_json_loads(tc.function.arguments),
                    )
                    for tc in message.tool_calls
                ]

            return ChatResponse(
                id=response.id,
                model=response.model,
                content=content,
                tool_calls=tool_calls,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                finish_reason=_map_finish_reason(choice.finish_reason),
            )

        except OpenAIAuthError as e:
            raise AuthenticationError(str(e))
        except APIConnectionError as e:
            raise ConnectionError(str(e))
        except APIStatusError as e:
            if e.status_code == 429:
                raise RateLimitError(str(e))
            raise ProviderAPIError(e.status_code, str(e))

    async def chat_stream(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion."""
        client = self._get_client(api_key)

        try:
            kwargs = self._build_request_kwargs(request)
            kwargs["stream"] = True
            logger.debug(
                f"{self.name} stream request: model={request.model}, params={list(kwargs.keys())}"
            )

            stream = await client.chat.completions.create(**kwargs)

            first_chunk = True
            current_tool_calls: dict[int, dict] = {}
            input_tokens = 0
            output_tokens = 0

            async for chunk in stream:
                if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
                    # Usage-only chunk (when stream_options.include_usage is true)
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
                    continue

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if first_chunk:
                    yield StreamChunk(type="start", model=chunk.model)
                    first_chunk = False

                # Handle content delta
                if delta.content:
                    yield StreamChunk(type="delta", content=delta.content)

                # Handle tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index

                        if idx not in current_tool_calls:
                            # New tool call
                            current_tool_calls[idx] = {
                                "id": tc.id,
                                "name": tc.function.name if tc.function else None,
                                "arguments": "",
                            }
                            yield StreamChunk(
                                type="tool_call_start",
                                tool_call=StreamToolCall(
                                    index=idx,
                                    id=tc.id,
                                    name=tc.function.name if tc.function else None,
                                ),
                            )

                        # Accumulate arguments
                        if tc.function and tc.function.arguments:
                            current_tool_calls[idx]["arguments"] += (
                                tc.function.arguments
                            )
                            yield StreamChunk(
                                type="tool_call_delta",
                                tool_call=StreamToolCall(
                                    index=idx, arguments_delta=tc.function.arguments
                                ),
                            )

                # Handle finish
                if choice.finish_reason:
                    yield StreamChunk(
                        type="stop",
                        finish_reason=_map_finish_reason(choice.finish_reason),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

        except OpenAIAuthError as e:
            raise AuthenticationError(str(e))
        except APIConnectionError as e:
            raise ConnectionError(str(e))
        except APIStatusError as e:
            if e.status_code == 429:
                raise RateLimitError(str(e))
            raise ProviderAPIError(e.status_code, str(e))

    def list_models(self) -> list[ModelInfo]:
        """Return available models."""
        return [
            ModelInfo(id=model, owned_by=self.name)
            for model in self._get_allowed_models()
        ]
