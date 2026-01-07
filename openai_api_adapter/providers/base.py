from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from openai_api_adapter.models.common import ChatRequest, ChatResponse, ModelInfo, StreamChunk


class Provider(ABC):
    """Abstract base class for AI providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'claude', 'openai')."""
        pass

    def normalize_model_name(self, model_name: str) -> str:
        return model_name

    @abstractmethod
    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        """
        Non-streaming chat completion.

        Args:
            request: Chat request with messages and parameters.
            api_key: API key for authentication.

        Returns:
            ChatResponse with the model's response.
        """
        pass

    @abstractmethod
    def chat_stream(
        self, request: ChatRequest, api_key: str
    ) -> AsyncIterator[StreamChunk]:
        """
        Streaming chat completion.

        Args:
            request: Chat request with messages and parameters.
            api_key: API key for authentication.

        Yields:
            StreamChunk objects with type 'start', 'delta', or 'stop'.
        """
        pass

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """
        Return available models for this provider.

        Returns:
            List of ModelInfo objects.
        """
        pass
