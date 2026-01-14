"""Aiberm provider - OpenAI-compatible API that doesn't support max_completion_tokens."""

from openai_api_adapter.config import settings
from openai_api_adapter.providers.openai_base import OpenAIBaseProvider


class AibermProvider(OpenAIBaseProvider):
    """Aiberm provider using OpenAI SDK.

    This provider forwards requests to an OpenAI-compatible API,
    filtering out the max_completion_tokens parameter which is not supported.
    """

    @property
    def name(self) -> str:
        return "aiberm"

    def _get_base_url(self) -> str | None:
        return settings.aiberm_base_url

    def _get_allowed_models(self) -> list[str]:
        return settings.aiberm_allowed_models

    def _get_default_model(self) -> str:
        return settings.aiberm_default_model

    def _filter_request_kwargs(self, kwargs: dict) -> dict:
        """Filter out max_completion_tokens which is not supported by aiberm."""
        # Remove max_completion_tokens, keep only max_tokens
        kwargs.pop("max_completion_tokens", None)
        return kwargs
