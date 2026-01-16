from openai_api_adapter.providers.aiberm import AibermProvider
from openai_api_adapter.providers.base import Provider
from openai_api_adapter.providers.claude import ClaudeProvider
from openai_api_adapter.providers.registry import ProviderRegistry

# Centralized provider mapping for easy extensibility
# To add a new provider: import it above and add it to this dict
AVAILABLE_PROVIDERS = {
    "claude": ClaudeProvider,
    "aiberm": AibermProvider,
}

__all__ = ["Provider", "ClaudeProvider", "AibermProvider", "ProviderRegistry", "AVAILABLE_PROVIDERS"]
