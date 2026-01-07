from openai_api_adapter.config import settings
from openai_api_adapter.exceptions import InvalidRequestError
from openai_api_adapter.providers.base import Provider
from openai_api_adapter.providers.registry import ProviderRegistry


def parse_model_with_prefix(model: str) -> tuple[str, str]:
    """
    Parse model string to extract provider prefix and actual model name.

    Supports OpenRouter-style prefixes like:
    - "claude/claude-3-5-sonnet" -> ("claude", "claude-3-5-sonnet")
    - "openai/gpt-4" -> ("openai", "gpt-4")
    - "claude-3-5-sonnet" -> (default_provider, "claude-3-5-sonnet")

    Args:
        model: Model string, optionally with provider prefix.

    Returns:
        Tuple of (provider_name, model_name).
    """
    if "/" in model:
        parts = model.split("/", 1)
        provider_prefix = parts[0].lower()
        model_name = parts[1]
        return provider_prefix, model_name
    else:
        # No prefix, use default provider
        return settings.default_provider, model


def get_provider_for_model(model: str) -> tuple[Provider, str]:
    """
    Get the appropriate provider and actual model name for a given model string.

    Note: Model name normalization (aliases, thinking mode) is handled by
    the provider internally. This function returns the original model name
    (without provider prefix) to preserve information needed for features
    like thinking mode detection.

    Args:
        model: Model string, optionally with provider prefix.

    Returns:
        Tuple of (Provider instance, model name without prefix).

    Raises:
        InvalidRequestError: If provider is not found.
    """
    provider_name, model_name = parse_model_with_prefix(model)

    try:
        provider = ProviderRegistry.get(provider_name)
    except KeyError:
        available = ProviderRegistry.list_providers()
        raise InvalidRequestError(
            f"Provider '{provider_name}' not found. Available providers: {', '.join(available)}"
        )

    return provider, model_name
