from app.config import settings
from app.exceptions import InvalidRequestError
from app.providers.base import Provider
from app.providers.registry import ProviderRegistry

# Model name aliases for compatibility with different naming conventions
# Maps short/common names to official Claude model IDs
MODEL_ALIASES: dict[str, str] = {
    # Claude 4 aliases
    "claude-4-sonnet": "claude-sonnet-4-20250514",
    "claude-4-opus": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    # Claude 3.5 aliases
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    # Claude 3 aliases
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


def normalize_model_name(model: str) -> str:
    """Normalize model name using aliases."""
    return MODEL_ALIASES.get(model, model)


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

    Args:
        model: Model string, optionally with provider prefix.

    Returns:
        Tuple of (Provider instance, actual model name).

    Raises:
        InvalidRequestError: If provider is not found.
    """
    provider_name, model_name = parse_model_with_prefix(model)

    # Normalize model name using aliases
    model_name = normalize_model_name(model_name)

    try:
        provider = ProviderRegistry.get(provider_name)
    except KeyError:
        available = ProviderRegistry.list_providers()
        raise InvalidRequestError(
            f"Provider '{provider_name}' not found. Available providers: {', '.join(available)}"
        )

    return provider, model_name
