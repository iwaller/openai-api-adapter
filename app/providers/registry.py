from app.providers.base import Provider


class ProviderRegistry:
    """Registry for managing AI providers."""

    _providers: dict[str, Provider] = {}
    _default: str | None = None

    @classmethod
    def register(cls, provider: Provider, default: bool = False) -> None:
        """
        Register a provider.

        Args:
            provider: Provider instance to register.
            default: If True, set this as the default provider.
        """
        cls._providers[provider.name] = provider
        if default or cls._default is None:
            cls._default = provider.name

    @classmethod
    def get(cls, name: str | None = None) -> Provider:
        """
        Get a provider by name.

        Args:
            name: Provider name. If None, returns the default provider.

        Returns:
            The requested Provider instance.

        Raises:
            KeyError: If provider is not found.
        """
        name = name or cls._default
        if name is None:
            raise KeyError("No default provider registered")
        if name not in cls._providers:
            raise KeyError(f"Provider '{name}' not found")
        return cls._providers[name]

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return list of registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (for testing)."""
        cls._providers.clear()
        cls._default = None
