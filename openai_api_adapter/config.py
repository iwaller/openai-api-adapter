from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    port: int = 6600
    debug: bool = False

    # Logging settings
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_dir: str = "logs"
    log_full_content: bool = True  # Set to False to redact message content
    log_max_content_length: int = 10000  # Max chars to log per message (0 = unlimited)

    # Token settings
    default_max_tokens: int = 65536  # Default max_tokens for Claude

    # Usage override (for special use cases)
    # When enabled, reported usage will use these fixed values instead of actual
    override_usage: bool = False
    override_prompt_tokens: int = 0
    override_completion_tokens: int = 0

    # Provider settings
    default_provider: str = "aiberm"
    enabled_providers: str = "aiberm"  # Comma-separated list of providers to enable, or "all" for all providers

    # Claude settings
    claude_base_url: str | None = None
    claude_default_model: str = "claude-opus-4-5"
    claude_budget_tokens: int = 8000  # Budget tokens for thinking mode

    # Thinking cache settings (for preserving thinking blocks during tool use)
    thinking_cache_ttl: int = 3600  # Cache TTL in seconds (default: 1 hour)
    thinking_cache_maxsize: int = 10000  # Max number of cached entries
    claude_allowed_models: list[str] = [
        # Claude 4
        "claude-4.5-opus-high-thinking",
        "claude-4.5-opus",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ]

    # Aiberm settings (OpenAI-compatible API that doesn't support max_completion_tokens)
    aiberm_base_url: str | None = None
    aiberm_default_model: str = "gpt-4o"
    aiberm_allowed_models: list[str] = [
        "openai/gpt-5.2-codex",
    ]

    model_config = {
        "env_file": ".env",
        "env_prefix": "",
        "env_nested_delimiter": "__",
    }

    def get_enabled_providers(self) -> list[str]:
        """Parse enabled_providers setting into a list of provider names.

        Returns:
            List of provider names to enable. If "all" is specified, returns all available providers.
        """
        # Import here to avoid circular dependency
        from openai_api_adapter.providers import AVAILABLE_PROVIDERS

        known_providers = set(AVAILABLE_PROVIDERS.keys())

        if self.enabled_providers.strip().lower() == "all":
            return list(known_providers)

        # Split by comma and strip whitespace
        providers = [p.strip() for p in self.enabled_providers.split(",") if p.strip()]

        # Validate provider names
        invalid = set(providers) - known_providers
        if invalid:
            import logging
            logging.warning(f"Unknown providers will be ignored: {invalid}")

        # Filter to valid providers only
        valid_providers = [p for p in providers if p in known_providers]

        # If no valid providers, default to the first available provider
        if not valid_providers:
            import logging
            default = list(known_providers)[0] if known_providers else "claude"
            logging.warning(f"No valid providers in ENABLED_PROVIDERS, defaulting to '{default}'")
            return [default]

        return valid_providers


settings = Settings()
