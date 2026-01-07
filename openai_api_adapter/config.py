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
    default_provider: str = "claude"

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

    model_config = {
        "env_file": ".env",
        "env_prefix": "",
        "env_nested_delimiter": "__",
    }


settings = Settings()
