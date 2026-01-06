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

    # Provider settings
    default_provider: str = "claude"

    # Claude settings
    claude_base_url: str | None = None
    claude_default_model: str = "claude-sonnet-4-20250514"
    claude_allowed_models: list[str] = [
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    model_config = {
        "env_file": ".env",
        "env_prefix": "",
        "env_nested_delimiter": "__",
    }


settings = Settings()
