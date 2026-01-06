from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    port: int = 6600
    debug: bool = False

    # Logging settings
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_dir: str = "logs"

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
