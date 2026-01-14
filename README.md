# OpenAI API Adapter

OpenAI-compatible API adapter for multiple AI providers. Currently supports Claude (Anthropic) and OpenAI-compatible APIs (like Aiberm) with a design that allows easy extension to other providers.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI API
- **Model prefix routing** - Route requests to different providers using model prefixes (e.g., `claude/claude-opus-4-5`)
- **Streaming support** - Full SSE streaming support
- **Multi-modal** - Support for text and images
- **Tool/Function calling** - OpenAI-style function calling (note: `strict` parameter is ignored)
- **Extended thinking** - Support for Claude's extended thinking mode
- **Prompt caching** - Automatic prompt caching for system prompts, tools, and conversation history
- **Extensible architecture** - Easy to add new AI providers
- **Custom base URL** - Support custom API endpoints for Claude

## API Behavior Notes

Based on Claude's official documentation:

| Feature | Behavior |
|---------|----------|
| Audio input | Not supported, automatically stripped |
| Function calling `strict` | Ignored (no guaranteed schema conformance) |
| System messages | Hoisted and concatenated to beginning |
| Extended thinking | Enabled via special model names (e.g., `claude-4.5-opus-high-thinking`) |

## Quick Start

### Using Docker

```bash
docker run -d -p 6600:6600 iwaller/openai-api-adapter:latest
```

### Using Docker Compose

```bash
# Download compose file
curl -O https://raw.githubusercontent.com/iwaller/openai-api-adapter/main/compose.prod.yaml

# Start service
docker compose -f compose.prod.yaml up -d
```

### From Source

```bash
# Clone repository
git clone https://github.com/iwaller/openai-api-adapter.git
cd openai-api-adapter

# Install dependencies (requires uv)
uv sync

# Run server
uv run uvicorn openai_api_adapter.main:app --port 6600
```

## Configuration

Configuration via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `6600` | Server port |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_DIR` | `logs` | Log directory |
| `LOG_FULL_CONTENT` | `true` | Log full message content |
| `LOG_MAX_CONTENT_LENGTH` | `10000` | Max chars to log per message (0 = unlimited) |
| `DEFAULT_PROVIDER` | `claude` | Default AI provider |
| `DEFAULT_MAX_TOKENS` | `65536` | Default max_tokens for requests |
| `CLAUDE_BASE_URL` | `None` | Custom Claude API endpoint |
| `CLAUDE_DEFAULT_MODEL` | `claude-opus-4-5` | Default model |
| `CLAUDE_BUDGET_TOKENS` | `8000` | Budget tokens for thinking mode |
| `CLAUDE_ALLOWED_MODELS` | See below | Allowed model list |
| `THINKING_CACHE_TTL` | `3600` | Thinking cache TTL in seconds |
| `THINKING_CACHE_MAXSIZE` | `10000` | Max cached thinking entries |
| `OVERRIDE_USAGE` | `false` | Override reported token usage |
| `OVERRIDE_PROMPT_TOKENS` | `0` | Fixed prompt tokens (when override enabled) |
| `OVERRIDE_COMPLETION_TOKENS` | `0` | Fixed completion tokens (when override enabled) |
| `AIBERM_BASE_URL` | `None` | Aiberm API endpoint (required for aiberm provider) |
| `AIBERM_DEFAULT_MODEL` | `gpt-4o` | Default Aiberm model |
| `AIBERM_ALLOWED_MODELS` | See below | Allowed Aiberm model list |

### Claude Allowed Models

```
claude-4.5-opus-high-thinking
claude-4.5-opus
claude-opus-4-5
claude-sonnet-4-5
claude-haiku-4-5
```

### Aiberm Allowed Models

```
gpt-4o
gpt-4o-mini
gpt-4-turbo
gpt-4
gpt-3.5-turbo
```

### Extended Thinking Mode

Use `claude-4.5-opus-high-thinking` as the model name to enable extended thinking:

```bash
curl http://localhost:6600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "claude-4.5-opus-high-thinking",
    "messages": [{"role": "user", "content": "Solve this complex problem..."}],
    "stream": true
  }'
```

When thinking mode is enabled:
- Temperature is forced to 1.0
- `top_p` is clamped to 0.95-1.0 range
- `tool_choice` only supports "auto" or "none"
- Thinking blocks are cached for tool use continuity

## API Endpoints

### Chat Completions

```bash
POST /v1/chat/completions
```

**Request:**
```bash
curl http://localhost:6600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "claude-opus-4-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Streaming:**
```bash
curl http://localhost:6600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "claude-opus-4-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Model Prefix Routing

Route to specific providers using prefixes:

```bash
# Explicit Claude routing
"model": "claude/claude-opus-4-5"

# Explicit Aiberm routing
"model": "aiberm/gpt-4o"

# Default provider (no prefix)
"model": "claude-opus-4-5"
```

### Aiberm Provider

Aiberm is an OpenAI-compatible API provider that doesn't support the `max_completion_tokens` parameter. This adapter automatically filters out unsupported parameters.

```bash
curl http://localhost:6600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "aiberm/gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_completion_tokens": 1000
  }'
```

The `max_completion_tokens` parameter will be automatically converted to `max_tokens` before forwarding to the Aiberm API.

### List Models

```bash
GET /v1/models
```

```bash
curl http://localhost:6600/v1/models
```

### Image Support

```bash
curl http://localhost:6600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "claude-opus-4-5",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
  }'
```

### Function Calling

```bash
curl http://localhost:6600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "claude-opus-4-5",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

## Project Structure

```
openai-api-adapter/
├── openai_api_adapter/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Pydantic settings
│   ├── exceptions.py        # Custom exceptions
│   ├── providers/
│   │   ├── base.py          # Provider abstract class
│   │   ├── openai_base.py   # OpenAI-compatible provider base class
│   │   ├── registry.py      # Provider registry
│   │   ├── claude.py        # Claude implementation
│   │   └── aiberm.py        # Aiberm implementation (OpenAI-compatible)
│   ├── models/
│   │   ├── common.py        # Internal models
│   │   └── openai.py        # OpenAI-compatible models
│   ├── routes/
│   │   ├── chat.py          # Chat completions endpoint
│   │   └── models.py        # Models endpoint
│   └── utils/
│       ├── converter.py     # Format conversion
│       ├── routing.py       # Model prefix routing
│       ├── streaming.py     # SSE streaming
│       ├── logger.py        # Logging utilities
│       └── thinking_cache.py # Thinking block cache
├── pyproject.toml
├── Dockerfile
├── compose.yaml
├── compose.prod.yaml
└── .env.example
```

## Adding New Providers

### Adding OpenAI-Compatible Providers (Recommended)

For providers that use OpenAI-compatible APIs, extend `OpenAIBaseProvider`:

1. Add configuration in `openai_api_adapter/config.py`:

```python
# MyProvider settings
myprovider_base_url: str | None = None
myprovider_default_model: str = "default-model"
myprovider_allowed_models: list[str] = ["model-1", "model-2"]
```

2. Create provider in `openai_api_adapter/providers/myprovider.py`:

```python
from openai_api_adapter.config import settings
from openai_api_adapter.providers.openai_base import OpenAIBaseProvider

class MyProvider(OpenAIBaseProvider):
    @property
    def name(self) -> str:
        return "myprovider"

    def _get_base_url(self) -> str | None:
        return settings.myprovider_base_url

    def _get_allowed_models(self) -> list[str]:
        return settings.myprovider_allowed_models

    def _get_default_model(self) -> str:
        return settings.myprovider_default_model

    def _filter_request_kwargs(self, kwargs: dict) -> dict:
        """Optional: Filter or modify request parameters."""
        # Example: Remove unsupported parameters
        kwargs.pop("unsupported_param", None)
        # Example: Add custom parameters
        kwargs["custom_param"] = "value"
        return kwargs
```

3. Register in `openai_api_adapter/main.py`:

```python
from openai_api_adapter.providers.myprovider import MyProvider

ProviderRegistry.register(MyProvider())
```

4. Use with model prefix:

```bash
"model": "myprovider/model-name"
```

### Adding Custom Providers

For providers with non-OpenAI APIs (like Claude), extend `Provider` directly:

```python
from openai_api_adapter.providers.base import Provider

class MyProvider(Provider):
    @property
    def name(self) -> str:
        return "myprovider"

    async def chat(self, request, api_key) -> ChatResponse:
        # Implementation
        pass

    async def chat_stream(self, request, api_key):
        # Implementation
        pass

    def list_models(self) -> list[ModelInfo]:
        # Implementation
        pass
```

## Development

```bash
# Install dependencies
uv sync

# Run in development mode
uv run uvicorn openai_api_adapter.main:app --reload --port 6600

# Run tests
uv run pytest
```

## License

MIT License

## Acknowledgments

- Inspired by [claude2openai](https://github.com/missuo/claude2openai)
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
