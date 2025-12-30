# OpenAI API Adapter

OpenAI-compatible API adapter for multiple AI providers. Currently supports Claude (Anthropic) with a design that allows easy extension to other providers.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI API
- **Model prefix routing** - Route requests to different providers using model prefixes (e.g., `claude/claude-3-5-sonnet`)
- **Streaming support** - Full SSE streaming support
- **Multi-modal** - Support for text and images
- **Tool/Function calling** - OpenAI-style function calling (note: `strict` parameter is ignored)
- **Extensible architecture** - Easy to add new AI providers
- **Custom base URL** - Support custom API endpoints for Claude

## API Behavior Notes

Based on Claude's official documentation:

| Feature | Behavior |
|---------|----------|
| Audio input | Not supported, automatically stripped |
| Function calling `strict` | Ignored (no guaranteed schema conformance) |
| System messages | Hoisted and concatenated to beginning |
| Prompt caching | Not supported via this adapter |

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
uv run uvicorn app.main:app --port 6600
```

## Configuration

Configuration via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `6600` | Server port |
| `DEBUG` | `false` | Enable debug mode |
| `DEFAULT_PROVIDER` | `claude` | Default AI provider |
| `CLAUDE_BASE_URL` | `None` | Custom Claude API endpoint |
| `CLAUDE_DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Default model |
| `CLAUDE_ALLOWED_MODELS` | See below | Allowed model list |

### Default Allowed Models

```
claude-sonnet-4-20250514
claude-3-5-haiku-20241022
claude-3-5-sonnet-20241022
claude-3-5-sonnet-20240620
claude-3-opus-20240229
claude-3-sonnet-20240229
claude-3-haiku-20240307
```

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
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Streaming:**
```bash
curl http://localhost:6600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Model Prefix Routing

Route to specific providers using prefixes:

```bash
# Explicit Claude routing
"model": "claude/claude-3-5-sonnet-20241022"

# Default provider (no prefix)
"model": "claude-3-5-sonnet-20241022"
```

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
    "model": "claude-3-5-sonnet-20241022",
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
    "model": "claude-3-5-sonnet-20241022",
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
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Pydantic settings
│   ├── exceptions.py        # Custom exceptions
│   ├── providers/
│   │   ├── base.py          # Provider abstract class
│   │   ├── registry.py      # Provider registry
│   │   └── claude.py        # Claude implementation
│   ├── models/
│   │   ├── common.py        # Internal models
│   │   └── openai.py        # OpenAI-compatible models
│   ├── routes/
│   │   ├── chat.py          # Chat completions endpoint
│   │   └── models.py        # Models endpoint
│   └── utils/
│       ├── converter.py     # Format conversion
│       ├── routing.py       # Model prefix routing
│       └── streaming.py     # SSE streaming
├── pyproject.toml
├── Dockerfile
├── compose.yaml
├── compose.prod.yaml
└── .env.example
```

## Adding New Providers

1. Create a new provider in `app/providers/`:

```python
from app.providers.base import Provider

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

2. Register in `app/main.py`:

```python
from app.providers.myprovider import MyProvider

ProviderRegistry.register(MyProvider())
```

3. Use with model prefix:

```bash
"model": "myprovider/model-name"
```

## Development

```bash
# Install dependencies
uv sync

# Run in development mode
uv run uvicorn app.main:app --reload --port 6600

# Run tests
uv run pytest
```

## License

MIT License

## Acknowledgments

- Inspired by [claude2openai](https://github.com/missuo/claude2openai)
- Built with [FastAPI](https://fastapi.tiangolo.com/) and [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python)
