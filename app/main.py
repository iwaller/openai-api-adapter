from contextlib import asynccontextmanager

from anthropic import APIError as AnthropicAPIError
from anthropic import AuthenticationError as AnthropicAuthError
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.exceptions import ProviderError
from app.providers.claude import ClaudeProvider
from app.providers.registry import ProviderRegistry
from app.routes import chat, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Register providers on startup
    ProviderRegistry.register(
        ClaudeProvider(),
        default=(settings.default_provider == "claude"),
    )

    if settings.debug:
        print(f"Registered providers: {ProviderRegistry.list_providers()}")
        print(f"Default provider: {settings.default_provider}")

    yield

    # Cleanup on shutdown
    ProviderRegistry.clear()


app = FastAPI(
    title="OpenAI API Adapter",
    description="OpenAI-compatible API adapter for multiple AI providers",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    """Convert provider errors to OpenAI error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(AnthropicAuthError)
async def anthropic_auth_handler(request: Request, exc: AnthropicAuthError):
    """Map Anthropic SDK auth errors to OpenAI format."""
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "type": "authentication_error",
                "message": str(exc),
            }
        },
    )


@app.exception_handler(AnthropicAPIError)
async def anthropic_api_handler(request: Request, exc: AnthropicAPIError):
    """Map Anthropic SDK errors to OpenAI format."""
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={
            "error": {
                "type": "api_error",
                "message": str(exc),
            }
        },
    )


# Routes


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "message": "OpenAI API Adapter",
        "version": "1.0.0",
        "providers": ProviderRegistry.list_providers(),
    }


# Include routers
app.include_router(chat.router)
app.include_router(models.router)


def main():
    """Run the application with uvicorn."""
    import uvicorn

    print(f"Starting OpenAI API Adapter on port {settings.port}")
    print(f"Debug mode: {settings.debug}")
    print(f"Default provider: {settings.default_provider}")

    if settings.claude_base_url:
        print(f"Claude base URL: {settings.claude_base_url}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
