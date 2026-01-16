from contextlib import asynccontextmanager

from anthropic import APIError as AnthropicAPIError
from anthropic import AuthenticationError as AnthropicAuthError
from fastapi import FastAPI, Request
from openai import APIError as OpenAIAPIError
from openai import AuthenticationError as OpenAIAuthError
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openai_api_adapter.config import settings
from openai_api_adapter.exceptions import ProviderError
from openai_api_adapter.providers import AVAILABLE_PROVIDERS
from openai_api_adapter.providers.registry import ProviderRegistry
from openai_api_adapter.routes import chat, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Get enabled providers from settings
    enabled_providers = settings.get_enabled_providers()

    # Register enabled providers on startup using centralized mapping
    # Note: enabled_providers is already validated by get_enabled_providers()
    for provider_name in enabled_providers:
        provider_class = AVAILABLE_PROVIDERS[provider_name]
        ProviderRegistry.register(
            provider_class(),
            default=(settings.default_provider == provider_name),
        )

    # Validate default provider is enabled
    if settings.default_provider not in enabled_providers:
        from openai_api_adapter.utils.logger import logger
        logger.warning(
            f"Default provider '{settings.default_provider}' is not enabled. "
            f"First enabled provider will be used as default."
        )

    # Ensure at least one provider is registered
    if not ProviderRegistry.list_providers():
        known_providers = ", ".join(AVAILABLE_PROVIDERS.keys())
        raise RuntimeError(
            f"No providers registered. ENABLED_PROVIDERS='{settings.enabled_providers}' "
            f"did not match any known providers ({known_providers})."
        )

    if settings.debug:
        print(f"Enabled providers: {enabled_providers}")
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
    from openai_api_adapter.utils.logger import logger
    logger.error(f"ProviderError: status={exc.status_code}, type={exc.error_type}, message={exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": exc.error_type,
                "message": exc.message,
                "code": None,
                "param": None,
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
                "code": "invalid_api_key",
                "param": None,
            }
        },
    )


@app.exception_handler(AnthropicAPIError)
async def anthropic_api_handler(request: Request, exc: AnthropicAPIError):
    """Map Anthropic SDK errors to OpenAI format."""
    from openai_api_adapter.utils.logger import logger
    status_code = getattr(exc, "status_code", 500)
    logger.error(f"AnthropicAPIError: status={status_code}, message={exc}")
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": "api_error",
                "message": str(exc),
                "code": None,
                "param": None,
            }
        },
    )


@app.exception_handler(OpenAIAuthError)
async def openai_auth_handler(request: Request, exc: OpenAIAuthError):
    """Map OpenAI SDK auth errors to OpenAI format."""
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "type": "authentication_error",
                "message": str(exc),
                "code": "invalid_api_key",
                "param": None,
            }
        },
    )


@app.exception_handler(OpenAIAPIError)
async def openai_api_handler(request: Request, exc: OpenAIAPIError):
    """Map OpenAI SDK errors to OpenAI format."""
    from openai_api_adapter.utils.logger import logger
    status_code = getattr(exc, "status_code", 500)
    logger.error(f"OpenAIAPIError: status={status_code}, message={exc}")
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": "api_error",
                "message": str(exc),
                "code": None,
                "param": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Convert FastAPI validation errors to OpenAI format."""
    from openai_api_adapter.utils.logger import logger
    errors = exc.errors()
    logger.error(f"RequestValidationError: {errors}")
    # Get the first error for the message
    first_error = errors[0] if errors else {}
    loc = first_error.get("loc", [])
    param = ".".join(str(x) for x in loc) if loc else None
    msg = first_error.get("msg", "Validation error")

    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "invalid_request_error",
                "message": f"{param}: {msg}" if param else msg,
                "code": "invalid_value",
                "param": param,
            }
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Catch-all error handler with logging."""
    from openai_api_adapter.utils.logger import logger
    import traceback
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "server_error",
                "message": str(exc),
                "code": None,
                "param": None,
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
        "openai_api_adapter.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
