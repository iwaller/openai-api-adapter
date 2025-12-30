from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.openai import OpenAIChatRequest
from app.utils.converter import convert_common_to_openai, convert_openai_to_common
from app.utils.routing import get_provider_for_model
from app.utils.streaming import stream_generator

router = APIRouter()


def extract_api_key(authorization: str) -> str:
    """Extract API key from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "authentication_error", "message": "Invalid Authorization header format"}},
        )
    return authorization.removeprefix("Bearer ").strip()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatRequest,
    authorization: str = Header(..., alias="Authorization"),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports model prefix routing like OpenRouter:
    - "claude/claude-3-5-sonnet" -> routes to Claude provider
    - "openai/gpt-4" -> routes to OpenAI provider (when implemented)
    - "claude-3-5-sonnet" -> uses default provider
    """
    api_key = extract_api_key(authorization)

    # Get provider and actual model name from model string
    provider, model_name = get_provider_for_model(request.model)

    # Convert OpenAI request to common format
    common_request = convert_openai_to_common(request, model_name)

    if request.stream:
        # Streaming response
        return StreamingResponse(
            stream_generator(provider, common_request, api_key),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming response
        response = await provider.chat(common_request, api_key)
        return convert_common_to_openai(response)
