import uuid

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.models.openai import OpenAIChatRequest
from app.utils.converter import convert_common_to_openai, convert_openai_to_common
from app.utils.logger import log_request, log_response
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
    request_id = str(uuid.uuid4())[:8]
    api_key = extract_api_key(authorization)

    # Get provider and model name (without provider prefix)
    # Model normalization is handled internally by the provider
    provider, model_name = get_provider_for_model(request.model)

    # Convert OpenAI request to common format
    common_request = convert_openai_to_common(request, model_name)

    # Log request
    log_request(
        request_id=request_id,
        model=model_name,
        messages=[{"role": m.role, "content": m.content} for m in request.messages],
        stream=request.stream,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        tools=request.tools,  # Log tools to diagnose tool calling issues
        tool_choice=request.tool_choice,
    )

    if request.stream:
        # Streaming response
        return StreamingResponse(
            stream_generator(provider, common_request, api_key, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-Id": request_id,
            },
        )
    else:
        # Non-streaming response
        try:
            response = await provider.chat(common_request, api_key)
            log_response(
                request_id=request_id,
                content=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                finish_reason=response.finish_reason,
            )
            openai_response = convert_common_to_openai(response)
            return JSONResponse(
                content=openai_response.model_dump(),
                headers={"X-Request-Id": request_id},
            )
        except Exception as e:
            log_response(request_id=request_id, error=str(e))
            raise
