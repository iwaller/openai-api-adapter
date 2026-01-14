from fastapi import APIRouter

from openai_api_adapter.models.openai import OpenAIModel, OpenAIModelsResponse
from openai_api_adapter.providers.registry import ProviderRegistry

router = APIRouter()


@router.get("/v1/models")
async def list_models() -> OpenAIModelsResponse:
    """
    List available models from all registered providers.

    Returns models with provider prefix (e.g., "claude/claude-3-5-sonnet")
    for explicit routing, and also without prefix for default provider.
    """
    models: list[OpenAIModel] = []

    default_provider = ProviderRegistry.get().name
    for provider_name in ProviderRegistry.list_providers():
        provider = ProviderRegistry.get(provider_name)

        for model_info in provider.list_models():
            # Add model with provider prefix (e.g., "claude/claude-3-5-sonnet")
            models.append(
                OpenAIModel(
                    id=f"{provider_name}/{model_info.id}",
                    object="model",
                    created=0,
                    owned_by=model_info.owned_by,
                )
            )

            # Only add without prefix for default provider convenience
            if provider_name == default_provider:
                models.append(
                    OpenAIModel(
                        id=model_info.id,
                        object="model",
                        created=0,
                        owned_by=model_info.owned_by,
                    )
                )

    return OpenAIModelsResponse(object="list", data=models)
