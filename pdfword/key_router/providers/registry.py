from __future__ import annotations

from .alibaba_dashscope import ALIBABA_DASHSCOPE_DEFINITION
from .base import ProviderAdapter, ProviderDefinition
from .cloudflare import CLOUDFLARE_DEFINITION
from .google_gemini import GOOGLE_GEMINI_DEFINITION, GoogleGeminiAdapter
from .huggingface import HUGGINGFACE_DEFINITION
from .openai_compatible import OpenAICompatibleAdapter, openai_compatible_definition

PROVIDER_REGISTRY: dict[str, ProviderDefinition] = {
    "openrouter": openai_compatible_definition(
        provider="openrouter",
        display_name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        supports_price_listing=True,
    ),
    "google_gemini": GOOGLE_GEMINI_DEFINITION,
    "alibaba_dashscope": ALIBABA_DASHSCOPE_DEFINITION,
    "requesty": openai_compatible_definition(
        provider="requesty",
        display_name="Requesty",
        base_url="https://router.requesty.ai/v1",
    ),
    "huggingface": HUGGINGFACE_DEFINITION,
    "nvidia_nim": openai_compatible_definition(
        provider="nvidia_nim",
        display_name="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com/v1",
    ),
    "sambanova": openai_compatible_definition(
        provider="sambanova",
        display_name="SambaNova Cloud",
        base_url="https://api.sambanova.ai/v1",
    ),
    "github_models": openai_compatible_definition(
        provider="github_models",
        display_name="GitHub Models",
        base_url="https://models.github.ai/inference",
        auth_header="Authorization",
    ),
    "cloudflare_workers_ai": CLOUDFLARE_DEFINITION,
    "siliconflow": openai_compatible_definition(
        provider="siliconflow",
        display_name="SiliconFlow",
        base_url="https://api.siliconflow.cn/v1",
    ),
    "deepinfra": openai_compatible_definition(
        provider="deepinfra",
        display_name="DeepInfra",
        base_url="https://api.deepinfra.com/v1/openai",
    ),
    "fireworks": openai_compatible_definition(
        provider="fireworks",
        display_name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
    ),
    "together": openai_compatible_definition(
        provider="together",
        display_name="Together AI",
        base_url="https://api.together.xyz/v1",
    ),
    "nebius": openai_compatible_definition(
        provider="nebius",
        display_name="Nebius AI Studio",
        base_url="https://api.studio.nebius.ai/v1",
    ),
    "replicate": openai_compatible_definition(
        provider="replicate",
        display_name="Replicate",
        base_url="https://api.replicate.com/v1",
        supports_model_listing=False,
        implementation_status="configuration-ready",
    ),
    "novita": openai_compatible_definition(
        provider="novita",
        display_name="Novita AI",
        base_url="https://api.novita.ai/v3/openai",
    ),
}


def get_provider_definition(provider: str) -> ProviderDefinition | None:
    return PROVIDER_REGISTRY.get(provider.strip().lower())


def list_provider_definitions() -> list[ProviderDefinition]:
    return [PROVIDER_REGISTRY[key] for key in sorted(PROVIDER_REGISTRY)]


def get_provider_adapter(provider: str) -> ProviderAdapter | None:
    definition = get_provider_definition(provider)
    if definition is None:
        return None
    if definition.implementation_status == "configuration-ready":
        return None
    if definition.adapter_type == "openai_compatible":
        return OpenAICompatibleAdapter(definition)
    if definition.adapter_type == "google_gemini":
        return GoogleGeminiAdapter(definition)
    return None
