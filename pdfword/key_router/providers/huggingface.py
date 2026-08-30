from __future__ import annotations

from .base import ProviderAdapter, ProviderDefinition
from ..capabilities import ProviderCapabilities


class HuggingFaceAdapter(ProviderAdapter):
    pass


HUGGINGFACE_DEFINITION = ProviderDefinition(
    provider="huggingface",
    display_name="Hugging Face Inference Providers",
    adapter_type="huggingface",
    base_url="https://router.huggingface.co",
    auth_header="Authorization",
    models_endpoint="/models",
    chat_completions_endpoint="/v1/chat/completions",
    timeout_seconds=180,
    capabilities=ProviderCapabilities(
        supports_text_input=True,
        supports_image_input=True,
        supports_text_output=True,
        supports_streaming=True,
        supports_usage_reporting=True,
        supports_model_listing=True,
        supports_rate_limit_headers=True,
    ),
    usage_parser="huggingface",
    implementation_status="configuration-ready",
)
