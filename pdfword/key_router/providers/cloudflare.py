from __future__ import annotations

from .base import ProviderAdapter, ProviderDefinition
from ..capabilities import ProviderCapabilities


class CloudflareWorkersAIAdapter(ProviderAdapter):
    pass


CLOUDFLARE_DEFINITION = ProviderDefinition(
    provider="cloudflare_workers_ai",
    display_name="Cloudflare Workers AI",
    adapter_type="cloudflare_workers_ai",
    base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
    auth_header="Authorization",
    models_endpoint="",
    chat_completions_endpoint="/{model}",
    timeout_seconds=180,
    capabilities=ProviderCapabilities(
        supports_text_input=True,
        supports_image_input=True,
        supports_text_output=True,
        supports_usage_reporting=False,
        supports_model_listing=False,
        supports_rate_limit_headers=True,
    ),
    usage_parser="cloudflare",
    implementation_status="configuration-ready",
)
