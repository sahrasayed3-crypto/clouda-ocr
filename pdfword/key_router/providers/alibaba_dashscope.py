from __future__ import annotations

from .base import ProviderAdapter, ProviderDefinition
from ..capabilities import ProviderCapabilities


class AlibabaDashScopeAdapter(ProviderAdapter):
    pass


ALIBABA_DASHSCOPE_DEFINITION = ProviderDefinition(
    provider="alibaba_dashscope",
    display_name="Alibaba Model Studio / DashScope",
    adapter_type="alibaba_dashscope",
    base_url="https://dashscope.aliyuncs.com",
    auth_header="Authorization",
    models_endpoint="/compatible-mode/v1/models",
    chat_completions_endpoint="/compatible-mode/v1/chat/completions",
    timeout_seconds=180,
    capabilities=ProviderCapabilities(
        supports_text_input=True,
        supports_image_input=True,
        supports_text_output=True,
        supports_structured_output=True,
        supports_usage_reporting=True,
        supports_model_listing=True,
        supports_rate_limit_headers=True,
    ),
    usage_parser="dashscope",
    implementation_status="configuration-ready",
)
