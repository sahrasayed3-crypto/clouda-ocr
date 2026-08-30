from __future__ import annotations

from .providers.registry import list_provider_definitions


def provider_capability_snapshot() -> list[dict[str, object]]:
    snapshot = []
    for definition in list_provider_definitions():
        snapshot.append(
            {
                "provider": definition.provider,
                "display_name": definition.display_name,
                "adapter_type": definition.adapter_type,
                "implementation_status": definition.implementation_status,
                "supports_text_input": definition.capabilities.supports_text_input,
                "supports_image_input": definition.capabilities.supports_image_input,
                "supports_pdf_input": definition.capabilities.supports_pdf_input,
                "supports_usage_reporting": definition.capabilities.supports_usage_reporting,
                "supports_price_listing": definition.capabilities.supports_price_listing,
            }
        )
    return snapshot
