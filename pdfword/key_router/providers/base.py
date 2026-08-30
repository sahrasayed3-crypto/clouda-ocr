from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..capabilities import ProviderCapabilities
from ..errors import NormalizedProviderError, normalize_provider_error


@dataclass(frozen=True)
class RateLimitHeaderMapping:
    retry_after: str = "Retry-After"
    remaining_requests: str = ""
    remaining_tokens: str = ""
    reset_requests: str = ""
    reset_tokens: str = ""


@dataclass(frozen=True)
class ProviderDefinition:
    provider: str
    display_name: str
    adapter_type: str
    base_url: str
    auth_header: str
    models_endpoint: str
    chat_completions_endpoint: str
    timeout_seconds: int
    capabilities: ProviderCapabilities
    rate_limit_headers: RateLimitHeaderMapping = field(
        default_factory=RateLimitHeaderMapping
    )
    usage_parser: str = "openai_compatible"
    pricing_support: str = "static_or_configured"
    image_input_support: str = "model_capability_required"
    implementation_status: str = "configuration-ready"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedProviderRequest:
    model: str
    system_prompt: str = ""
    user_text: str = ""
    image_b64: str = ""
    image_url: str = ""
    image_mime: str = "image/png"
    max_output_tokens: int = 0
    temperature: float | None = None
    structured_json: bool = False
    json_schema: dict[str, Any] | None = None
    stream: bool = False


@dataclass(frozen=True)
class ProviderRequest:
    method: str
    url: str
    headers: dict[str, str]
    json_body: dict[str, Any]
    timeout_seconds: int


@dataclass(frozen=True)
class ProviderResponse:
    status_code: int
    headers: Mapping[str, str]
    json_body: Any


Transport = Callable[[ProviderRequest], ProviderResponse]


class ProviderAdapter:
    def __init__(self, definition: ProviderDefinition) -> None:
        self.definition = definition

    @property
    def provider(self) -> str:
        return self.definition.provider

    def health_check(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "status": "not_executed",
            "reason": "network health checks require explicit credentials and activation",
        }

    def validate_configuration(self, *, secret_value: str, model: str) -> None:
        if not secret_value or not model:
            raise ValueError("Provider adapter configuration is incomplete")

    def build_request(
        self, request: UnifiedProviderRequest, *, secret_value: str
    ) -> ProviderRequest:
        raise NotImplementedError

    def send_request(
        self, provider_request: ProviderRequest, *, transport: Transport | None
    ) -> ProviderResponse:
        if transport is None:
            raise RuntimeError("An explicit transport is required")
        return transport(provider_request)

    def parse_response(self, response: ProviderResponse) -> str:
        raise NotImplementedError

    def normalize_error(
        self,
        *,
        response: ProviderResponse | None = None,
        exception: BaseException | None = None,
        provider_account_id: str = "",
        provider_model_id: str = "",
        request_sent: bool = False,
        default_cooldown_seconds: int = 60,
    ) -> NormalizedProviderError:
        return normalize_provider_error(
            provider=self.provider,
            provider_account_id=provider_account_id,
            provider_model_id=provider_model_id,
            http_status=response.status_code if response else None,
            payload=response.json_body if response else None,
            headers=response.headers if response else None,
            exception=exception,
            request_sent=request_sent,
            default_cooldown_seconds=default_cooldown_seconds,
        )

    def extract_usage(self, response: ProviderResponse) -> dict[str, float | int]:
        return {}

    def extract_retry_after(self, response: ProviderResponse) -> str:
        for key, value in response.headers.items():
            if key.lower() == "retry-after":
                return str(value)
        return ""

    def supports_capability(self, capability: str) -> bool:
        mapping = {
            "text": self.definition.capabilities.supports_text_input,
            "vision": self.definition.capabilities.supports_image_input,
            "native_pdf": self.definition.capabilities.supports_pdf_input,
            "structured_json": self.definition.capabilities.supports_structured_output,
            "streaming": self.definition.capabilities.supports_streaming,
        }
        return bool(mapping.get(capability, False))

    def redact_request_for_logging(self, request: ProviderRequest) -> dict[str, Any]:
        headers = {
            key: (
                "[redacted]"
                if key.lower() in {"authorization", "x-api-key", "x-goog-api-key"}
                else value
            )
            for key, value in request.headers.items()
        }

        def clean(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: (
                        "[redacted]"
                        if any(
                            item in key.lower()
                            for item in ("token", "secret", "api_key")
                        )
                        else clean(item)
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, str) and "base64," in value:
                return value.split("base64,", 1)[0] + "base64,[redacted]"
            return value

        return {
            "method": request.method,
            "url": request.url.split("?", 1)[0],
            "headers": headers,
            "json_body": clean(request.json_body),
            "timeout_seconds": request.timeout_seconds,
        }
