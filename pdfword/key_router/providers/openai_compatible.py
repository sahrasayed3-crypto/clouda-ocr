from __future__ import annotations

from .base import (
    ProviderAdapter,
    ProviderDefinition,
    ProviderRequest,
    ProviderResponse,
    RateLimitHeaderMapping,
    UnifiedProviderRequest,
)
from ..capabilities import ProviderCapabilities


class OpenAICompatibleAdapter(ProviderAdapter):
    def build_request(
        self, request: UnifiedProviderRequest, *, secret_value: str
    ) -> ProviderRequest:
        self.validate_configuration(secret_value=secret_value, model=request.model)
        messages: list[dict] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        content: str | list[dict] = request.user_text
        if request.image_b64 or request.image_url:
            url = request.image_url or (
                f"data:{request.image_mime};base64,{request.image_b64}"
            )
            content = [
                {"type": "text", "text": request.user_text},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        messages.append({"role": "user", "content": content})
        body: dict = {"model": request.model, "messages": messages}
        if request.max_output_tokens:
            body["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.stream:
            body["stream"] = True
        if request.structured_json:
            body["response_format"] = (
                {"type": "json_schema", "json_schema": request.json_schema}
                if request.json_schema
                else {"type": "json_object"}
            )
        auth_value = (
            f"Bearer {secret_value}"
            if self.definition.auth_header.lower() == "authorization"
            else secret_value
        )
        return ProviderRequest(
            method="POST",
            url=self.definition.chat_completions_endpoint,
            headers={
                self.definition.auth_header: auth_value,
                "Content-Type": "application/json",
            },
            json_body=body,
            timeout_seconds=self.definition.timeout_seconds,
        )

    def parse_response(self, response: ProviderResponse) -> str:
        if response.status_code >= 400:
            raise ValueError("Provider returned a non-success response")
        body = response.json_body
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Malformed provider response") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty provider response")
        return content

    def extract_usage(self, response: ProviderResponse) -> dict[str, float | int]:
        body = response.json_body if isinstance(response.json_body, dict) else {}
        raw_usage = body.get("usage")
        usage: dict = raw_usage if isinstance(raw_usage, dict) else {}
        return {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "cost": float(usage.get("cost") or 0.0),
        }


def openai_compatible_definition(
    *,
    provider: str,
    display_name: str,
    base_url: str,
    auth_header: str = "Authorization",
    timeout_seconds: int = 180,
    supports_model_listing: bool = True,
    supports_price_listing: bool = False,
    supports_rate_limit_headers: bool = True,
    implementation_status: str = "mock-tested; not live-tested",
) -> ProviderDefinition:
    base = base_url.rstrip("/")
    return ProviderDefinition(
        provider=provider,
        display_name=display_name,
        adapter_type="openai_compatible",
        base_url=base,
        auth_header=auth_header,
        models_endpoint=f"{base}/models",
        chat_completions_endpoint=f"{base}/chat/completions",
        timeout_seconds=timeout_seconds,
        capabilities=ProviderCapabilities(
            supports_text_input=True,
            supports_image_input=True,
            supports_text_output=True,
            supports_structured_output=True,
            supports_streaming=True,
            supports_usage_reporting=True,
            supports_model_listing=supports_model_listing,
            supports_price_listing=supports_price_listing,
            supports_rate_limit_headers=supports_rate_limit_headers,
        ),
        rate_limit_headers=RateLimitHeaderMapping(),
        implementation_status=implementation_status,
    )
