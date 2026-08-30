from __future__ import annotations

from .base import (
    ProviderAdapter,
    ProviderDefinition,
    ProviderRequest,
    ProviderResponse,
    UnifiedProviderRequest,
)
from ..capabilities import ProviderCapabilities


class GoogleGeminiAdapter(ProviderAdapter):
    def build_request(
        self, request: UnifiedProviderRequest, *, secret_value: str
    ) -> ProviderRequest:
        self.validate_configuration(secret_value=secret_value, model=request.model)
        parts: list[dict] = []
        if request.user_text:
            parts.append({"text": request.user_text})
        if request.image_b64:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": request.image_mime,
                        "data": request.image_b64,
                    }
                }
            )
        elif request.image_url:
            parts.append(
                {
                    "fileData": {
                        "mimeType": request.image_mime,
                        "fileUri": request.image_url,
                    }
                }
            )
        body: dict = {"contents": [{"role": "user", "parts": parts}]}
        if request.system_prompt:
            body["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        generation: dict = {}
        if request.max_output_tokens:
            generation["maxOutputTokens"] = request.max_output_tokens
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.structured_json:
            generation["responseMimeType"] = "application/json"
            if request.json_schema:
                generation["responseSchema"] = request.json_schema
        if generation:
            body["generationConfig"] = generation
        url = self.definition.chat_completions_endpoint.format(model=request.model)
        if not url.startswith("http"):
            url = self.definition.base_url.rstrip("/") + "/" + url.lstrip("/")
        return ProviderRequest(
            method="POST",
            url=url,
            headers={
                "x-goog-api-key": secret_value,
                "Content-Type": "application/json",
            },
            json_body=body,
            timeout_seconds=self.definition.timeout_seconds,
        )

    def parse_response(self, response: ProviderResponse) -> str:
        if response.status_code >= 400:
            raise ValueError("Provider returned a non-success response")
        try:
            content = response.json_body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Malformed provider response") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty provider response")
        return content

    def extract_usage(self, response: ProviderResponse) -> dict[str, float | int]:
        body = response.json_body if isinstance(response.json_body, dict) else {}
        raw_usage = body.get("usageMetadata")
        usage: dict = raw_usage if isinstance(raw_usage, dict) else {}
        return {
            "input_tokens": int(usage.get("promptTokenCount") or 0),
            "output_tokens": int(usage.get("candidatesTokenCount") or 0),
            "cost": 0.0,
        }


GOOGLE_GEMINI_DEFINITION = ProviderDefinition(
    provider="google_gemini",
    display_name="Google AI Studio / Gemini API",
    adapter_type="google_gemini",
    base_url="https://generativelanguage.googleapis.com",
    auth_header="x-goog-api-key",
    models_endpoint="/v1beta/models",
    chat_completions_endpoint="/v1beta/models/{model}:generateContent",
    timeout_seconds=180,
    capabilities=ProviderCapabilities(
        supports_text_input=True,
        supports_image_input=True,
        supports_pdf_input=True,
        supports_text_output=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_usage_reporting=True,
        supports_model_listing=True,
        supports_rate_limit_headers=True,
    ),
    usage_parser="google_gemini",
    implementation_status="mock-tested; not live-tested",
)
