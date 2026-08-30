import os
import time
from dataclasses import dataclass
from typing import Protocol

import requests

from .constants import MODEL_ACCURATE_PRIMARY, MODEL_FAST
from .openrouter_client import (
    openrouter_chat,
    openrouter_chat_text,
    openrouter_chat_with_image,
    resolve_models,
)


class AIProviderClient(Protocol):
    name: str

    def resolve_models(
        self, default_fast: str, default_accurate: str
    ) -> tuple[str, str, bool]: ...

    def chat_with_image(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        image_b64: str,
        max_tokens: int,
        image_mime: str = "image/png",
        temperature: float = 0.0,
    ) -> str: ...

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        image_b64: str,
        max_tokens: int,
        image_mime: str = "image/png",
        temperature: float = 0.0,
    ) -> str: ...

    def chat_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str: ...


@dataclass
class OpenRouterClient:
    api_key: str
    name: str = "openrouter"

    def resolve_models(
        self, default_fast: str, default_accurate: str
    ) -> tuple[str, str, bool]:
        try:
            return resolve_models()
        except Exception:
            return default_fast, default_accurate, True

    def chat_with_image(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        image_b64: str,
        max_tokens: int,
        image_mime: str = "image/png",
        temperature: float = 0.0,
    ) -> str:
        return openrouter_chat_with_image(
            api_key=self.api_key,
            model=model,
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64=image_b64,
            max_tokens=max_tokens,
            image_mime=image_mime,
            temperature=temperature,
        )

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        image_b64: str,
        max_tokens: int,
        image_mime: str = "image/png",
        temperature: float = 0.0,
    ) -> str:
        return openrouter_chat(
            api_key=self.api_key,
            model=model,
            system_prompt=system_prompt,
            image_b64=image_b64,
            max_tokens=max_tokens,
            image_mime=image_mime,
            temperature=temperature,
        )

    def chat_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        return openrouter_chat_text(
            api_key=self.api_key,
            model=model,
            system_prompt=system_prompt,
            user_text=user_text,
            max_tokens=max_tokens,
            temperature=temperature,
        )


@dataclass
class OpenAICompatibleProviderClient:
    name: str
    api_key: str
    api_url: str
    fast_model: str = MODEL_FAST
    accurate_model: str = MODEL_ACCURATE_PRIMARY
    timeout_sec: int = 180

    def resolve_models(
        self, default_fast: str, default_accurate: str
    ) -> tuple[str, str, bool]:
        fast = self.fast_model or default_fast
        accurate = self.accurate_model or default_accurate
        return fast, accurate, accurate != default_accurate

    def _post(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_sec,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"Transient HTTP {resp.status_code}", response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_err = exc
                if attempt < 3:
                    time.sleep(1.1 * attempt)
                    continue
                break
        raise RuntimeError(f"{self.name} request failed: {last_err}") from last_err

    def _extract_text(self, data: dict) -> str:
        if not isinstance(data, dict):
            raise RuntimeError(f"{self.name} invalid response payload")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.name} returned no choices")
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str) and part.strip():
                    chunks.append(part.strip())
                elif (
                    isinstance(part, dict)
                    and isinstance(part.get("text"), str)
                    and part["text"].strip()
                ):
                    chunks.append(part["text"].strip())
            if chunks:
                return "\n".join(chunks).strip()
        raise RuntimeError(f"{self.name} response content was empty")

    def chat_with_image(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        image_b64: str,
        max_tokens: int,
        image_mime: str = "image/png",
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_mime};base64,{image_b64}"
                            },
                        },
                    ],
                },
            ],
        }
        return self._extract_text(self._post(payload))

    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        image_b64: str,
        max_tokens: int,
        image_mime: str = "image/png",
        temperature: float = 0.0,
    ) -> str:
        return self.chat_with_image(
            model=model,
            system_prompt=system_prompt,
            user_text="Analyze this Arabic PDF page image according to the system instruction.",
            image_b64=image_b64,
            max_tokens=max_tokens,
            image_mime=image_mime,
            temperature=temperature,
        )

    def chat_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        }
        return self._extract_text(self._post(payload))


def get_provider_client(provider: str, primary_api_key: str) -> AIProviderClient:
    name = (provider or "openrouter").strip().lower()
    if name == "openrouter":
        return OpenRouterClient(api_key=primary_api_key)
    if name == "together":
        return OpenAICompatibleProviderClient(
            name="together",
            api_key=os.getenv("TOGETHER_API_KEY", "").strip(),
            api_url=os.getenv(
                "TOGETHER_API_URL", "https://api.together.xyz/v1/chat/completions"
            ).strip(),
            fast_model=os.getenv("TOGETHER_FAST_MODEL", MODEL_FAST).strip(),
            accurate_model=os.getenv(
                "TOGETHER_ACCURATE_MODEL", MODEL_ACCURATE_PRIMARY
            ).strip(),
        )
    if name == "fireworks":
        return OpenAICompatibleProviderClient(
            name="fireworks",
            api_key=os.getenv("FIREWORKS_API_KEY", "").strip(),
            api_url=os.getenv(
                "FIREWORKS_API_URL",
                "https://api.fireworks.ai/inference/v1/chat/completions",
            ).strip(),
            fast_model=os.getenv("FIREWORKS_FAST_MODEL", MODEL_FAST).strip(),
            accurate_model=os.getenv(
                "FIREWORKS_ACCURATE_MODEL", MODEL_ACCURATE_PRIMARY
            ).strip(),
        )
    return OpenRouterClient(api_key=primary_api_key)
