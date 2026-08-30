import time
import threading
import json
import logging
import uuid
import os
from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from .constants import (
    ACCURATE_FALLBACKS,
    MODEL_ACCURATE_PRIMARY,
    MODEL_FAST,
    OPENROUTER_API_URL,
    OPENROUTER_MODELS_API,
    REQUEST_TIMEOUT,
    VISION_MODEL_PRIORITY,
)
from .models import ModelEndpointError, OpenRouterEmptyContentError

logger = logging.getLogger(__name__)
_SESSION_LOCAL = threading.local()
_TELEMETRY_CALLBACK: ContextVar = ContextVar(
    "openrouter_telemetry_callback", default=None
)
MODEL_CACHE_PATH = Path("data") / "openrouter_models.json"


def _price_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_http_session() -> requests.Session:
    session = getattr(_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _SESSION_LOCAL.session = session
    return session


def _extract_text_from_response(data: dict) -> str:
    if not isinstance(data, dict):
        raise RuntimeError("استجابة OpenRouter غير صالحة.")

    choices = data.get("choices") or []
    if not choices:
        if data.get("error"):
            raise RuntimeError(f"OpenRouter error: {data.get('error')}")
        raise RuntimeError("لم يتم استلام نتائج من OpenRouter.")

    first_choice = choices[0] or {}
    message = first_choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        txt = content.strip()
        if txt:
            return txt

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                val = item.strip()
                if val:
                    parts.append(val)
                continue
            if not isinstance(item, dict):
                continue
            candidate = ""
            if isinstance(item.get("text"), str):
                candidate = item["text"]
            elif isinstance(item.get("content"), str):
                candidate = item["content"]
            val = candidate.strip()
            if val:
                parts.append(val)
        if parts:
            return "\n".join(parts).strip()

    if isinstance(first_choice.get("text"), str) and first_choice["text"].strip():
        return first_choice["text"].strip()

    if isinstance(message.get("refusal"), str) and message["refusal"].strip():
        raise RuntimeError(f"رفض النموذج الاستجابة: {message['refusal'].strip()}")

    raise OpenRouterEmptyContentError(
        "OpenRouter response did not include usable text content."
    )


def discover_openrouter_models(
    force: bool = False, cache_hours: int = 24
) -> list[dict]:
    if not force and MODEL_CACHE_PATH.is_file():
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            MODEL_CACHE_PATH.stat().st_mtime,
            tz=timezone.utc,
        )
        if age < timedelta(hours=max(1, cache_hours)):
            try:
                cached = json.loads(MODEL_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    return cached
            except (OSError, ValueError):
                pass
    try:
        resp = _get_http_session().get(OPENROUTER_MODELS_API, timeout=25)
        resp.raise_for_status()
        payload = resp.json()
        models = []
        for model in payload.get("data", []):
            if not isinstance(model, dict) or not model.get("id"):
                continue
            architecture = model.get("architecture") or {}
            modalities = architecture.get("input_modalities") or []
            modality = str(architecture.get("modality") or "")
            supports_vision = "image" in modalities or "image" in modality
            pricing = model.get("pricing") or {}
            models.append(
                {
                    "id": model["id"],
                    "name": model.get("name") or model["id"],
                    "supports_vision": supports_vision,
                    "input_modalities": modalities,
                    "output_modalities": architecture.get("output_modalities")
                    or ["text"],
                    "prompt_price": pricing.get("prompt"),
                    "completion_price": pricing.get("completion"),
                    "image_price": pricing.get("image"),
                    "context_length": model.get("context_length"),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_CACHE_PATH.write_text(
            json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return models
    except Exception:
        if MODEL_CACHE_PATH.is_file():
            try:
                cached = json.loads(MODEL_CACHE_PATH.read_text(encoding="utf-8"))
                return cached if isinstance(cached, list) else []
            except (OSError, ValueError):
                pass
        return []


def get_openrouter_model_ids() -> set[str]:
    return {model["id"] for model in discover_openrouter_models() if model.get("id")}


def verified_vision_model_ids(configured_models: list[str] | None = None) -> list[str]:
    catalog = {
        model["id"]: model
        for model in discover_openrouter_models()
        if model.get("id")
        and model.get("supports_vision")
        and not str(model["id"]).lower().startswith("deepseek/")
        and _price_or_none(model.get("prompt_price")) is not None
        and _price_or_none(model.get("completion_price")) is not None
        and _price_or_none(model.get("image_price")) is not None
    }
    requested = configured_models if configured_models else VISION_MODEL_PRIORITY
    verified: list[str] = []
    for model_id in dict.fromkeys(requested):
        if str(model_id).lower().startswith("deepseek/"):
            logger.warning(
                "Skipping Vision model=%s reason=deepseek_image_blocked", model_id
            )
        elif model_id not in catalog:
            logger.info(
                "Skipping Vision model=%s reason=missing_or_text_only_catalog_entry",
                model_id,
            )
        else:
            verified.append(model_id)
            logger.info(
                "Verified Vision model=%s reason=catalog_supports_vision", model_id
            )
    return verified


def estimate_model_cost(
    model_id: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    model = next(
        (item for item in discover_openrouter_models() if item.get("id") == model_id),
        None,
    )
    if not model:
        return None
    prompt_price = _price_or_none(model.get("prompt_price"))
    completion_price = _price_or_none(model.get("completion_price"))
    if prompt_price is None or completion_price is None:
        return None
    return (prompt_tokens * prompt_price) + (completion_tokens * completion_price)


def estimate_vision_request_cost(
    model_id: str,
    max_tokens: int,
    *,
    estimated_prompt_tokens: int = 12000,
) -> float | None:
    model = next(
        (
            item
            for item in discover_openrouter_models()
            if item.get("id") == model_id and item.get("supports_vision")
        ),
        None,
    )
    if not model:
        return None
    prompt_price = _price_or_none(model.get("prompt_price"))
    completion_price = _price_or_none(model.get("completion_price"))
    image_price = _price_or_none(model.get("image_price"))
    if prompt_price is None or completion_price is None or image_price is None:
        return None
    estimated = (
        max(1, int(estimated_prompt_tokens)) * prompt_price
        + max(1, int(max_tokens)) * completion_price
        + max(0.0, image_price)
    )
    return max(0.0, estimated * 1.25)


@contextmanager
def capture_openrouter_telemetry(callback):
    token = _TELEMETRY_CALLBACK.set(callback)
    try:
        yield
    finally:
        _TELEMETRY_CALLBACK.reset(token)


def _emit_telemetry(data: dict, payload: dict, elapsed: float) -> None:
    callback = _TELEMETRY_CALLBACK.get()
    if callback is None:
        return
    usage = data.get("usage") or {}
    raw_cost = usage.get("cost", data.get("cost"))
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    estimated_cost = estimate_model_cost(
        data.get("model") or payload.get("model") or "",
        prompt_tokens,
        completion_tokens,
    )
    cost_unknown = raw_cost in (None, "") and estimated_cost is None
    cost = float(raw_cost) if raw_cost not in (None, "") else float(estimated_cost or 0)
    callback(
        {
            "model": data.get("model") or payload.get("model") or "",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "cost_is_estimated": raw_cost in (None, ""),
            "cost_unknown": cost_unknown,
            "processing_time": elapsed,
            "request_id": data.get("id"),
        }
    )


def resolve_models() -> tuple[str, str, bool]:
    vision_models = verified_vision_model_ids()
    fast_model = vision_models[0] if vision_models else MODEL_FAST

    candidates = [MODEL_ACCURATE_PRIMARY] + ACCURATE_FALLBACKS
    accurate_model = fast_model
    for c in candidates:
        if c in vision_models:
            accurate_model = c
            break

    used_fallback = accurate_model != MODEL_ACCURATE_PRIMARY
    return fast_model, accurate_model, used_fallback


def _post_with_retries(payload: dict, headers: dict) -> dict:
    for attempt in range(1, 4):
        try:
            started = time.perf_counter()
            resp = _get_http_session().post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 404:
                text = resp.text.lower()
                model = payload.get("model", "")
                if "no endpoints found" in text:
                    raise ModelEndpointError(f"النموذج غير متاح حاليًا: {model}")

            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(
                    f"Transient HTTP {resp.status_code}", response=resp
                )

            resp.raise_for_status()
            data = resp.json()
            _emit_telemetry(data, payload, time.perf_counter() - started)
            return data
        except ModelEndpointError:
            raise
        except Exception as exc:
            if attempt < 3:
                time.sleep(1.2 * attempt)
                continue
            raise RuntimeError(
                f"فشل الاتصال مع OpenRouter بعد عدة محاولات: {exc}"
            ) from exc

    raise RuntimeError("فشل غير متوقع أثناء الاتصال بـ OpenRouter")


def openrouter_chat_with_image(
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_b64: str,
    max_tokens: int,
    image_mime: str = "image/png",
    temperature: float = 0.0,
) -> str:
    estimated_cost = estimate_vision_request_cost(model, max_tokens)
    if estimated_cost is None:
        raise ModelEndpointError(
            f"Vision model is unavailable or missing pricing metadata: {model}"
        )
    from .key_router.config import KeyRouterConfig
    from .key_router.enums import KeyRouterMode

    if KeyRouterConfig.from_env().mode is KeyRouterMode.ACTIVE:
        from .key_router.facade import dispatch_active_legacy_request

        return dispatch_active_legacy_request(
            model=model,
            request_id=uuid.uuid4().hex,
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64=image_b64,
            image_mime=image_mime,
            max_tokens=max_tokens,
            temperature=temperature,
            estimated_cost=estimated_cost,
        )
    try:
        from .key_router.facade import route_legacy_openrouter_request

        api_key = route_legacy_openrouter_request(
            api_key=api_key,
            model=model,
            request_id=uuid.uuid4().hex,
            requires_vision=True,
            max_tokens=max_tokens,
            estimated_cost=estimated_cost,
        )
    except Exception as exc:
        from .key_router.config import KeyRouterConfig
        from .key_router.enums import KeyRouterMode

        if KeyRouterConfig.from_env().mode in {
            KeyRouterMode.DISABLED,
            KeyRouterMode.SHADOW,
        }:
            pass
        else:
            raise ModelEndpointError(
                "Key Router rejected the OpenRouter request"
            ) from exc
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Arabic PDF to DOCX Smart Converter",
    }

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
                        "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
                    },
                ],
            },
        ],
    }

    last_err: Exception | None = None
    for attempt in range(1, 4):
        data = _post_with_retries(payload=payload, headers=headers)
        try:
            return _extract_text_from_response(data)
        except OpenRouterEmptyContentError as exc:
            last_err = exc
            if attempt < 3:
                time.sleep(0.7 * attempt)
                continue
            break
    raise RuntimeError(
        "OpenRouter returned an empty response multiple times for this page."
    ) from last_err


def openrouter_chat(
    api_key: str,
    model: str,
    system_prompt: str,
    image_b64: str,
    max_tokens: int,
    image_mime: str = "image/png",
    temperature: float = 0.0,
) -> str:
    return openrouter_chat_with_image(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_text="Analyze this Arabic PDF page image according to the system instruction.",
        image_b64=image_b64,
        image_mime=image_mime,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def openrouter_chat_text(
    api_key: str,
    model: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    estimated_cost = (
        estimate_model_cost(model, 0, max_tokens)
        if os.getenv("KEY_ROUTER_MODE", "disabled").strip().lower() != "disabled"
        else None
    )
    from .key_router.config import KeyRouterConfig
    from .key_router.enums import KeyRouterMode

    if KeyRouterConfig.from_env().mode is KeyRouterMode.ACTIVE:
        from .key_router.facade import dispatch_active_legacy_request

        return dispatch_active_legacy_request(
            model=model,
            request_id=uuid.uuid4().hex,
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64="",
            image_mime="",
            max_tokens=max_tokens,
            temperature=temperature,
            estimated_cost=estimated_cost,
        )
    try:
        from .key_router.facade import route_legacy_openrouter_request

        api_key = route_legacy_openrouter_request(
            api_key=api_key,
            model=model,
            request_id=uuid.uuid4().hex,
            requires_vision=False,
            max_tokens=max_tokens,
            estimated_cost=estimated_cost,
        )
    except Exception as exc:
        from .key_router.config import KeyRouterConfig
        from .key_router.enums import KeyRouterMode

        if KeyRouterConfig.from_env().mode in {
            KeyRouterMode.DISABLED,
            KeyRouterMode.SHADOW,
        }:
            pass
        else:
            raise ModelEndpointError(
                "Key Router rejected the OpenRouter request"
            ) from exc
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Arabic PDF to DOCX Smart Converter",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    last_err: Exception | None = None
    for attempt in range(1, 4):
        data = _post_with_retries(payload=payload, headers=headers)
        try:
            return _extract_text_from_response(data)
        except OpenRouterEmptyContentError as exc:
            last_err = exc
            if attempt < 3:
                time.sleep(0.7 * attempt)
                continue
            break
    raise RuntimeError(
        "OpenRouter returned an empty response multiple times for this text task."
    ) from last_err
