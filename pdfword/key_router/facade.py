from __future__ import annotations

from .config import KeyRouterConfig
from .enums import KeyRouterMode, Modality
from .exceptions import KeyRouterUnavailable
from .models import RequestContext
from .service import KeyRouter
from .providers.base import UnifiedProviderRequest
from .providers.http_transport import RequestsTransport


def route_legacy_openrouter_request(
    *,
    api_key: str,
    model: str,
    request_id: str,
    requires_vision: bool,
    max_tokens: int,
    estimated_cost: float | None,
) -> str:
    config = KeyRouterConfig.from_env()
    if config.mode is KeyRouterMode.DISABLED:
        return api_key
    if config.mode is KeyRouterMode.ACTIVE:
        raise KeyRouterUnavailable(
            "Active mode must use the reservation-aware provider dispatcher"
        )
    context = RequestContext(
        request_id=request_id,
        operation_id=request_id,
        task_type="legacy_openrouter",
        modality=Modality.IMAGE if requires_vision else Modality.TEXT,
        requested_model=model,
        allowed_providers=("openrouter",),
        free_only=config.free_only_default,
        estimated_output_tokens=max_tokens,
        estimated_cost=estimated_cost,
        requires_vision=requires_vision,
        router_id=config.router_id,
    )
    router = KeyRouter(config=config)
    decision = router.select_account(context)
    if config.mode is KeyRouterMode.SHADOW:
        return api_key
    if decision.selected_account is None:
        raise KeyRouterUnavailable("No eligible OpenRouter account")
    handle = router.secret_resolver.get(decision.selected_account.secret_ref)
    if handle is None:
        raise KeyRouterUnavailable("Selected account secret is unavailable")
    return handle.value


def dispatch_active_legacy_request(
    *,
    model: str,
    request_id: str,
    system_prompt: str,
    user_text: str,
    image_b64: str,
    image_mime: str,
    max_tokens: int,
    temperature: float,
    estimated_cost: float | None,
) -> str:
    config = KeyRouterConfig.from_env()
    if config.mode is not KeyRouterMode.ACTIVE:
        raise KeyRouterUnavailable("Active dispatcher requires KEY_ROUTER_MODE=active")
    requires_vision = bool(image_b64)
    context = RequestContext(
        request_id=request_id,
        operation_id=request_id,
        task_type=("OCR_ACCURATE_VISION" if requires_vision else "TEXT_POST_PROCESS"),
        policy_id=("OCR_ACCURATE_VISION" if requires_vision else "TEXT_POST_PROCESS"),
        modality=Modality.IMAGE if requires_vision else Modality.TEXT,
        requested_model=model,
        canonical_model_id=model,
        free_only=config.free_only_default,
        estimated_output_tokens=max_tokens,
        estimated_cost=estimated_cost,
        requires_vision=requires_vision,
        router_id=config.router_id,
        mime_type=image_mime if requires_vision else "",
        payload_bytes=(len(image_b64) * 3 // 4) if requires_vision else 0,
        image_count=1 if requires_vision else 0,
    )
    result = KeyRouter(config=config).dispatch(
        context,
        UnifiedProviderRequest(
            model=model,
            system_prompt=system_prompt,
            user_text=user_text,
            image_b64=image_b64,
            image_mime=image_mime,
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
        transport=RequestsTransport(),
    )
    return result.text
