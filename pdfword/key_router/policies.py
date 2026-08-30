from __future__ import annotations

from dataclasses import dataclass

from .enums import CapabilityVerificationStatus, PrivacyClassification


@dataclass(frozen=True)
class RoutingPolicy:
    policy_id: str
    required_capabilities: tuple[str, ...] = ()
    preferred_capabilities: tuple[str, ...] = ()
    forbidden_capabilities: tuple[str, ...] = ()
    local_ocr_first: bool = False
    vision_required: bool = False
    structured_output_required: bool = False
    maximum_cost: float = 0.0
    maximum_latency_ms: int = 0
    maximum_attempts: int = 1
    maximum_provider_failovers: int = 0
    maximum_account_failovers: int = 0
    allowed_providers: tuple[str, ...] = ()
    blocked_providers: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    minimum_verified_status: CapabilityVerificationStatus = (
        CapabilityVerificationStatus.DECLARED
    )
    quality_target: float = 90.0
    fallback_policy: str = ""
    privacy_classification: PrivacyClassification = PrivacyClassification.INTERNAL
    allow_retry_after_send_timeout: bool = False


POLICIES: dict[str, RoutingPolicy] = {
    "DIGITAL_TEXT_VALIDATE": RoutingPolicy(
        policy_id="DIGITAL_TEXT_VALIDATE",
        required_capabilities=("text",),
        maximum_attempts=1,
        fallback_policy="TEXT_POST_PROCESS",
    ),
    "OCR_FAST": RoutingPolicy(
        policy_id="OCR_FAST",
        required_capabilities=("text",),
        local_ocr_first=True,
        maximum_attempts=2,
        maximum_provider_failovers=1,
        maximum_account_failovers=1,
    ),
    "OCR_ACCURATE_VISION": RoutingPolicy(
        policy_id="OCR_ACCURATE_VISION",
        required_capabilities=("vision",),
        vision_required=True,
        maximum_attempts=2,
        maximum_provider_failovers=1,
        maximum_account_failovers=1,
        minimum_verified_status=CapabilityVerificationStatus.VERIFIED,
    ),
    "OCR_ARABIC_COMPLEX": RoutingPolicy(
        policy_id="OCR_ARABIC_COMPLEX",
        required_capabilities=("vision", "system_prompt"),
        preferred_capabilities=("structured_json",),
        vision_required=True,
        maximum_attempts=3,
        maximum_provider_failovers=1,
        maximum_account_failovers=1,
        minimum_verified_status=CapabilityVerificationStatus.VERIFIED,
        quality_target=90.0,
    ),
    "OCR_MIXED_LANGUAGE": RoutingPolicy(
        policy_id="OCR_MIXED_LANGUAGE",
        required_capabilities=("vision",),
        vision_required=True,
        maximum_attempts=2,
        maximum_provider_failovers=1,
        minimum_verified_status=CapabilityVerificationStatus.VERIFIED,
    ),
    "OCR_RETRY_LOW_QUALITY": RoutingPolicy(
        policy_id="OCR_RETRY_LOW_QUALITY",
        required_capabilities=("vision",),
        vision_required=True,
        maximum_attempts=2,
        maximum_provider_failovers=1,
        maximum_account_failovers=1,
        minimum_verified_status=CapabilityVerificationStatus.VERIFIED,
    ),
    "TEXT_POST_PROCESS": RoutingPolicy(
        policy_id="TEXT_POST_PROCESS",
        required_capabilities=("text",),
        maximum_attempts=2,
        maximum_provider_failovers=1,
    ),
    "STRUCTURED_EXTRACTION": RoutingPolicy(
        policy_id="STRUCTURED_EXTRACTION",
        required_capabilities=("text", "structured_json"),
        structured_output_required=True,
        maximum_attempts=2,
        maximum_provider_failovers=1,
    ),
}


def get_policy(policy_id: str) -> RoutingPolicy:
    normalized = (policy_id or "OCR_FAST").strip().upper()
    normalized = {
        "OCR": "OCR_FAST",
        "LEGACY_OPENROUTER": "TEXT_POST_PROCESS",
    }.get(normalized, normalized)
    try:
        return POLICIES[normalized]
    except KeyError as exc:
        raise ValueError("Unknown routing policy") from exc


def choose_ocr_policy(
    *,
    born_digital: bool,
    local_text_usable: bool,
    page_quality: float | None,
    complex_layout: bool = False,
    mixed_language: bool = False,
) -> RoutingPolicy:
    if born_digital and local_text_usable:
        return POLICIES["DIGITAL_TEXT_VALIDATE"]
    if mixed_language:
        return POLICIES["OCR_MIXED_LANGUAGE"]
    if complex_layout:
        return POLICIES["OCR_ARABIC_COMPLEX"]
    if page_quality is not None and page_quality < 90.0:
        return POLICIES["OCR_RETRY_LOW_QUALITY"]
    return POLICIES["OCR_FAST"]
