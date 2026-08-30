from __future__ import annotations

from dataclasses import dataclass, field

from .enums import CapabilityVerificationStatus, Modality
from .exceptions import UnsupportedCapability
from .models import RequestContext


@dataclass(frozen=True)
class ProviderCapabilities:
    """Provider-wide transport features, never a substitute for endpoint evidence."""

    supports_text_input: bool = True
    supports_image_input: bool = False
    supports_pdf_input: bool = False
    supports_text_output: bool = True
    supports_structured_output: bool = False
    supports_streaming: bool = False
    supports_usage_reporting: bool = False
    supports_model_listing: bool = False
    supports_price_listing: bool = False
    supports_rate_limit_headers: bool = False
    max_image_size: int | None = None
    max_request_size: int | None = None
    max_context_tokens: int | None = None
    supported_regions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CanonicalModelMapping:
    canonical_model_id: str
    provider: str
    provider_model_id: str
    endpoint_id: str


@dataclass(frozen=True)
class ModelEndpointCapability:
    endpoint_id: str
    canonical_model_id: str
    provider: str
    provider_model_id: str
    endpoint_type: str = "chat"
    supports_text: bool = True
    supports_vision_declared: bool = False
    supports_vision_verified: bool = False
    supports_base64: bool = False
    supports_image_url: bool = False
    supports_native_pdf: bool = False
    supports_multiple_images: bool = False
    max_images_per_request: int = 1
    supported_mime_types: tuple[str, ...] = ()
    max_image_width: int = 0
    max_image_height: int = 0
    max_image_pixels: int = 0
    max_payload_bytes: int = 0
    max_context_tokens: int = 0
    max_output_tokens: int = 0
    supports_structured_json: bool = False
    supports_json_schema: bool = False
    supports_streaming: bool = False
    supports_system_prompt: bool = True
    supports_temperature: bool = True
    supports_seed: bool = False
    supports_tools: bool = False
    supports_parallel_tools: bool = False
    supports_reasoning: bool = False
    pricing_input: float = 0.0
    pricing_output: float = 0.0
    pricing_image: float = 0.0
    currency: str = "USD"
    region_availability: tuple[str, ...] = ()
    declared_status: CapabilityVerificationStatus = (
        CapabilityVerificationStatus.DECLARED
    )
    verified_status: CapabilityVerificationStatus = CapabilityVerificationStatus.UNKNOWN
    declared_at: str = ""
    last_verified_at: str = ""
    capability_source: str = ""
    notes: str = ""
    enabled: bool = False
    expected_latency_ms: int = 0
    quality_score: float = 0.0
    supports_text_output: bool = True
    privacy_compatibility: tuple[str, ...] = ()
    region_support: tuple[str, ...] = ()
    teacher_role_support: tuple[str, ...] = ()
    training_output_policy_status: str = "unknown"

    @property
    def mapping(self) -> CanonicalModelMapping:
        return CanonicalModelMapping(
            canonical_model_id=self.canonical_model_id,
            provider=self.provider,
            provider_model_id=self.provider_model_id,
            endpoint_id=self.endpoint_id,
        )

    def supports_required_vision(
        self,
        *,
        require_verified: bool,
        allow_declared_only: bool,
    ) -> bool:
        if self.supports_vision_verified:
            return self.verified_status is CapabilityVerificationStatus.VERIFIED
        if require_verified:
            return False
        return bool(self.supports_vision_declared and allow_declared_only)

    def validate_request(
        self,
        context: RequestContext,
        *,
        require_verified_vision: bool,
        allow_declared_only: bool,
    ) -> None:
        modality = context.normalized_modality
        vision_required = context.requires_vision or modality == Modality.IMAGE.value
        if vision_required and not self.supports_required_vision(
            require_verified=require_verified_vision,
            allow_declared_only=allow_declared_only,
        ):
            raise UnsupportedCapability("verified_vision_required")
        if modality == Modality.PDF.value and not self.supports_native_pdf:
            raise UnsupportedCapability("native_pdf_not_supported")
        if context.requires_structured_output and not self.supports_structured_json:
            raise UnsupportedCapability("structured_output_not_supported")
        if not self.supports_text_output:
            raise UnsupportedCapability("text_output_not_supported")
        if context.teacher_role and context.teacher_role not in set(
            self.teacher_role_support
        ):
            raise UnsupportedCapability("teacher_role_not_supported")
        if context.region_type and context.region_type not in set(self.region_support):
            raise UnsupportedCapability("region_type_not_supported")
        if self.privacy_compatibility and context.normalized_privacy not in set(
            self.privacy_compatibility
        ):
            raise UnsupportedCapability("endpoint_privacy_not_compatible")
        if (
            context.required_training_output_policy_status
            and self.training_output_policy_status
            != context.required_training_output_policy_status
        ):
            raise UnsupportedCapability("training_output_policy_not_satisfied")
        if context.mime_type:
            normalized_mimes = {item.lower() for item in self.supported_mime_types}
            if normalized_mimes and context.mime_type.lower() not in normalized_mimes:
                raise UnsupportedCapability("unsupported_mime_type")
        if self.max_payload_bytes and context.payload_bytes > self.max_payload_bytes:
            raise UnsupportedCapability("payload_too_large")
        if (
            self.max_images_per_request
            and context.image_count > self.max_images_per_request
        ):
            raise UnsupportedCapability("too_many_images")
        if self.max_image_width and context.image_width > self.max_image_width:
            raise UnsupportedCapability("image_too_wide")
        if self.max_image_height and context.image_height > self.max_image_height:
            raise UnsupportedCapability("image_too_tall")
        if (
            self.max_image_pixels
            and context.image_width
            and context.image_height
            and context.image_width * context.image_height > self.max_image_pixels
        ):
            raise UnsupportedCapability("image_too_many_pixels")
