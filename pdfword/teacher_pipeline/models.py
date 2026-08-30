from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .enums import DatasetDecision, DispatchMode, TeacherRole


@dataclass(frozen=True)
class Region:
    region_id: str
    region_type: str
    coordinates: tuple[int, int, int, int]
    reading_order: int = 0
    confidence: float = 0.0

    def validate(self, width: int, height: int) -> None:
        left, top, right, bottom = self.coordinates
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("invalid_original_pixel_coordinates")
        if self.reading_order < 0:
            raise ValueError("invalid_reading_order")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("invalid_region_confidence")


@dataclass(frozen=True)
class PageAnalysis:
    page_id: str
    source_document_id: str
    page_type: str
    primary_language: str
    secondary_languages: tuple[str, ...]
    is_born_digital: bool
    is_scanned: bool
    quality_score: float
    orientation: int
    deskew_needed: bool
    layout_type: str
    column_count: int
    reading_order: tuple[str, ...]
    main_text_regions: tuple[Region, ...] = ()
    margin_regions: tuple[Region, ...] = ()
    footnote_regions: tuple[Region, ...] = ()
    header_regions: tuple[Region, ...] = ()
    footer_regions: tuple[Region, ...] = ()
    page_number_regions: tuple[Region, ...] = ()
    illustration_regions: tuple[Region, ...] = ()
    table_regions: tuple[Region, ...] = ()
    uncertain_regions: tuple[Region, ...] = ()
    has_small_marginalia: bool = False
    has_footnotes: bool = False
    has_mixed_language: bool = False
    recommended_teacher_roles: tuple[TeacherRole, ...] = ()
    analysis_confidence: float = 0.0
    warnings: tuple[str, ...] = ()

    def validate(self, width: int, height: int) -> None:
        if not self.page_id or not self.source_document_id:
            raise ValueError("missing_page_identity")
        if not 0.0 <= self.quality_score <= 100.0:
            raise ValueError("invalid_quality_score")
        if self.orientation not in {0, 90, 180, 270}:
            raise ValueError("invalid_orientation")
        if self.column_count < 0:
            raise ValueError("invalid_column_count")
        if not 0.0 <= self.analysis_confidence <= 1.0:
            raise ValueError("invalid_analysis_confidence")
        regions = self.all_regions()
        for region in regions:
            region.validate(width, height)
        ids = [item.region_id for item in regions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_region_id")
        if set(self.reading_order) - set(ids):
            raise ValueError("unknown_reading_order_region")
        if self.has_footnotes and not self.footnote_regions:
            raise ValueError("footnote_regions_required")

    def all_regions(self) -> tuple[Region, ...]:
        return (
            self.main_text_regions
            + self.margin_regions
            + self.footnote_regions
            + self.header_regions
            + self.footer_regions
            + self.page_number_regions
            + self.illustration_regions
            + self.table_regions
            + self.uncertain_regions
        )


@dataclass(frozen=True)
class SourceRightsRecord:
    source_document_id: str
    source_page_id: str
    source_hash: str
    rights_record_id: str
    rights_status: str
    allowed_for_processing: bool
    allowed_for_training: bool
    allowed_for_commercial_training: bool
    allowed_for_weight_release: bool
    allowed_for_data_redistribution: bool
    permission_source: str
    permission_version: str
    permission_date: str
    data_classification: str
    retention_policy: str

    def validate(self) -> None:
        required = (
            self.source_document_id,
            self.source_page_id,
            self.source_hash,
            self.rights_record_id,
            self.rights_status,
            self.permission_source,
            self.permission_version,
            self.permission_date,
            self.data_classification,
            self.retention_policy,
        )
        if not all(str(item).strip() for item in required):
            raise PermissionError("source_rights_incomplete")
        try:
            date.fromisoformat(self.permission_date)
        except ValueError as exc:
            raise PermissionError("invalid_permission_date") from exc
        if self.rights_status.lower() not in {"verified", "authorized"}:
            raise PermissionError("source_rights_not_verified")
        if not self.allowed_for_processing:
            raise PermissionError("processing_not_authorized")


@dataclass(frozen=True)
class TeacherRoleAssignment:
    assignment_id: str
    account_id: str
    owner_id: str
    provider_id: str
    quota_domain_id: str
    allowed_teacher_roles: tuple[TeacherRole, ...]
    allowed_task_types: tuple[str, ...]
    allowed_data_classifications: tuple[str, ...]
    enabled: bool = False
    dispatch_mode: DispatchMode = DispatchMode.DISABLED
    secret_ref: str = ""
    priority: int = 0
    daily_request_budget: int = 0
    daily_token_budget: int = 0
    daily_cost_budget: float = 0.0
    cooldown_until: str = ""
    terms_metadata: dict[str, Any] = field(default_factory=dict)
    provenance_metadata: dict[str, Any] = field(default_factory=dict)

    def eligible(
        self, role: TeacherRole, task_type: str, classification: str
    ) -> tuple[bool, str]:
        if not self.enabled:
            return False, "teacher_assignment_disabled"
        if self.dispatch_mode is DispatchMode.LIVE:
            return False, "live_assignment_locked"
        if role not in self.allowed_teacher_roles:
            return False, "teacher_role_not_allowed"
        if task_type not in self.allowed_task_types:
            return False, "task_type_not_allowed"
        if classification not in self.allowed_data_classifications:
            return False, "data_classification_not_allowed"
        if not self.owner_id:
            return False, "owner_missing"
        if not self.quota_domain_id:
            return False, "quota_domain_missing"
        return True, "eligible"


@dataclass(frozen=True)
class UncertaintySpan:
    start: int
    end: int
    reason: str
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class TeacherOutput:
    output_id: str
    job_id: str
    provider: str
    model: str
    endpoint: str
    provider_account_id: str
    owner_id: str
    quota_domain_id: str
    teacher_role: TeacherRole
    task_contract_version: str
    prompt_version: str
    request_id: str
    attempt_id: str
    reservation_id: str
    source_hash: str
    region_hash: str
    input_manifest_hash: str
    normalized_output_hash: str
    raw_output_retention_status: str
    latency_ms: int
    token_estimate: int
    cost_estimate: float
    mock_or_live: str
    created_at: str
    decision_lineage: tuple[str, ...]
    transcription: str = ""
    uncertainty_spans: tuple[UncertaintySpan, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class AgreementMetrics:
    normalized_character_agreement: float
    normalized_word_agreement: float
    line_count_disagreement: float
    missing_line_indicators: int
    duplicate_text_indicators: int
    reading_order_disagreement: float
    region_coverage: float
    main_text_coverage: float
    margin_coverage: float
    footnote_coverage: float
    uncertain_span_ratio: float
    teacher_confidence: float
    judge_confidence: float
    source_image_coverage: float
    hallucination_indicators: int


@dataclass(frozen=True)
class DecisionRecord:
    decision: DatasetDecision
    policy_version: str
    metrics: AgreementMetrics
    reason_codes: tuple[str, ...]
    human_verified: bool = False
    trusted_reference: bool = False
