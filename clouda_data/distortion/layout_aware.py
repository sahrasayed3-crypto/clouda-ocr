from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from clouda_data.layout.regions import Region, RegionKind


class DistortionTarget(str, Enum):
    FULL_PAGE = "full_page"
    REGION = "region"
    PAGE_EDGES = "page_edges"
    BINDING_AREA = "binding_area"
    BACKGROUND_ONLY = "background_only"
    TEXT_ONLY = "text_only"
    LOW_PRIORITY_REGIONS = "low_priority_regions"
    HIGH_PRIORITY_REGIONS = "high_priority_regions"


@dataclass(frozen=True)
class LayoutAwareRule:
    operation_name: str
    target: DistortionTarget
    include_kinds: set[RegionKind]
    exclude_kinds: set[RegionKind]
    max_severity: str = "medium"


def regions_for_rule(regions: list[Region], rule: LayoutAwareRule) -> list[Region]:
    selected = regions
    if rule.include_kinds:
        selected = [region for region in selected if region.kind in rule.include_kinds]
    if rule.exclude_kinds:
        selected = [
            region for region in selected if region.kind not in rule.exclude_kinds
        ]
    if rule.target == DistortionTarget.LOW_PRIORITY_REGIONS:
        selected = [region for region in selected if region.priority <= 3]
    if rule.target == DistortionTarget.HIGH_PRIORITY_REGIONS:
        selected = [region for region in selected if region.priority >= 7]
    return selected
