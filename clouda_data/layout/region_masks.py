from __future__ import annotations

from .regions import Box, Region, RegionKind


def select_regions(
    regions: list[Region],
    kinds: set[RegionKind] | None = None,
    min_priority: int | None = None,
) -> list[Region]:
    selected = regions
    if kinds is not None:
        selected = [region for region in selected if region.kind in kinds]
    if min_priority is not None:
        selected = [region for region in selected if region.priority >= min_priority]
    return selected


def page_edge_boxes(width: int, height: int, edge_size: int) -> dict[str, Box]:
    return {
        "top": Box(0, 0, width, edge_size),
        "bottom": Box(0, height - edge_size, width, edge_size),
        "left": Box(0, 0, edge_size, height),
        "right": Box(width - edge_size, 0, edge_size, height),
    }
