from __future__ import annotations

from .regions import Region


def validate_regions(regions: list[Region], page_width: int, page_height: int) -> None:
    seen: set[str] = set()
    for region in regions:
        if region.id in seen:
            raise ValueError(f"Duplicate region id: {region.id}")
        seen.add(region.id)
        if region.box.width <= 0 or region.box.height <= 0:
            raise ValueError(f"Region {region.id} has invalid dimensions.")
        if region.box.x < 0 or region.box.y < 0:
            raise ValueError(f"Region {region.id} has negative coordinates.")
        if (
            region.box.x + region.box.width > page_width
            or region.box.y + region.box.height > page_height
        ):
            raise ValueError(f"Region {region.id} extends beyond page bounds.")
        if region.reading_order < 0:
            raise ValueError(f"Region {region.id} has invalid reading order.")
