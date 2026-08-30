from __future__ import annotations

from .regions import Region


def sort_reading_order(regions: list[Region]) -> list[Region]:
    return sorted(regions, key=lambda r: (r.reading_order, r.box.y, r.box.x, r.id))


def reading_order_ids(regions: list[Region]) -> list[str]:
    return [region.id for region in sort_reading_order(regions)]
