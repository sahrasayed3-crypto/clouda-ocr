from __future__ import annotations


def validate_region_count(source_count: int, output_count: int) -> None:
    if output_count < source_count:
        raise ValueError("Generated metadata is missing text/layout regions.")


def validate_crop_ratio(crop_ratio: float, max_allowed: float) -> None:
    if crop_ratio > max_allowed:
        raise ValueError("Excessive crop detected.")
