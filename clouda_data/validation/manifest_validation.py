from __future__ import annotations

from clouda_data.core.manifest import GeneratedPageManifestEntry


def validate_generated_manifest(entry: GeneratedPageManifestEntry) -> None:
    required = [
        entry.source_document_id,
        entry.source_page_id,
        entry.generated_page_id,
        entry.source_checksum,
        entry.text_checksum,
        entry.distortion_profile,
    ]
    if not all(required):
        raise ValueError("Manifest entry is missing required metadata.")
    if entry.image_width <= 0 or entry.image_height <= 0:
        raise ValueError("Manifest image dimensions are invalid.")
    if not entry.distortion_operations:
        raise ValueError(
            "Manifest must list distortion operations, even for clean_control."
        )
