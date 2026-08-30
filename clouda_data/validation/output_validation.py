from __future__ import annotations

from pathlib import Path

from .image_quality import validate_image_file


def validate_output_bundle(
    image_path: str | Path, ground_truth_path: str | Path, metadata_path: str | Path
) -> None:
    image_result = validate_image_file(image_path)
    if not image_result.ok:
        raise ValueError(f"Invalid output image: {image_result.reason}")
    for path in [ground_truth_path, metadata_path]:
        if not Path(path).exists():
            raise ValueError(f"Missing output bundle file: {path}")
