from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance

from .models import Region


@dataclass(frozen=True)
class DerivedAsset:
    path: Path
    source_page_hash: str
    region_id: str
    region_type: str
    original_coordinates: tuple[int, int, int, int]
    transform_chain: tuple[str, ...]
    derived_asset_hash: str
    parent_asset_hash: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_regions(
    source: str | Path,
    regions: tuple[Region, ...],
    output_dir: str | Path,
    *,
    enhance_contrast: bool = False,
) -> tuple[DerivedAsset, ...]:
    source_path = Path(source).resolve()
    output_root = Path(output_dir).resolve()
    if not source_path.is_file():
        raise FileNotFoundError("source_page_missing")
    output_root.mkdir(parents=True, exist_ok=True)
    source_hash_before = sha256_file(source_path)
    assets: list[DerivedAsset] = []
    with Image.open(source_path) as opened:
        original = opened.convert("RGB")
        for region in regions:
            region.validate(original.width, original.height)
            crop = original.crop(region.coordinates)
            transforms = ["crop_from_original_pixels"]
            if enhance_contrast:
                crop = ImageEnhance.Contrast(crop).enhance(1.25)
                transforms.append("contrast_1.25")
            target = output_root / f"{region.region_id}.png"
            crop.save(target, format="PNG", optimize=False)
            assets.append(
                DerivedAsset(
                    path=target,
                    source_page_hash=source_hash_before,
                    region_id=region.region_id,
                    region_type=region.region_type,
                    original_coordinates=region.coordinates,
                    transform_chain=tuple(transforms),
                    derived_asset_hash=sha256_file(target),
                    parent_asset_hash=source_hash_before,
                )
            )
    if sha256_file(source_path) != source_hash_before:
        raise RuntimeError("source_page_was_modified")
    return tuple(assets)
