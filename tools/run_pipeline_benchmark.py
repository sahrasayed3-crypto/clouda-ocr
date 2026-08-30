from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw

from clouda_contracts.storage import StorageRoots
from clouda_data.distortion.base import DistortionSpec
from clouda_data.distortion.pipeline import DistortionPipeline
from clouda_data.distortion.registry import default_registry
from clouda_data.pipeline.profiles import load_profile, profile_to_specs

OPERATORS = (
    "rotation",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
    "uneven_illumination",
    "yellow_paper",
    "ink_erosion",
)
PROFILES = (
    "clean_control",
    "modern_scan_medium",
    "old_book_medium",
    "jpeg_damage",
    "weak_scan_arabic",
)


def _fixture() -> Image.Image:
    image = Image.new("RGB", (512, 640), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 488, 616), outline="black", width=2)
    for row in range(12):
        draw.text((60, 70 + row * 38), f"Synthetic OCR row {row:02d}", fill="black")
    return image


def _encoded_size(image: Image.Image) -> int:
    import io

    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return len(buffer.getvalue())


def _timed_operator(name: str, image: Image.Image, seed: int) -> tuple[float, int]:
    operator = default_registry().create(DistortionSpec(name=name, severity="medium"))
    started = time.perf_counter()
    output, _ = operator.apply(image, seed)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, _encoded_size(output)


def run() -> dict[str, Any]:
    roots = StorageRoots.from_env()
    project_root = Path(
        os.getenv("CLOUDA_PROJECT_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    image = _fixture()
    source_bytes = _encoded_size(image)

    operator_results: dict[str, Any] = {}
    for index, name in enumerate(OPERATORS):
        samples = [
            _timed_operator(name, image, 20260723 + index + repeat)
            for repeat in range(3)
        ]
        average_ms = mean(sample[0] for sample in samples)
        operator_results[name] = {
            "milliseconds": round(average_ms, 3),
            "images_per_minute": round(60_000 / max(average_ms, 0.001), 2),
            "output_size_growth_ratio": round(
                mean(sample[1] for sample in samples) / source_bytes,
                4,
            ),
        }

    profile_results: dict[str, Any] = {}
    for index, profile_name in enumerate(PROFILES):
        profile_path = (
            project_root
            / "configs"
            / "data_foundation"
            / "distortions"
            / f"{profile_name}.yaml"
        )
        profile = load_profile(profile_path)
        pipeline = DistortionPipeline(
            profile_name=profile_name,
            operations=profile_to_specs(profile),
            base_seed=20260723,
        )
        started = time.perf_counter()
        output, transformations = pipeline.run(image, f"benchmark:{index}")
        elapsed_ms = (time.perf_counter() - started) * 1000
        profile_results[profile_name] = {
            "milliseconds": round(elapsed_ms, 3),
            "images_per_minute": round(60_000 / max(elapsed_ms, 0.001), 2),
            "operators_applied": len(transformations),
            "output_size_growth_ratio": round(
                _encoded_size(output) / source_bytes,
                4,
            ),
        }

    worker_scaling: dict[str, Any] = {}
    tasks = [(OPERATORS[index % len(OPERATORS)], 9000 + index) for index in range(12)]
    for workers in (1, 2, 4):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(
                executor.map(
                    lambda item: _timed_operator(item[0], image, item[1]),
                    tasks,
                )
            )
        elapsed = time.perf_counter() - started
        worker_scaling[str(workers)] = {
            "tasks": len(tasks),
            "elapsed_seconds": round(elapsed, 4),
            "images_per_minute": round(len(tasks) * 60 / max(elapsed, 0.001), 2),
            "scope": "operator-level benchmark; batch manifests remain sequential",
        }

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bounded": True,
        "network_accessed": False,
        "fixture_dimensions": list(image.size),
        "memory_estimate_bytes_per_rgb_image": image.width * image.height * 3,
        "operator_results": operator_results,
        "profile_results": profile_results,
        "worker_scaling": worker_scaling,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_root = roots.artifact_root / "reports" / "performance"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"pipeline-benchmark-{stamp}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
