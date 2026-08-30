from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat

from clouda_contracts.storage import StorageRoots
from clouda_contracts.security import sanitize_spreadsheet_cell
from clouda_data.datasets.license_gate import decide_dataset_use
from clouda_data.pipeline.profiles import load_profile, profile_to_specs

from .pipeline import DistortionPipeline
from .checkpoints import RunCheckpointStore

PAGE_STATES = {
    "queued",
    "processing",
    "complete",
    "failed",
    "skipped",
    "quarantined",
    "cancelled",
    "manual_review",
}
CONFLICT_POLICIES = {
    "reject",
    "skip_identical",
    "version_new",
    "overwrite_only_with_explicit_flag",
}


class RunInterrupted(RuntimeError):
    """Raised only when an explicit interruption hook is requested."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _atomic_save(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{target.stem}-", suffix=".png", dir=target.parent
    )
    os.close(fd)
    temp = Path(name)
    try:
        image.save(temp, "PNG", optimize=True)
        if target.exists():
            raise FileExistsError(target)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def read_jsonl(
    path: str | Path,
    *,
    max_bytes: int = 64 * 1024 * 1024,
    max_records: int = 100_000,
    max_line_bytes: int = 2 * 1024 * 1024,
) -> list[dict[str, Any]]:
    source = Path(path)
    if source.stat().st_size > max_bytes:
        raise ValueError("JSONL manifest exceeds the configured byte limit")
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > max_line_bytes:
                raise ValueError(f"JSONL line {line_number} exceeds the byte limit")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            records.append(value)
            if len(records) > max_records:
                raise ValueError("JSONL manifest contains too many records")
    return records


def _dataset_uri_path(roots: StorageRoots, value: object) -> Path:
    uri = str(value)
    if not uri.startswith("dataset://"):
        raise PermissionError("Expected a dataset:// storage URI")
    return roots.resolve_uri(uri)


def _safe_generated_id(value: object) -> str:
    identifier = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
        raise ValueError("Unsafe generated page identifier")
    return identifier


def _validate_resume_record(
    record: dict[str, Any],
    *,
    roots: StorageRoots,
    expected_output: Path,
    dry_run: bool,
) -> None:
    status = str(record.get("status", ""))
    if status == "skipped" and dry_run:
        return
    output = _dataset_uri_path(roots, record.get("output_uri"))
    if output.resolve() != expected_output.resolve():
        raise ValueError("Resume manifest output path does not match the run")
    checksum = str(record.get("output_checksum") or "")
    if not output.is_file() or not checksum or _sha256(output) != checksum:
        raise ValueError("Resume manifest output is missing or has a checksum mismatch")


def _matches_existing_image(path: Path, expected: Image.Image) -> bool:
    try:
        with Image.open(path) as existing:
            existing.load()
            return (
                existing.size == expected.size
                and existing.convert("RGBA").tobytes()
                == expected.convert("RGBA").tobytes()
            )
    except Exception:
        return False


def classify_visual_difficulty(
    image: Image.Image,
    severity: str,
    distortion_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean = float(stat.mean[0])
    stddev = float(stat.stddev[0])
    histogram = gray.histogram()
    pixels = max(1, image.width * image.height)
    entropy = -sum(
        (count / pixels) * math.log2(count / pixels) for count in histogram if count
    )
    white_ratio = sum(histogram[250:]) / pixels
    black_ratio = sum(histogram[:6]) / pixels
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_histogram = edges.histogram()
    edge_density = sum(edge_histogram[32:]) / pixels
    edge_variance = float(ImageStat.Stat(edges).var[0])
    blur_score = 1.0 / (1.0 + edge_variance / 1000.0)
    median = gray.filter(ImageFilter.MedianFilter(3))
    noise_image = ImageChops.difference(gray, median)
    noise_level = float(ImageStat.Stat(noise_image).mean[0]) / 255.0
    foreground_ratio = 1.0 - white_ratio
    boundary_differences: list[float] = []
    sample = gray.resize(
        (min(gray.width, 1024), min(gray.height, 1024)), Image.Resampling.BILINEAR
    )
    for x in range(8, sample.width, 8):
        boundary_differences.append(
            float(
                ImageStat.Stat(
                    ImageChops.difference(
                        sample.crop((x - 1, 0, x, sample.height)),
                        sample.crop((x, 0, x + 1, sample.height)),
                    )
                ).mean[0]
            )
        )
    compression_artifact_score = (
        sum(boundary_differences) / len(boundary_differences) / 255.0
        if boundary_differences
        else 0.0
    )
    metadata = distortion_metadata or {}
    transformations = [str(item) for item in metadata.get("transformation_chain", [])]
    parameters = metadata.get("transformation_parameters", [])
    skew_degrees = 0.0
    for name, values in zip(transformations, parameters):
        if name in {"rotation", "small_rotation", "skew", "page_misalignment"}:
            skew_degrees = max(
                skew_degrees,
                (
                    abs(float(values.get("degrees", 0.0)))
                    if isinstance(values, dict)
                    else 0
                ),
            )
    region_heights = [
        abs(float(region["bbox"][3]) - float(region["bbox"][1]))
        for region in metadata.get("regions", [])
        if isinstance(region, dict)
        and isinstance(region.get("bbox"), (list, tuple))
        and len(region["bbox"]) == 4
    ]
    estimated_text_size = (
        sum(region_heights) / len(region_heights) if region_heights else 0.0
    )
    score = {"minimal": 0, "light": 1, "medium": 2, "heavy": 3, "extreme": 4}.get(
        severity, 2
    )
    if stddev < 5 or white_ratio > 0.9999 or black_ratio > 0.995 or entropy < 0.05:
        label = "invalid"
    elif (
        score >= 4
        or stddev < 30
        or blur_score > 0.92
        or noise_level > 0.14
        or compression_artifact_score > 0.22
    ):
        label = "extreme"
    elif score >= 3 or blur_score > 0.78 or noise_level > 0.08 or skew_degrees > 4:
        label = "difficult"
    elif score >= 2:
        label = "medium"
    else:
        label = "easy"
    return {
        "estimated_visual_difficulty": label,
        "brightness": mean,
        "contrast": stddev,
        "page_entropy": entropy,
        "blankness": white_ratio,
        "blackness": black_ratio,
        "blur_score": blur_score,
        "noise_level": noise_level,
        "edge_density": edge_density,
        "compression_artifact_score": compression_artifact_score,
        "foreground_ratio": foreground_ratio,
        "estimated_skew_degrees": skew_degrees,
        "estimated_text_size": estimated_text_size,
    }


def _validate_asset(
    record: dict[str, Any],
    *,
    roots: StorageRoots,
    source_path: Path,
    output_path: Path,
    image: Image.Image,
) -> list[str]:
    errors: list[str] = []
    if not _inside(output_path, roots.dataset_root):
        errors.append("output_root_escape")
    if not output_path.is_file():
        errors.append("missing_output")
        return errors
    if _sha256(output_path) != record.get("output_checksum"):
        errors.append("checksum_mismatch")
    try:
        with Image.open(output_path) as decoded:
            decoded.verify()
    except Exception:
        errors.append("decode_failed")
    if image.width < 1 or image.height < 1:
        errors.append("invalid_dimensions")
    if image.width * image.height > 100_000_000:
        errors.append("maximum_pixels_exceeded")
    if max(image.size) > 20_000:
        errors.append("maximum_dimension_exceeded")
    if record.get("transformation_chain"):
        with Image.open(source_path) as source_image:
            source_pixels = source_image.convert("RGB")
            output_pixels = image.convert("RGB")
            if (
                source_pixels.size == output_pixels.size
                and not ImageChops.difference(source_pixels, output_pixels).getbbox()
            ):
                errors.append("identical_to_source")
    difficulty = classify_visual_difficulty(
        image, str(record.get("overall_severity", "medium"))
    )
    if difficulty["estimated_visual_difficulty"] == "invalid" and record.get(
        "page_type"
    ) not in {"blank", "near_blank"}:
        errors.append("invalid_visual_quality")
    if not record.get("seeds"):
        errors.append("missing_seed")
    if not record.get("profile_hash"):
        errors.append("missing_profile_hash")
    if not record.get("ground_truth_reference"):
        errors.append("missing_ground_truth_reference")
    if not record.get("run_id"):
        errors.append("missing_run_id")
    if not record.get("transformation_parameters"):
        errors.append("missing_transformation_parameters")
    if not record.get("affected_regions"):
        errors.append("missing_affected_regions")
    if record.get("license_status") in {"blocked", "pending", "expired"}:
        errors.append("license_not_eligible")
    if record.get("run_id") and record.get("variant") is not None:
        expected_id = hashlib.sha256(
            (
                f"{record['run_id']}:{record.get('source_page_id')}:"
                f"{record['variant']}"
            ).encode()
        ).hexdigest()[:32]
        if expected_id != record.get("generated_page_id"):
            errors.append("non_deterministic_generated_id")
    return errors


def run_distortion_batch(
    input_manifest: str | Path,
    profile_path: str | Path,
    *,
    output_root: str | Path | None = None,
    seed: int = 1,
    variants: int = 1,
    max_pages: int = 100,
    allow_large_run: bool = False,
    dry_run: bool = False,
    resume: bool = False,
    fail_fast: bool = False,
    conflict_policy: str = "reject",
    interrupt_after: int | None = None,
    start_page: int = 1,
    end_page: int | None = None,
    include_dataset_ids: set[str] | None = None,
    exclude_dataset_ids: set[str] | None = None,
    maximum_output_bytes: int = 1024 * 1024 * 1024,
    workers: int = 1,
    max_retries: int = 2,
    stale_after_seconds: int = 300,
) -> Path:
    if max_pages < 1 or variants < 1:
        raise ValueError("max_pages and variants must be positive")
    if max_pages > 100 and not allow_large_run:
        raise PermissionError("--allow-large-run is required above 100 pages")
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError("invalid conflict policy")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if maximum_output_bytes < 1:
        raise ValueError("maximum_output_bytes must be positive")
    if not 0 <= max_retries <= 10:
        raise ValueError("max_retries must be between 0 and 10")
    if stale_after_seconds < 1:
        raise ValueError("stale_after_seconds must be positive")
    roots = StorageRoots.from_env()
    manifest_path = Path(input_manifest).expanduser().resolve()
    if not _inside(manifest_path, roots.dataset_root):
        raise PermissionError("input manifest must be inside dataset root")
    profile = load_profile(profile_path)
    profile_hash = hashlib.sha256(
        json.dumps(profile, sort_keys=True).encode()
    ).hexdigest()
    input_records = read_jsonl(manifest_path)
    input_records = input_records[start_page - 1 : end_page]
    if include_dataset_ids:
        input_records = [
            item
            for item in input_records
            if str(item.get("dataset_id")) in include_dataset_ids
        ]
    if exclude_dataset_ids:
        input_records = [
            item
            for item in input_records
            if str(item.get("dataset_id")) not in exclude_dataset_ids
        ]
    input_records = input_records[:max_pages]
    material = (
        f"{_sha256(manifest_path)}:{profile['name']}:{profile_hash}:{seed}:{variants}"
    )
    run_id = hashlib.sha256(material.encode()).hexdigest()[:24]
    root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else roots.dataset_root / "distorted"
    )
    if not _inside(root, roots.dataset_root):
        raise PermissionError("output root must be inside dataset root")
    run_root = root / run_id
    output_manifest = run_root / "distortion_manifest.v1.jsonl"
    existing = (
        read_jsonl(output_manifest) if resume and output_manifest.is_file() else []
    )
    by_id = {str(item["generated_page_id"]): item for item in existing}
    records = list(existing)
    pipeline = DistortionPipeline(
        profile_name=str(profile["name"]),
        operations=profile_to_specs(profile),
        base_seed=seed,
    )
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoints = RunCheckpointStore(run_root / "checkpoints.sqlite3")
    checkpoints.start_run(
        run_id,
        input_checksum=_sha256(manifest_path),
        profile_hash=profile_hash,
        metadata={
            "profile_id": profile["name"],
            "max_pages": max_pages,
            "variants": variants,
            "seed": seed,
        },
        resume=resume,
    )
    if resume:
        checkpoints.recover_stale(stale_after_seconds=stale_after_seconds)
    run_manifest = run_root / "run_manifest.v1.json"
    _write_json_atomic(
        run_manifest,
        {
            "schema_version": 1,
            "run_id": run_id,
            "profile_id": profile["name"],
            "profile_hash": profile_hash,
            "state": "processing",
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "max_pages": max_pages,
            "variants": variants,
            "dry_run": dry_run,
            "workers_requested": workers,
            "execution_mode": "deterministic_sequential",
        },
    )
    output_bytes = sum(
        _dataset_uri_path(roots, item["output_uri"]).stat().st_size
        for item in existing
        if item.get("output_uri")
        and _dataset_uri_path(roots, item["output_uri"]).is_file()
    )
    for source in input_records:
        source_uri = str(source.get("output_uri") or source.get("image_uri") or "")
        if not source_uri.startswith("dataset://"):
            if fail_fast:
                raise ValueError("unsafe source URI")
            continue
        source_path = _dataset_uri_path(roots, source_uri)
        if not _inside(source_path, roots.dataset_root) or not source_path.is_file():
            if fail_fast:
                raise FileNotFoundError(source_path)
            continue
        dataset_id = str(source.get("dataset_id", "synthetic_evaluation"))
        license_status = str(source.get("license_status", "evaluation_only"))
        commercial_allowed = False
        if dataset_id != "synthetic_evaluation":
            decision = decide_dataset_use(dataset_id, purpose="evaluation")
            if not decision.allowed:
                if fail_fast:
                    raise PermissionError(f"Dataset {dataset_id} is not approved")
                continue
            license_status = decision.status
            commercial_allowed = decide_dataset_use(
                dataset_id, purpose="commercial_training"
            ).allowed
        source_page_id = str(
            source.get("page_id")
            or f"{source.get('source_document_id')}:{source.get('source_page_number')}"
        )
        for variant in range(variants):
            generated_id = hashlib.sha256(
                f"{run_id}:{source_page_id}:{variant}".encode()
            ).hexdigest()[:32]
            output_path = run_root / "pages" / f"{generated_id}.png"
            checkpoint_status = checkpoints.queue_page(
                run_id=run_id,
                generated_page_id=generated_id,
                source_page_id=source_page_id,
                variant=variant,
                max_retries=max_retries,
            )
            existing_record = by_id.get(generated_id)
            if existing_record and existing_record.get("status") in {
                "complete",
                "manual_review",
                "quarantined",
                "skipped",
            }:
                _validate_resume_record(
                    existing_record,
                    roots=roots,
                    expected_output=output_path,
                    dry_run=dry_run,
                )
                checkpoints.reconcile_page(
                    generated_id,
                    status=str(existing_record["status"]),
                    output_uri=(
                        str(existing_record["output_uri"])
                        if existing_record.get("output_uri")
                        else None
                    ),
                    output_checksum=(
                        str(existing_record["output_checksum"])
                        if existing_record.get("output_checksum")
                        else None
                    ),
                    error=(
                        ";".join(
                            str(item)
                            for item in existing_record.get("warning_list", [])
                        )
                        or None
                    ),
                )
                continue
            if checkpoint_status in {
                "complete",
                "manual_review",
                "quarantined",
                "skipped",
                "quarantined",
            }:
                if not resume:
                    continue
                checkpoints.requeue_for_manifest_recovery(generated_id)
            if not checkpoints.claim_page(generated_id):
                if fail_fast:
                    raise RuntimeError(
                        f"Page {generated_id} cannot be claimed for processing"
                    )
                continue
            if output_path.exists() and not resume:
                raise FileExistsError(output_path)
            started = time.perf_counter()
            try:
                with Image.open(source_path) as opened:
                    opened.load()
                    source_dimensions = list(opened.size)
                    result, transformations = pipeline.run(
                        opened,
                        f"{source_page_id}:{variant}",
                        context={"regions": source.get("regions", [])},
                    )
            except Exception as exc:
                checkpoints.finish_page(
                    generated_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {str(exc)[:500]}",
                )
                if fail_fast:
                    raise
                continue
            record: dict[str, Any] = {
                "schema_version": 1,
                "run_id": run_id,
                "variant": variant,
                "generated_page_id": generated_id,
                "source_page_id": source_page_id,
                "source_document_id": source.get("source_document_id"),
                "dataset_id": dataset_id,
                "source_uri": source_uri,
                "source_checksum": _sha256(source_path),
                "output_uri": f"dataset://{output_path.relative_to(roots.dataset_root).as_posix()}",
                "output_checksum": None,
                "ground_truth_reference": source.get(
                    "ground_truth_reference", "synthetic://unchanged"
                ),
                "ground_truth_text": source.get("ground_truth_text"),
                "attribution": source.get("attribution", ""),
                "page_type": source.get("page_type", "body"),
                "language": source.get("language", "ar"),
                "regions": source.get("regions", []),
                "profile_id": profile["name"],
                "profile_version": profile.get("schema_version", 1),
                "profile_hash": profile_hash,
                "transformation_chain": [item.name for item in transformations],
                "transformation_parameters": [
                    item.parameters for item in transformations
                ],
                "parameters": [item.parameters for item in transformations],
                "seeds": [item.random_seed for item in transformations],
                "severity_per_transformation": [
                    item.severity for item in transformations
                ],
                "overall_severity": profile.get("severity", "medium"),
                "severity": profile.get("severity", "medium"),
                "layout_mode": (
                    "regions" if source.get("regions") else "whole_page_fallback"
                ),
                "affected_regions": [
                    region
                    for item in transformations
                    for region in item.output_metadata.get("affected_regions", [])
                ],
                "source_dimensions": source_dimensions,
                "output_dimensions": list(result.size),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "software_version": "clouda-pdf/0.2.0",
                "python_version": platform.python_version(),
                "pillow_version": Image.__version__,
                "host": {"system": platform.system(), "machine": platform.machine()},
                "license_status": license_status,
                "commercial_training_allowed": commercial_allowed,
                "status": "skipped" if dry_run else "processing",
                "retry_count": 0,
                "processing_seconds": time.perf_counter() - started,
                "warning_list": [],
                "warnings": [],
                "error": None,
            }
            try:
                if not dry_run:
                    free_threshold = max(
                        64 * 1024 * 1024,
                        int(
                            os.getenv(
                                "CLOUDA_MIN_FREE_DISK_BYTES",
                                str(512 * 1024 * 1024),
                            )
                        ),
                    )
                    if shutil.disk_usage(run_root).free < free_threshold:
                        raise OSError(
                            "Free disk space is below CLOUDA_MIN_FREE_DISK_BYTES"
                        )
                    if output_path.exists():
                        if not resume or not _matches_existing_image(
                            output_path, result
                        ):
                            raise FileExistsError(
                                f"Existing resume output does not match {output_path}"
                            )
                    else:
                        _atomic_save(result, output_path)
                    output_bytes += output_path.stat().st_size
                    if output_bytes > maximum_output_bytes:
                        output_path.unlink(missing_ok=True)
                        raise OSError("Run exceeded maximum output bytes")
                    record["output_checksum"] = _sha256(output_path)
                    quality = classify_visual_difficulty(
                        result, str(record["overall_severity"]), record
                    )
                    record.update(quality)
                    errors = _validate_asset(
                        record,
                        roots=roots,
                        source_path=source_path,
                        output_path=output_path,
                        image=result,
                    )
                    record["status"] = "manual_review" if errors else "complete"
                    record["warning_list"] = errors
                    record["warnings"] = errors
                else:
                    record["warning_list"] = ["dry_run"]
                    record["warnings"] = ["dry_run"]
            except Exception as exc:
                checkpoints.finish_page(
                    generated_id,
                    status="failed",
                    output_uri=str(record["output_uri"]),
                    error=f"{type(exc).__name__}: {str(exc)[:500]}",
                )
                if fail_fast:
                    raise
                continue
            records.append(record)
            by_id[generated_id] = record
            _write_jsonl_atomic(output_manifest, records)
            checkpoints.finish_page(
                generated_id,
                status=str(record["status"]),
                output_uri=str(record["output_uri"]),
                output_checksum=(
                    str(record["output_checksum"])
                    if record["output_checksum"] is not None
                    else None
                ),
                error=(
                    ";".join(str(item) for item in record["warning_list"])
                    if record["warning_list"]
                    else None
                ),
            )
            if interrupt_after is not None and len(records) >= interrupt_after:
                payload = json.loads(run_manifest.read_text(encoding="utf-8"))
                payload.update(
                    {
                        "state": "interrupted",
                        "interrupted_at": datetime.now(timezone.utc).isoformat(),
                        "records": len(records),
                    }
                )
                _write_json_atomic(run_manifest, payload)
                checkpoints.finish_run(run_id, "interrupted")
                raise RunInterrupted(
                    f"Intentional interruption after {len(records)} records"
                )
    status = (
        "complete"
        if all(
            item.get("status") in {"complete", "manual_review", "skipped"}
            for item in records
        )
        else "failed"
    )
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "state": status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "records": len(records),
        }
    )
    _write_json_atomic(run_manifest, payload)
    checkpoints.finish_run(run_id, status)
    if not dry_run:
        (run_root / "COMPLETE.v1.json").write_text(
            json.dumps(
                {"schema_version": 1, "run_id": run_id, "records": len(records)},
                indent=2,
            ),
            encoding="utf-8",
        )
    return output_manifest


def validate_distortion_manifest(
    manifest: str | Path,
    *,
    quarantine: bool = False,
) -> dict[str, Any]:
    roots = StorageRoots.from_env()
    path = Path(manifest).expanduser().resolve()
    if not _inside(path, roots.dataset_root):
        raise PermissionError("Distortion manifest must stay inside dataset root")
    records = read_jsonl(path)
    failures: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    for record in records:
        source = _dataset_uri_path(roots, record["source_uri"])
        output = _dataset_uri_path(roots, record["output_uri"])
        try:
            with Image.open(output) as image:
                image.load()
                errors = _validate_asset(
                    record,
                    roots=roots,
                    source_path=source,
                    output_path=output,
                    image=image,
                )
        except Exception as exc:
            errors = [f"decode_failed:{type(exc).__name__}"]
        if errors:
            failure: dict[str, Any] = {
                "generated_page_id": record["generated_page_id"],
                "errors": errors,
            }
            if quarantine and output.is_file():
                destination = (
                    roots.dataset_root / "quarantine" / path.parent.name / output.name
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(output, destination)
                failure["quarantine_uri"] = (
                    f"dataset://{destination.relative_to(roots.dataset_root).as_posix()}"
                )
                failure["quarantine_checksum"] = _sha256(destination)
            failures.append(failure)
            validation_records.append(
                {
                    **record,
                    "status": "quarantined" if quarantine else "failed",
                    "validation_errors": errors,
                    "quarantine_uri": failure.get("quarantine_uri"),
                }
            )
        else:
            validation_records.append(
                {**record, "validation_errors": [], "validated": True}
            )
    report = {
        "schema_version": 1,
        "records": len(records),
        "failures": failures,
        "passed": not failures,
    }
    report_root = roots.artifact_root / "reports" / "validation"
    report_root.mkdir(parents=True, exist_ok=True)
    stem = path.parent.name
    (report_root / f"{stem}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_jsonl_atomic(report_root / f"{stem}.jsonl", validation_records)
    with (report_root / f"{stem}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["generated_page_id", "errors"])
        for item in failures:
            writer.writerow(
                [
                    sanitize_spreadsheet_cell(item["generated_page_id"]),
                    sanitize_spreadsheet_cell(";".join(item["errors"])),
                ]
            )
    (report_root / f"{stem}.md").write_text(
        f"# Validation\n\n- Records: {len(records)}\n- Failures: {len(failures)}\n- Passed: {not failures}\n",
        encoding="utf-8",
    )
    return report


def generate_preview(
    manifest: str | Path,
    *,
    limit: int = 10,
    difference: bool = True,
    layout_overlay: bool = False,
) -> Path:
    roots = StorageRoots.from_env()
    manifest_path = Path(manifest).expanduser().resolve()
    if not _inside(manifest_path, roots.dataset_root):
        raise PermissionError("Preview manifest must stay inside dataset root")
    records = read_jsonl(manifest_path)[: max(1, min(limit, 100))]
    preview_root = roots.artifact_root / "previews" / manifest_path.parent.name
    preview_root.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    contact_images: list[tuple[str, Image.Image]] = []
    for record in records:
        source = _dataset_uri_path(roots, record["source_uri"])
        output = _dataset_uri_path(roots, record["output_uri"])
        generated_id = _safe_generated_id(record["generated_page_id"])
        with Image.open(source) as src_file, Image.open(output) as dst_file:
            src = src_file.convert("RGB")
            dst = dst_file.convert("RGB")
            if layout_overlay:
                overlay = ImageDraw.Draw(dst)
                for region in record.get("regions", []):
                    bbox = region.get("bbox") if isinstance(region, dict) else None
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        overlay.rectangle(
                            tuple(int(float(value)) for value in bbox),
                            outline=(220, 20, 60),
                            width=3,
                        )
            thumb_size = (640, 640)
            src.thumbnail(thumb_size)
            dst.thumbnail(thumb_size)
            canvas = Image.new(
                "RGB", (src.width + dst.width, max(src.height, dst.height)), "white"
            )
            canvas.paste(src, (0, 0))
            canvas.paste(dst, (src.width, 0))
            if difference and src.size == dst.size:
                diff = ImageChops.difference(src, dst)
                diff.save(preview_root / f"{generated_id}-diff.png")
            preview = preview_root / f"{generated_id}.jpg"
            canvas.save(preview, "JPEG", quality=85)
            contact = canvas.copy()
            contact.thumbnail((400, 260))
            contact_images.append((generated_id, contact))
        seeds = ", ".join(str(value) for value in record.get("seeds", []))
        rows.append(
            f"<tr><td>{html.escape(str(record['source_page_id']))}</td>"
            f"<td>{html.escape(str(record['generated_page_id']))}</td>"
            f"<td>{html.escape(str(record['profile_id']))}</td>"
            f"<td>{html.escape(str(record.get('overall_severity')))}</td>"
            f"<td>{html.escape(seeds)}</td>"
            f"<td><img loading='lazy' width='640' src='{preview.name}'></td></tr>"
        )
    if contact_images:
        columns = 4
        cell_width, cell_height = 420, 300
        selected = contact_images[:25]
        rows_count = math.ceil(len(selected) / columns)
        sheet = Image.new(
            "RGB", (cell_width * columns, cell_height * rows_count), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for contact_index, (identifier, contact) in enumerate(selected):
            x = (contact_index % columns) * cell_width
            y = (contact_index // columns) * cell_height
            sheet.paste(contact, (x + 10, y + 28))
            draw.text((x + 10, y + 8), identifier[:48], fill="black")
        sheet.save(preview_root / "contact-sheet.jpg", "JPEG", quality=85)
    document = (
        "<!doctype html><meta charset='utf-8'><title>Clouda distortion preview</title>"
        "<h1>Distortion preview</h1><table><thead><tr><th>Source</th><th>Generated</th>"
        "<th>Profile</th><th>Severity</th><th>Seeds</th><th>Preview</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    index_path = preview_root / "index.html"
    index_path.write_text(document, encoding="utf-8")
    (preview_root / "README.md").write_text(
        "# Preview index\n\n"
        "- [Contact sheet](contact-sheet.jpg)\n\n"
        + "\n".join(
            f"- `{record['generated_page_id']}` — {record['profile_id']}"
            for record in records
        ),
        encoding="utf-8",
    )
    return index_path
