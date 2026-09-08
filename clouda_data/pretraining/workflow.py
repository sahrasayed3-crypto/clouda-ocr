"""End-to-end dataset preparation workflow and per-stage operations.

Workspace layout (all artifacts are plain files; nothing is copied except
tiny reports, so source data is never duplicated):

    <workspace>/
      sources.jsonl                    registered source definitions
      sources/<id>/index.jsonl         discovered files + hashes (resumable)
      cache/file_hashes.jsonl          incremental hash cache
      manifest/samples.v1.jsonl        canonical dataset manifest
      manifest/validation.json         validation report
      manifest/dedupe_report.json      duplicate classification report
      manifest/split_report.json       split + leakage report
      manifest/stats.json              dataset statistics
      export/<exporter>/<split>.jsonl  training-ready exports
      handoff/...                      optional data-factory handoff

Every stage is deterministic given (inputs, seed, config). ``dry_run``
computes everything in memory and writes nothing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PreparationConfig
from .dedupe import classify_duplicates
from .discovery import DiscoveredFile, draft_samples, scan_source
from .export import ExportConfig, get_exporter
from .hashing import HashCache, sha256_file, sha256_text
from .manifest import read_manifest, write_manifest
from .normalize import normalize_text
from .schema import DatasetSample, SplitName, sort_key, stable_sample_id
from .sources import (
    SourceDefinition,
    SourceRegistryError,
    get_source_definition,
    load_source_registry,
    register_source,
)
from .splitting import DEFAULT_RATIOS, assign_splits
from .validation import ValidationThresholds, apply_validation, set_thresholds

logger = logging.getLogger(__name__)

MANIFEST_REL = "manifest/samples.v1.jsonl"
STATS_SCHEMA_VERSION = "clouda.pretraining.stats.v1"

_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path

    @property
    def sources_file(self) -> Path:
        return self.root / "sources.jsonl"

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_REL

    @property
    def manifest_dir(self) -> Path:
        return self.root / "manifest"

    def source_index(self, source_id: str) -> Path:
        return self.root / "sources" / source_id / "index.jsonl"

    @property
    def hash_cache(self) -> Path:
        return self.root / "cache" / "file_hashes.jsonl"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def handoff_dir(self) -> Path:
        return self.root / "handoff"


def resolve_source(workspace: Path | str, value: str | Path) -> SourceDefinition:
    """Resolve a registered source id or a local directory path."""

    registry = load_source_registry(WorkspacePaths(Path(workspace)).sources_file)
    if Path(value).is_dir():
        root = Path(value).resolve()
        source_id = root.name or "local"
        for source in registry:
            if source.local_root and Path(source.local_root).resolve() == root:
                return source
        return SourceDefinition(
            source_id=source_id,
            name=root.name,
            local_root=str(root),
            adapter="auto",
        )
    for source in registry:
        if source.source_id == str(value):
            return source
    raise KeyError(
        f"{value!r} is neither a registered source id nor an existing directory."
    )


def _load_config(config: PreparationConfig | None) -> PreparationConfig:
    return config or PreparationConfig()


def _apply_thresholds(config: PreparationConfig) -> None:
    set_thresholds(
        ValidationThresholds(
            min_width=config.min_width,
            min_height=config.min_height,
            max_pixels=config.max_pixels,
            max_text_chars=config.max_text_chars,
            require_text=config.require_text,
            require_image=config.require_image,
        )
    )


def ensure_source_registered(workspace: Path | str, source: SourceDefinition) -> Path:
    paths = WorkspacePaths(Path(workspace))
    try:
        existing = get_source_definition(paths.sources_file, source.source_id)
    except KeyError:
        return register_source(paths.sources_file, source)
    if existing.to_dict() != source.to_dict():
        return register_source(paths.sources_file, source, replace=True)
    return paths.sources_file


def _compute_index_rows(
    source: SourceDefinition,
    previous: dict[str, tuple[int, str]],
    cache: HashCache | None,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    hashed_now = 0
    reused = 0
    for item in sorted(scan_source(source), key=lambda f: f.rel_path):
        cached = previous.get(item.rel_path)
        file_hash: str | None
        if cached and cached[0] == item.size_bytes:
            file_hash = cached[1]
            reused += 1
        else:
            file_hash = cache.get(item.rel_path, item.size_bytes) if cache else None
            if file_hash is None:
                file_hash = sha256_file(Path(source.local_root) / item.rel_path)
                hashed_now += 1
                if cache is not None:
                    cache.put(item.rel_path, item.size_bytes, file_hash)
            else:
                hashed_now += 1
        rows.append(
            {
                "source_id": item.source_id,
                "rel_path": item.rel_path,
                "kind": item.kind,
                "size_bytes": item.size_bytes,
                "file_sha256": file_hash,
            }
        )
    return rows, hashed_now, reused


def index_source(
    workspace: Path | str,
    source: SourceDefinition,
    config: PreparationConfig | None = None,
    *,
    resume: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan + hash a source into its index file (incremental, resumable)."""

    cfg = _load_config(config)
    paths = WorkspacePaths(Path(workspace))
    index_path = paths.source_index(source.source_id)
    previous: dict[str, tuple[int, str]] = {}
    if resume and index_path.exists():
        for row in _iter_index(index_path):
            previous[row["rel_path"]] = (row["size_bytes"], row["file_sha256"])

    cache = (
        HashCache(paths.hash_cache) if cfg.hash_cache_enabled and not dry_run else None
    )
    rows, hashed_now, reused = _compute_index_rows(source, previous, cache)

    report = {
        "source_id": source.source_id,
        "files": len(rows),
        "hashed_now": hashed_now,
        "reused_from_index": reused,
        "dry_run": dry_run,
    }
    if not dry_run:
        _write_index(index_path, rows)
    return report


def _iter_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "rel_path" in payload:
            rows.append(payload)
    return rows


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        header = {
            "_schema_version": "clouda.pretraining.index.v1",
            "_row_count": len(rows),
        }
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _guess_script(text: str | None) -> str:
    if not text:
        return "unknown"
    arabic = sum(
        1 for ch in text if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f"
    )
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if arabic and latin:
        return "mixed"
    if arabic:
        return "arabic"
    if latin:
        return "latin"
    return "unknown"


def _image_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:  # noqa: BLE001 - dimensions unknown -> validation decides
        return None, None


def _build_samples(
    source: SourceDefinition,
    index_rows: list[dict[str, Any]],
    cfg: PreparationConfig,
) -> list[DatasetSample]:
    files = [
        DiscoveredFile(
            source_id=row["source_id"],
            rel_path=row["rel_path"],
            kind=row["kind"],
            size_bytes=row["size_bytes"],
        )
        for row in index_rows
    ]
    drafts = draft_samples(
        source,
        files,
        text_without_image_policy=cfg.text_without_image_policy,
    )
    hash_by_rel = {row["rel_path"]: row["file_sha256"] for row in index_rows}
    root = Path(source.local_root)
    language = source.languages[0] if source.languages else "ar"

    samples: list[DatasetSample] = []
    for draft in drafts:
        primary_hash = hash_by_rel.get(draft.source_path)
        raw_text = draft.raw_text
        applied: tuple[str, ...] = ()
        normalized: str | None = None
        if raw_text is not None:
            result = normalize_text(raw_text, cfg.normalization)
            normalized, applied = result.value, result.applied
        width = height = None
        if draft.image_rel_path:
            width, height = _image_dimensions(root / draft.image_rel_path)
        provenance = {
            "source_root": str(root),
            "adapter": source.adapter,
            "classification": source.classification,
            "license": source.license,
            "encoding_ok": draft.encoding_ok,
            "malformed_metadata": draft.malformed_metadata,
            "record_file": (draft.notes or {}).get("record_file"),
            "text_without_image": (draft.notes or {}).get("text_without_image"),
        }
        provenance = {k: v for k, v in provenance.items() if v is not None}
        samples.append(
            DatasetSample(
                sample_id=(
                    draft.sample_id
                    if draft.sample_id
                    else stable_sample_id(source.source_id, draft.source_path)
                ),
                source_id=source.source_id,
                source_dataset=source.name,
                source_path=draft.source_path,
                source_record_id=draft.source_record_id,
                source_license=source.license,
                document_id=draft.document_id,
                page_id=draft.page_id,
                page_index=draft.page_index,
                group_id=(
                    f"{source.source_id}:{draft.document_id}"
                    if draft.document_id
                    else None
                ),
                image_path=draft.image_rel_path,
                text=normalized,
                raw_text=raw_text,
                language=language,
                script=_guess_script(raw_text or normalized),
                width=width,
                height=height,
                file_size=draft.file_size,
                file_extension=draft.file_extension,
                mime_type=_MIME_BY_EXTENSION.get(draft.file_extension or ""),
                file_sha256=primary_hash,
                normalized_text_sha256=(
                    sha256_text(normalized) if normalized is not None else None
                ),
                provenance=provenance,
                transformations=list(applied),
            )
        )
    samples.sort(key=sort_key)
    return samples


def _write_report(
    paths: WorkspacePaths, name: str, payload: dict[str, Any], dry_run: bool
) -> None:
    if dry_run:
        return
    target = paths.manifest_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(target)


def prepare_dataset(
    workspace: Path | str,
    source_value: str | Path,
    config: PreparationConfig | None = None,
    *,
    seed: int | None = None,
    dry_run: bool = False,
    resume: bool = True,
    write_handoff: bool = False,
    handoff_profiles: list[str] | None = None,
    intended_output: str | None = None,
) -> dict[str, Any]:
    """Run the full preparation pipeline for one source."""

    cfg = _load_config(config)
    _apply_thresholds(cfg)
    root = Path(workspace)
    paths = WorkspacePaths(root)
    source = resolve_source(root, source_value)
    if not source.enabled:
        raise SourceRegistryError(f"Source is disabled: {source.source_id}")
    ensure_source_registered(root, source)

    index_report = index_source(root, source, cfg, resume=resume, dry_run=dry_run)
    if dry_run:
        index_rows, _hashed, _reused = _compute_index_rows(source, {}, None)
    else:
        index_rows = _iter_index(paths.source_index(source.source_id))
    samples = _build_samples(source, index_rows, cfg)

    samples, validation_report = apply_validation(samples, Path(source.local_root))
    samples, dedupe_report = classify_duplicates(samples)
    samples, split_report = assign_splits(
        samples,
        seed=seed if seed is not None else cfg.split_seed,
        ratios=cfg.split_ratios,
    )

    rows = [sample.to_dict() for sample in samples]
    if not dry_run:
        # Accumulate across sources: keep other sources' rows, replace this one.
        existing_header, existing_rows = read_manifest(paths.manifest)
        kept = [
            row for row in existing_rows if row.get("source_id") != source.source_id
        ]
        merged = kept + rows
        merged.sort(
            key=lambda row: (
                row.get("source_id", ""),
                row.get("source_path", ""),
                row.get("sample_id", ""),
            )
        )
        write_manifest(paths.manifest, merged)
    _write_report(paths, "validation.json", validation_report, dry_run)
    _write_report(paths, "dedupe_report.json", dedupe_report.to_dict(), dry_run)
    _write_report(paths, "split_report.json", split_report.to_dict(), dry_run)
    stats = compute_stats(rows)
    _write_report(paths, "stats.json", stats, dry_run)

    export_result: dict[str, Any] | None = None
    if not dry_run:
        export_result = export_workspace(root, cfg).to_dict()

    handoff: dict[str, Any] | None = None
    if write_handoff and not dry_run:
        from .handoff import build_data_factory_handoff

        _handoff, _candidates, request_path = build_data_factory_handoff(
            Path(source.local_root),
            root,
            samples,
            requested_profiles=handoff_profiles or [],
            seed=seed if seed is not None else cfg.split_seed,
            intended_output=intended_output or str(paths.export_dir),
            dataset_manifest_path=paths.manifest,
        )
        handoff = {"request": str(request_path)}

    return {
        "source_id": source.source_id,
        "dry_run": dry_run,
        "index": index_report,
        "samples": len(rows),
        "validation": validation_report,
        "dedupe": dedupe_report.to_dict()["counts"],
        "split": split_report.to_dict(),
        "stats": stats,
        "export": export_result,
        "handoff": handoff,
        "manifest": str(paths.manifest),
        "config_version": "clouda.pretraining.config.v1",
    }


def _load_manifest_samples(paths: WorkspacePaths) -> list[DatasetSample]:
    _, rows = read_manifest(paths.manifest)
    return [DatasetSample.from_dict(row) for row in rows]


def validate_workspace(
    workspace: Path | str, config: PreparationConfig | None = None
) -> dict[str, Any]:
    cfg = _load_config(config)
    _apply_thresholds(cfg)
    paths = WorkspacePaths(Path(workspace))
    source = resolve_source(paths.root, _sole_source_id(paths))
    samples = _load_manifest_samples(paths)
    samples, report = apply_validation(samples, Path(source.local_root))
    write_manifest(paths.manifest, [sample.to_dict() for sample in samples])
    _write_report(paths, "validation.json", report, dry_run=False)
    return report


def normalize_workspace(
    workspace: Path | str, config: PreparationConfig | None = None
) -> dict[str, Any]:
    """Re-apply the normalization policy; raw text is never overwritten."""

    cfg = _load_config(config)
    paths = WorkspacePaths(Path(workspace))
    samples = _load_manifest_samples(paths)
    changed = 0
    updated: list[DatasetSample] = []
    for sample in samples:
        if sample.raw_text is None:
            updated.append(sample)
            continue
        result = normalize_text(sample.raw_text, cfg.normalization)
        text_hash = sha256_text(result.value) if result.value else None
        if result.value != (sample.text or ""):
            changed += 1
        updated.append(
            sample.evolve(
                text=result.value,
                normalized_text_sha256=text_hash,
                transformations=sorted(
                    set(sample.transformations) | set(result.applied)
                ),
            )
        )
    write_manifest(paths.manifest, [sample.to_dict() for sample in updated])
    report = {
        "samples": len(updated),
        "normalized_now": changed,
        "policy_version": cfg.normalization.version(),
        "schema_version": "clouda.pretraining.normalize_report.v1",
    }
    _write_report(paths, "normalize_report.json", report, dry_run=False)
    return report


def dedupe_workspace(workspace: Path | str) -> dict[str, Any]:
    paths = WorkspacePaths(Path(workspace))
    samples = _load_manifest_samples(paths)
    samples, report = classify_duplicates(samples)
    write_manifest(paths.manifest, [sample.to_dict() for sample in samples])
    _write_report(paths, "dedupe_report.json", report.to_dict(), dry_run=False)
    return report.to_dict()


def split_workspace(
    workspace: Path | str,
    *,
    seed: int | None = None,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    paths = WorkspacePaths(Path(workspace))
    samples = _load_manifest_samples(paths)
    samples, report = assign_splits(
        samples,
        seed=seed if seed is not None else PreparationConfig().split_seed,
        ratios=ratios or DEFAULT_RATIOS,
    )
    write_manifest(paths.manifest, [sample.to_dict() for sample in samples])
    _write_report(paths, "split_report.json", report.to_dict(), dry_run=False)
    return report.to_dict()


def export_workspace(
    workspace: Path | str, config: PreparationConfig | None = None
) -> Any:
    cfg = _load_config(config)
    paths = WorkspacePaths(Path(workspace))
    samples = _load_manifest_samples(paths)
    exporter = get_exporter(cfg.exporter)
    export_config = ExportConfig(
        include_holdout=cfg.include_holdout_in_export,
        include_duplicates=cfg.include_duplicates_in_export,
        include_raw_text=cfg.include_raw_text_in_export,
        splits=(
            ("train", "validation", "test", "holdout")
            if cfg.include_holdout_in_export
            else ("train", "validation", "test")
        ),
    )
    result = exporter.export(samples, paths.export_dir / exporter.name, export_config)
    return result


def compute_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total_samples": len(rows),
        "by_validation_status": {},
        "by_duplicate_state": {},
        "by_split": {},
        "by_source": {},
        "by_script": {},
        "findings_by_severity": {},
        "total_source_bytes": 0,
        "excluded_samples": 0,
        "holdout_samples": 0,
        "schema_version": STATS_SCHEMA_VERSION,
    }
    for row in rows:
        stats["by_validation_status"][row["validation_status"]] = (
            stats["by_validation_status"].get(row["validation_status"], 0) + 1
        )
        stats["by_duplicate_state"][row["duplicate_state"]] = (
            stats["by_duplicate_state"].get(row["duplicate_state"], 0) + 1
        )
        stats["by_split"][row["target_split"]] = (
            stats["by_split"].get(row["target_split"], 0) + 1
        )
        stats["by_source"][row["source_id"]] = (
            stats["by_source"].get(row["source_id"], 0) + 1
        )
        stats["by_script"][row.get("script") or "unknown"] = (
            stats["by_script"].get(row.get("script") or "unknown", 0) + 1
        )
        for finding in row.get("validation_findings", []):
            severity = finding.get("severity", "info")
            stats["findings_by_severity"][severity] = (
                stats["findings_by_severity"].get(severity, 0) + 1
            )
        if row.get("file_size"):
            stats["total_source_bytes"] += row["file_size"]
        if row["validation_status"] in ("error", "excluded"):
            stats["excluded_samples"] += 1
        if row["target_split"] == SplitName.HOLDOUT.value:
            stats["holdout_samples"] += 1
    return stats


def stats_workspace(workspace: Path | str) -> dict[str, Any]:
    paths = WorkspacePaths(Path(workspace))
    _, rows = read_manifest(paths.manifest)
    return compute_stats(rows)


def _sole_source_id(paths: WorkspacePaths) -> str:
    ids = sorted({sample.source_id for sample in _load_manifest_samples(paths)})
    if not ids:
        raise FileNotFoundError("Dataset manifest is empty or missing.")
    return ids[0]
