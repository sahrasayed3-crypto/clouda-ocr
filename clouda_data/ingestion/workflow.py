from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import overload

from clouda_data.ground_truth.checksums import sha256_text

from .file_inspection import guess_source_type, inspect_file, text_from_ground_truth
from .manifest import write_page_manifest
from .registry import canonical_name, read_registry, register_file, write_registry
from .schema import PageRecord, SourcePage
from .source_manifest import read_source_manifest
from .validators import validate_page_record, validate_source_manifest


@dataclass(frozen=True)
class IngestionIssue:
    code: str
    message: str
    path: str = ""
    page_id: str = ""


@dataclass(frozen=True)
class IngestionPlan:
    manifest_path: str
    dry_run: bool
    ok: bool
    issues: list[IngestionIssue] = field(default_factory=list)
    pages: list[PageRecord] = field(default_factory=list)
    registrations: list[dict] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@overload
def _resolve(base: Path, value: str) -> Path: ...


@overload
def _resolve(base: Path, value: None) -> None: ...


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _inside_manifest_root(path: Path, manifest_root: Path) -> bool:
    try:
        path.resolve().relative_to(manifest_root.resolve())
        return True
    except ValueError:
        return False


def _page_ground_truth(page: SourcePage, manifest_dir: Path) -> tuple[str, str | None]:
    if page.clean_text is not None:
        return page.clean_text, None
    if not page.ground_truth_path:
        raise ValueError("missing_ground_truth")
    gt_path = _resolve(manifest_dir, page.ground_truth_path)
    assert gt_path is not None
    gt_type = guess_source_type(gt_path)
    return text_from_ground_truth(gt_path, gt_type), str(gt_path)


def validate_source_manifest_file(
    path: str | Path, project_root: Path
) -> IngestionPlan:
    manifest_path = Path(path).resolve()
    issues: list[IngestionIssue] = []
    pages: list[PageRecord] = []
    try:
        manifest = read_source_manifest(manifest_path)
        validate_source_manifest(manifest)
    except Exception as exc:
        return IngestionPlan(
            str(manifest_path),
            True,
            False,
            [IngestionIssue("invalid_manifest", str(exc))],
        )
    manifest_dir = manifest_path.parent
    seen_page_ids: set[str] = set()
    seen_checksums: dict[str, str] = {}
    for document in manifest.documents:
        doc_path = _resolve(manifest_dir, document.source_path)
        if doc_path is None:
            issues.append(
                IngestionIssue(
                    "missing_source_path",
                    "Document source_path is required.",
                    page_id=document.document_id,
                )
            )
            continue
        if not _inside_manifest_root(doc_path, manifest_dir):
            issues.append(
                IngestionIssue(
                    "unsafe_source_path",
                    "Document source_path must remain inside the manifest directory.",
                    str(doc_path),
                    document.document_id,
                )
            )
            continue
        inspection = inspect_file(doc_path, document.source_type)
        if not inspection.ok:
            issues.extend(
                IngestionIssue(
                    code,
                    f"Invalid document file: {code}",
                    str(doc_path),
                    document.document_id,
                )
                for code in inspection.issues
            )
        if inspection.checksum:
            prior_path = seen_checksums.get(inspection.checksum)
            if prior_path and prior_path != str(doc_path):
                issues.append(
                    IngestionIssue(
                        "duplicate_file",
                        "Duplicate source document checksum.",
                        str(doc_path),
                        document.document_id,
                    )
                )
            seen_checksums[inspection.checksum] = str(doc_path)
    for page in manifest.pages:
        if page.page_id in seen_page_ids:
            issues.append(
                IngestionIssue(
                    "duplicate_page_id", "Duplicate page_id.", page_id=page.page_id
                )
            )
        seen_page_ids.add(page.page_id)
        source_path = _resolve(manifest_dir, page.source_path)
        if source_path is None:
            issues.append(
                IngestionIssue(
                    "missing_source_path",
                    "Page source_path is required.",
                    page_id=page.page_id,
                )
            )
            continue
        if not _inside_manifest_root(source_path, manifest_dir):
            issues.append(
                IngestionIssue(
                    "unsafe_source_path",
                    "Page source_path must remain inside the manifest directory.",
                    str(source_path),
                    page.page_id,
                )
            )
            continue
        inspection = inspect_file(source_path, page.source_type)
        if not inspection.ok:
            issues.extend(
                IngestionIssue(
                    code, f"Invalid page file: {code}", str(source_path), page.page_id
                )
                for code in inspection.issues
            )
        if inspection.checksum:
            prior_path = seen_checksums.get(inspection.checksum)
            if prior_path and prior_path != str(source_path):
                issues.append(
                    IngestionIssue(
                        "duplicate_file",
                        "Duplicate source/page checksum.",
                        str(source_path),
                        page.page_id,
                    )
                )
            seen_checksums[inspection.checksum] = str(source_path)
        try:
            clean_text, gt_path = _page_ground_truth(page, manifest_dir)
        except Exception as exc:
            issues.append(
                IngestionIssue(
                    str(exc), "Missing or invalid ground truth.", page_id=page.page_id
                )
            )
            continue
        if not clean_text.strip():
            issues.append(
                IngestionIssue(
                    "empty_text", "Ground truth text is empty.", page_id=page.page_id
                )
            )
        actual_text_checksum = sha256_text(clean_text)
        if page.text_checksum and page.text_checksum != actual_text_checksum:
            issues.append(
                IngestionIssue(
                    "ground_truth_checksum_mismatch",
                    "Ground-truth checksum does not match extracted text.",
                    page_id=page.page_id,
                )
            )
        if page.ground_truth_path:
            gt_path_obj = _resolve(manifest_dir, page.ground_truth_path)
            if gt_path_obj and not _inside_manifest_root(gt_path_obj, manifest_dir):
                issues.append(
                    IngestionIssue(
                        "unsafe_source_path",
                        "Ground-truth path must remain inside the manifest directory.",
                        str(gt_path_obj),
                        page.page_id,
                    )
                )
                continue
            gt_inspection = inspect_file(gt_path_obj) if gt_path_obj else None
            if gt_inspection and not gt_inspection.ok:
                issues.extend(
                    IngestionIssue(
                        code,
                        f"Invalid ground-truth file: {code}",
                        str(gt_path_obj),
                        page.page_id,
                    )
                    for code in gt_inspection.issues
                )
        record = PageRecord(
            document_id=page.document_id,
            page_id=page.page_id,
            source_path=(
                str(source_path.relative_to(project_root))
                if source_path.is_relative_to(project_root)
                else str(source_path)
            ),
            source_type=page.source_type,
            language=page.language,
            page_number=page.page_number,
            clean_text=clean_text,
            text_checksum=actual_text_checksum,
            image_checksum=inspection.checksum if page.source_type == "image" else None,
            reading_order=page.reading_order,
            source_license=page.source_license,
        )
        try:
            validate_page_record(
                record,
                project_root if source_path.is_relative_to(project_root) else None,
            )
        except Exception as exc:
            issues.append(
                IngestionIssue(
                    "invalid_page_record", str(exc), str(source_path), page.page_id
                )
            )
        pages.append(record)
    return IngestionPlan(str(manifest_path), True, not issues, issues, pages)


def ingest_source_manifest(
    path: str | Path, project_root: Path, *, dry_run: bool
) -> IngestionPlan:
    validation = validate_source_manifest_file(path, project_root)
    if not validation.ok:
        return validation
    manifest = read_source_manifest(path)
    manifest_dir = Path(path).resolve().parent
    known = {
        item["checksum"]: item["canonical_path"]
        for item in read_registry(project_root / "data/manifests/file_registry.json")
    }
    registrations = []
    if dry_run:
        return IngestionPlan(
            str(Path(path).resolve()), True, True, pages=validation.pages
        )
    for document in manifest.documents:
        registrations.append(
            asdict(
                register_file(
                    _resolve(manifest_dir, document.source_path),
                    project_root=project_root,
                    role="document",
                    identifier=document.document_id,
                    source_type=document.source_type,
                    known_checksums=known,
                    copy=True,
                )
            )
        )
    for page in manifest.pages:
        registrations.append(
            asdict(
                register_file(
                    _resolve(manifest_dir, page.source_path),
                    project_root=project_root,
                    role="page",
                    identifier=page.page_id,
                    source_type=page.source_type,
                    known_checksums=known,
                    copy=True,
                )
            )
        )
        if page.ground_truth_path:
            registrations.append(
                asdict(
                    register_file(
                        _resolve(manifest_dir, page.ground_truth_path),
                        project_root=project_root,
                        role="ground_truth",
                        identifier=page.page_id,
                        source_type=guess_source_type(page.ground_truth_path) or "text",
                        known_checksums=known,
                        copy=True,
                    )
                )
            )
        if page.layout_path:
            registrations.append(
                asdict(
                    register_file(
                        _resolve(manifest_dir, page.layout_path),
                        project_root=project_root,
                        role="layout",
                        identifier=page.page_id,
                        source_type=guess_source_type(page.layout_path)
                        or "json_layout",
                        known_checksums=known,
                        copy=True,
                    )
                )
            )
    from .schema import FileRegistration

    existing_registrations = [
        FileRegistration(**item)
        for item in read_registry(project_root / "data/manifests/file_registry.json")
    ]
    write_registry(
        existing_registrations + [FileRegistration(**item) for item in registrations],
        project_root / "data/manifests/file_registry.json",
    )
    canonical_documents = [
        replace(
            document,
            source_path=str(
                (
                    Path("data/raw/documents")
                    / canonical_name(document.document_id, document.source_path)
                ).as_posix()
            ),
        )
        for document in manifest.documents
    ]
    canonical_pages = [
        replace(
            page,
            source_path=str(
                (
                    Path("data/raw/pages")
                    / canonical_name(page.page_id, page.source_path)
                ).as_posix()
            ),
        )
        for page in validation.pages
    ]
    Path(project_root / "data/manifests/source_document_manifest.json").write_text(
        json.dumps(
            [asdict(document) for document in canonical_documents],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_page_manifest(
        canonical_pages, project_root / "data/manifests/page_manifest.json"
    )
    report = IngestionPlan(
        str(Path(path).resolve()),
        False,
        True,
        pages=canonical_pages,
        registrations=registrations,
    )
    Path(project_root / "outputs/reports/ingestion_last_report.json").write_text(
        json.dumps(plan_to_dict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def plan_to_dict(plan: IngestionPlan) -> dict:
    return {
        "manifest_path": plan.manifest_path,
        "dry_run": plan.dry_run,
        "ok": plan.ok,
        "issues": [asdict(issue) for issue in plan.issues],
        "pages": [asdict(page) for page in plan.pages],
        "registrations": plan.registrations,
        "created_at": plan.created_at,
    }
