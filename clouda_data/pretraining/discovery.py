"""Deterministic local file discovery and sample drafting.

The scanner walks source directories in sorted order (stable across runs
and machines for the same tree), skips junk, and never loads large files
into memory. Drafting pairs images with sidecar text and adapts record
files (JSONL/CSV/TSV) into samples. Symlinked directories are not followed.
"""

from __future__ import annotations

import csv
import json
import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .sources import SourceDefinition

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
TEXT_EXTENSIONS = {".txt"}
RECORD_EXTENSIONS = {".jsonl", ".json", ".csv", ".tsv"}
REFERENCE_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = (
    IMAGE_EXTENSIONS | TEXT_EXTENSIONS | RECORD_EXTENSIONS | REFERENCE_EXTENSIONS
)

JUNK_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
}
JUNK_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}

# ``book-p03`` / ``doc_12`` style stems become document ``book`` page 3.
PAGE_STEM_RE = re.compile(r"^(?P<base>.+?)[-_]p?(?P<page>\d{1,4})$", re.IGNORECASE)

JSONL_IMAGE_FIELDS = ("image", "image_path", "file_name", "image_file")
JSONL_TEXT_FIELDS = ("text", "ground_truth", "label", "transcript")
RECORD_ID_FIELDS = ("id", "record_id", "sample_id")
MAX_RECORD_LINE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DiscoveredFile:
    source_id: str
    rel_path: str  # posix-style, relative to the source local root
    kind: str  # image | text | record | reference
    size_bytes: int
    mtime_ns: int


@dataclass
class SampleDraft:
    """A candidate sample before hashing/validation/normalization."""

    source_id: str
    source_path: str  # posix rel path of the primary file inside the source
    sample_id: str
    image_rel_path: str | None = None
    text_rel_path: str | None = None
    raw_text: str | None = None
    source_record_id: str | None = None
    document_id: str | None = None
    page_id: str | None = None
    page_index: int | None = None
    file_size: int | None = None
    file_extension: str | None = None
    encoding_ok: bool = True
    malformed_metadata: bool = False
    notes: dict[str, object] | None = None


def classify_kind(suffix: str) -> str:
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in RECORD_EXTENSIONS:
        return "record"
    if suffix in REFERENCE_EXTENSIONS:
        return "reference"
    return "unknown"


def scan_source(
    source: SourceDefinition, *, allowed_extensions: frozenset[str] | None = None
) -> list[DiscoveredFile]:
    """Recursively discover files under a source root, deterministically."""

    if not source.local_root:
        return []
    root = Path(source.local_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {root}")
    discovered: list[DiscoveredFile] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in JUNK_DIRECTORIES and not (Path(dirpath) / name).is_symlink()
        )
        for name in sorted(filenames):
            if name in JUNK_FILES:
                continue
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            from .schema import canonical_relative_path

            try:
                rel = PurePosixPath(
                    canonical_relative_path(path.relative_to(root).as_posix())
                )
            except ValueError:
                # Invisible direction-changing characters and other unsafe
                # provenance names must never enter manifests or exports.
                continue
            if (
                allowed_extensions is not None
                and rel.suffix.lower() not in allowed_extensions
            ):
                continue
            kind = classify_kind(rel.suffix.lower())
            if kind == "unknown":
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            discovered.append(
                DiscoveredFile(
                    source_id=source.source_id,
                    rel_path=rel.as_posix(),
                    kind=kind,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
    discovered.sort(key=lambda item: item.rel_path)
    return discovered


def split_document_pages(stem: str) -> tuple[str, int | None]:
    match = PAGE_STEM_RE.match(stem)
    if match:
        return match.group("base"), int(match.group("page"))
    return stem, None


def _read_text_bytes(path: Path) -> tuple[str | None, bool]:
    try:
        data = path.read_bytes()
    except OSError:
        return None, False
    try:
        return data.decode("utf-8"), True
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), False


def _pair_sidecar(
    files_by_rel: dict[str, DiscoveredFile], image: DiscoveredFile
) -> DiscoveredFile | None:
    stem = PurePosixPath(image.rel_path).stem
    parent = PurePosixPath(image.rel_path).parent
    for ext in sorted(TEXT_EXTENSIONS):
        candidate = (parent / f"{stem}{ext}").as_posix()
        if candidate in files_by_rel:
            return files_by_rel[candidate]
    return None


def _document_parts(rel_path: str) -> tuple[str, int | None, str]:
    pure = PurePosixPath(rel_path)
    document_id, page_index = split_document_pages(pure.stem)
    page_id = pure.stem
    return document_id, page_index, page_id


def _draft_from_image(
    source: SourceDefinition,
    image: DiscoveredFile,
    sidecar: DiscoveredFile | None,
    root: Path,
    files_by_rel: dict[str, DiscoveredFile],
) -> SampleDraft:
    document_id, page_index, page_id = _document_parts(image.rel_path)
    raw_text: str | None = None
    encoding_ok = True
    if sidecar is not None:
        raw_text, encoding_ok = _read_text_bytes(root / sidecar.rel_path)
    record_key = sidecar.rel_path if sidecar else ""
    from .schema import stable_sample_id

    return SampleDraft(
        source_id=source.source_id,
        source_path=image.rel_path,
        sample_id=stable_sample_id(source.source_id, image.rel_path, record_key),
        image_rel_path=image.rel_path,
        text_rel_path=sidecar.rel_path if sidecar else None,
        raw_text=raw_text,
        document_id=document_id,
        page_id=page_id,
        page_index=page_index,
        file_size=image.size_bytes,
        file_extension=PurePosixPath(image.rel_path).suffix.lower(),
        encoding_ok=encoding_ok,
    )


def _jsonl_field(record: dict[str, object], fields: tuple[str, ...]) -> object:
    for name in fields:
        if name in record:
            return record[name]
    return None


def _strict_json_object(line: str) -> object:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(line, object_pairs_hook=reject_duplicate_keys)


def _record_image_path(value: object, record_dir: PurePosixPath, root: Path) -> str:
    from .schema import canonical_relative_path

    if not isinstance(value, str):
        raise ValueError("record image path must be a string")
    record_relative = canonical_relative_path(value)
    joined = posixpath.join(record_dir.as_posix(), record_relative)
    relative = canonical_relative_path(joined)
    resolved_root = root.resolve()
    resolved_candidate = (root / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("record image path resolves outside source root") from exc
    return relative


def _record_identity(record: dict[str, object], fallback: str) -> str:
    value = _jsonl_field(record, RECORD_ID_FIELDS)
    if isinstance(value, (str, int)) and str(value):
        return f"{fallback.rsplit('#', 1)[0]}#id={value}"
    return fallback


def _malformed_record_draft(
    source: SourceDefinition,
    record_file: DiscoveredFile,
    record_key: str,
    *,
    raw_text: str | None = None,
    encoding_ok: bool = True,
    reason: str | None = None,
) -> SampleDraft:
    from .schema import stable_sample_id

    notes: dict[str, object] = {"record_file": record_file.rel_path}
    if reason:
        notes["malformed_reason"] = reason
    return SampleDraft(
        source_id=source.source_id,
        source_path=record_file.rel_path,
        sample_id=stable_sample_id(source.source_id, record_file.rel_path, record_key),
        source_record_id=record_key,
        raw_text=raw_text,
        encoding_ok=encoding_ok,
        malformed_metadata=True,
        file_size=record_file.size_bytes,
        file_extension=PurePosixPath(record_file.rel_path).suffix.lower(),
        notes=notes,
    )


def _drafts_from_records(
    source: SourceDefinition,
    record_file: DiscoveredFile,
    root: Path,
    files_by_rel: dict[str, DiscoveredFile],
) -> list[SampleDraft]:
    from .schema import stable_sample_id

    drafts: list[SampleDraft] = []
    path = root / record_file.rel_path
    record_dir = PurePosixPath(record_file.rel_path).parent

    if record_file.rel_path.lower().endswith(".jsonl"):
        with path.open("rb") as handle:
            lines = enumerate(handle, start=1)
            for line_number, raw_line in lines:
                if not raw_line.strip():
                    continue
                record_key = f"{record_file.rel_path}#L{line_number}"
                if len(raw_line) > MAX_RECORD_LINE_BYTES:
                    drafts.append(
                        _malformed_record_draft(
                            source,
                            record_file,
                            record_key,
                            reason="record line exceeds safe parsing limit",
                        )
                    )
                    continue
                try:
                    line = raw_line.decode("utf-8")
                    encoding_ok = True
                except UnicodeDecodeError:
                    line = raw_line.decode("utf-8", errors="replace")
                    encoding_ok = False
                try:
                    record = _strict_json_object(line)
                except (json.JSONDecodeError, ValueError) as exc:
                    drafts.append(
                        _malformed_record_draft(
                            source,
                            record_file,
                            record_key,
                            encoding_ok=encoding_ok,
                            reason=str(exc),
                        )
                    )
                    continue
                if not isinstance(record, dict):
                    drafts.append(
                        _malformed_record_draft(
                            source,
                            record_file,
                            record_key,
                            encoding_ok=encoding_ok,
                            reason="record must be a JSON object",
                        )
                    )
                    continue
                record_key = _record_identity(record, record_key)
                image_value = _jsonl_field(record, JSONL_IMAGE_FIELDS)
                text_value = _jsonl_field(record, JSONL_TEXT_FIELDS)
                try:
                    image_rel = (
                        _record_image_path(image_value, record_dir, root)
                        if image_value is not None
                        else None
                    )
                except ValueError as exc:
                    drafts.append(
                        _malformed_record_draft(
                            source,
                            record_file,
                            record_key,
                            raw_text=(
                                str(text_value) if text_value is not None else None
                            ),
                            encoding_ok=encoding_ok,
                            reason=str(exc),
                        )
                    )
                    continue
                sidecar = files_by_rel.get(image_rel) if image_rel else None
                image_size = sidecar.size_bytes if sidecar else None
                drafts.append(
                    SampleDraft(
                        source_id=source.source_id,
                        source_path=image_rel or record_file.rel_path,
                        sample_id=stable_sample_id(
                            source.source_id, record_file.rel_path, record_key
                        ),
                        image_rel_path=image_rel,
                        raw_text=(str(text_value) if text_value is not None else None),
                        source_record_id=record_key,
                        document_id=(
                            PurePosixPath(image_rel).stem if image_rel else None
                        ),
                        page_id=(PurePosixPath(image_rel).stem if image_rel else None),
                        file_size=image_size,
                        file_extension=(
                            PurePosixPath(image_rel).suffix.lower()
                            if image_rel
                            else ".jsonl"
                        ),
                        encoding_ok=encoding_ok,
                        notes={"record_file": record_file.rel_path},
                    )
                )
        return drafts

    if record_file.rel_path.lower().endswith((".csv", ".tsv")):
        delimiter = "\t" if record_file.rel_path.lower().endswith(".tsv") else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for index, row in enumerate(reader):
                if row is None:
                    continue
                record_key = f"{record_file.rel_path}#R{index + 1}"
                image_value = _jsonl_field(row, JSONL_IMAGE_FIELDS)
                text_value = _jsonl_field(row, JSONL_TEXT_FIELDS)
                try:
                    image_rel = (
                        _record_image_path(image_value, record_dir, root)
                        if image_value is not None
                        else None
                    )
                except ValueError as exc:
                    drafts.append(
                        _malformed_record_draft(
                            source,
                            record_file,
                            record_key,
                            raw_text=(
                                str(text_value) if text_value is not None else None
                            ),
                            reason=str(exc),
                        )
                    )
                    continue
                sidecar = files_by_rel.get(image_rel) if image_rel else None
                drafts.append(
                    SampleDraft(
                        source_id=source.source_id,
                        source_path=image_rel or record_file.rel_path,
                        sample_id=stable_sample_id(
                            source.source_id, record_key, str(image_rel or "")
                        ),
                        image_rel_path=image_rel if sidecar else None,
                        raw_text=str(text_value) if text_value is not None else None,
                        source_record_id=record_key,
                        document_id=(
                            PurePosixPath(image_rel).stem if image_rel else None
                        ),
                        page_id=(PurePosixPath(image_rel).stem if image_rel else None),
                        file_size=sidecar.size_bytes if sidecar else None,
                        file_extension=(
                            PurePosixPath(image_rel).suffix.lower()
                            if image_rel
                            else ".csv"
                        ),
                        notes={"record_file": record_file.rel_path},
                    )
                )
        return drafts

    return drafts


def draft_samples(
    source: SourceDefinition,
    files: list[DiscoveredFile],
    *,
    text_without_image_policy: str = "exclude",
) -> list[SampleDraft]:
    """Turn discovered files into candidate samples.

    ``text_without_image_policy``: ``exclude`` (default) skips standalone
    text files with no paired image; ``keep`` drafts them as image-less
    samples (they will fail OCR training validation and be excluded there).
    """

    root = Path(source.local_root)
    files_by_rel = {item.rel_path: item for item in files}
    drafts: list[SampleDraft] = []
    images = [item for item in files if item.kind == "image"]
    records = [item for item in files if item.kind == "record"]
    paired_text: set[str] = set()

    for image in images:
        sidecar = _pair_sidecar(files_by_rel, image)
        if sidecar is not None:
            paired_text.add(sidecar.rel_path)
        drafts.append(_draft_from_image(source, image, sidecar, root, files_by_rel))

    for record in records:
        if record.rel_path.lower().endswith(".json"):
            continue  # json sidecar metadata is not a sample source here
        drafts.extend(_drafts_from_records(source, record, root, files_by_rel))

    if text_without_image_policy == "keep":
        for item in files:
            if item.kind != "text" or item.rel_path in paired_text:
                continue
            raw_text, encoding_ok = _read_text_bytes(root / item.rel_path)
            from .schema import stable_sample_id

            drafts.append(
                SampleDraft(
                    source_id=source.source_id,
                    source_path=item.rel_path,
                    sample_id=stable_sample_id(source.source_id, item.rel_path),
                    raw_text=raw_text,
                    document_id=PurePosixPath(item.rel_path).stem,
                    file_size=item.size_bytes,
                    file_extension=PurePosixPath(item.rel_path).suffix.lower(),
                    encoding_ok=encoding_ok,
                    notes={"text_without_image": True},
                )
            )

    drafts.sort(key=lambda draft: (draft.source_path, draft.sample_id))
    return drafts
