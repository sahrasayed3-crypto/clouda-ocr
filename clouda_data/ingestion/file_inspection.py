from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defusedxml import ElementTree

from clouda_contracts.archive_security import ArchiveLimits, validate_zip_archive
from clouda_data.ground_truth.checksums import sha256_file, sha256_text

from .schema import SourceType

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS: dict[str, SourceType] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text",
    ".json": "ground_truth_json",
    ".xml": "page_xml",
    **{ext: "image" for ext in IMAGE_EXTENSIONS},
}


@dataclass(frozen=True)
class SourceInspection:
    path: str
    exists: bool
    source_type: SourceType | None
    size_bytes: int
    checksum: str | None
    ok: bool
    issues: list[str]


def guess_source_type(path: str | Path) -> SourceType | None:
    return SUPPORTED_EXTENSIONS.get(Path(path).suffix.lower())


def _is_pdf(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def _is_docx(path: Path) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            validate_zip_archive(
                archive,
                limits=ArchiveLimits(
                    max_members=5_000,
                    max_total_uncompressed_bytes=200 * 1024 * 1024,
                    max_member_uncompressed_bytes=50 * 1024 * 1024,
                    max_compression_ratio=100,
                ),
            )
            return "[Content_Types].xml" in archive.namelist() and any(
                name.startswith("word/") for name in archive.namelist()
            )
    except (ValueError, zipfile.BadZipFile):
        return False


def _is_image(path: Path) -> bool:
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = 40_000_000
        with Image.open(path) as image:
            if image.width * image.height > 40_000_000:
                return False
            image.verify()
        return True
    except Exception:
        return False


def _is_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _is_xml(path: Path) -> bool:
    try:
        ElementTree.parse(path)
        return True
    except Exception:
        return False


def inspect_file(
    path: str | Path, expected_type: SourceType | None = None
) -> SourceInspection:
    source_path = Path(path)
    if not source_path.exists():
        return SourceInspection(
            str(source_path),
            False,
            expected_type or guess_source_type(source_path),
            0,
            None,
            False,
            ["missing_file"],
        )
    if not source_path.is_file():
        return SourceInspection(
            str(source_path),
            True,
            expected_type or guess_source_type(source_path),
            0,
            None,
            False,
            ["not_a_file"],
        )
    source_type = expected_type or guess_source_type(source_path)
    issues: list[str] = []
    if source_type is None:
        issues.append("unsupported_format")
    size = source_path.stat().st_size
    if size == 0:
        issues.append("empty_file")
    if source_type == "pdf" and not _is_pdf(source_path):
        issues.append("corrupt_pdf")
    elif source_type == "docx" and not _is_docx(source_path):
        issues.append("corrupt_docx")
    elif source_type == "image" and not _is_image(source_path):
        issues.append("corrupt_image")
    elif source_type in {"ground_truth_json", "json_layout"} and not _is_json(
        source_path
    ):
        issues.append("invalid_json")
    elif source_type in {"page_xml", "alto_xml"} and not _is_xml(source_path):
        issues.append("invalid_xml")
    elif source_type == "text":
        try:
            source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append("text_not_utf8")
    checksum = sha256_file(source_path) if source_path.is_file() and size > 0 else None
    return SourceInspection(
        str(source_path), True, source_type, size, checksum, not issues, issues
    )


def text_from_ground_truth(
    path: str | Path, source_type: SourceType | None = None
) -> str:
    gt_path = Path(path)
    resolved_type = source_type or guess_source_type(gt_path)
    if resolved_type == "text":
        return gt_path.read_text(encoding="utf-8")
    if resolved_type == "ground_truth_json":
        data: Any = json.loads(gt_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ["clean_text", "text", "ground_truth"]:
                if isinstance(data.get(key), str):
                    return data[key]
            if isinstance(data.get("pages"), list):
                return "\n".join(
                    str(page.get("clean_text", ""))
                    for page in data["pages"]
                    if isinstance(page, dict)
                )
        raise ValueError(
            "Structured JSON ground truth must contain clean_text, text, ground_truth, or pages[].clean_text."
        )
    if resolved_type == "page_xml":
        root = ElementTree.parse(gt_path).getroot()
        texts = []
        for element in root.iter():
            if (
                element.tag.rsplit("}", 1)[-1] == "Unicode"
                and element.text
                and element.text.strip()
            ):
                texts.append(element.text.strip())
        return " ".join(texts)
    if resolved_type == "alto_xml":
        root = ElementTree.parse(gt_path).getroot()
        texts = [
            element.attrib["CONTENT"]
            for element in root.iter()
            if element.attrib.get("CONTENT")
        ]
        return " ".join(texts)
    raise ValueError(f"Cannot extract ground truth text from {resolved_type}.")


def checksum_for_text_or_file(text: str | None, path: str | Path | None) -> str:
    return sha256_text(text) if text is not None else sha256_file(Path(path))  # type: ignore[arg-type]
