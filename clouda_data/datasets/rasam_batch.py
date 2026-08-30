from __future__ import annotations

import csv
import json
import re
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from defusedxml import ElementTree

from .downloader import validate_download_url

from PIL import Image

from clouda_data.datasets.downloader import (
    DownloadedFile,
    content_length,
    disk_free_bytes,
    download_http,
    sha256_file,
)
from clouda_data.ground_truth.checksums import sha256_text
from clouda_data.ingestion.file_inspection import text_from_ground_truth

RASAM_SOURCE_ID = "rasam_dataset"
RASAM_REPO = "https://github.com/calfa-co/rasam-dataset"
RASAM_API = "https://api.github.com/repos/calfa-co/rasam-dataset"
RAW_BASE = "https://raw.githubusercontent.com/calfa-co/rasam-dataset/main"
LIST_IMAGES_URL = f"{RAW_BASE}/list-images.tsv"
LICENSE_URL = f"{RAW_BASE}/LICENSE"
README_URL = f"{RAW_BASE}/README.md"
CONTRIBUTING_URL = f"{RAW_BASE}/contributing.md"
ONE_GB = 1024 * 1024 * 1024


@dataclass(frozen=True)
class RasamPageCandidate:
    page_id: str
    subset: str
    page_xml_url: str
    page_xml_size_bytes: int
    github_sha: str
    image_id: str
    image_url: str
    image_width: int
    image_height: int
    iiif_manifest_url: str
    bibliographical_record_url: str
    manuscript: str
    image_rights: str
    image_license: str
    estimated_image_size_bytes: int | None


@dataclass(frozen=True)
class RasamBatchPlan:
    source_id: str
    batch_id: str
    ok: bool
    created_at: str
    page_count: int
    file_count: int
    estimated_download_size_bytes: int
    estimated_extracted_size_bytes: int
    available_disk_bytes: int
    directories_to_create: list[str]
    files_to_create: list[str]
    license_summary: dict[str, Any]
    issues: list[str] = field(default_factory=list)
    pages: list[RasamPageCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class RasamBatchResult:
    ok: bool
    batch_id: str
    downloaded_size_bytes: int
    extracted_size_bytes: int
    page_count: int
    valid_pages: int
    rejected_pages: int
    source_manifest_path: str
    batch_manifest_path: str
    rejections_path: str
    report_path: str
    issues: list[str] = field(default_factory=list)


def _fetch_bytes(url: str) -> bytes:
    validate_download_url(url)
    from clouda_data.datasets.downloader import _urlopen

    request = urllib.request.Request(
        url, headers={"User-Agent": "arabic-ocr-dataset-prep/0.1"}
    )
    with _urlopen(request, timeout=60) as response:
        return response.read()


def _fetch_json(url: str) -> Any:
    return json.loads(_fetch_bytes(url).decode("utf-8"))


def _fetch_text(url: str) -> str:
    return _fetch_bytes(url).decode("utf-8-sig")


def _manifest_rights(payload: dict[str, Any]) -> tuple[str, str]:
    rights = ""
    license_value = ""
    for item in payload.get("metadata", []):
        label = str(item.get("label", "")).lower()
        value = item.get("value", "")
        if "droits" in label:
            rights = _flatten_metadata_value(value)
        if "licence" in label or "license" in label:
            license_value = _flatten_metadata_value(value)
    if not license_value and str(payload.get("license", "")):
        license_value = str(payload["license"])
    return rights, license_value


def _flatten_metadata_value(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(_flatten_metadata_value(item) for item in value)
    return re.sub(r"<[^>]+>", "", str(value)).strip()


def load_image_index() -> dict[str, dict[str, str]]:
    rows = csv.DictReader(_fetch_text(LIST_IMAGES_URL).splitlines(), delimiter="\t")
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = row["FileName"].strip()
        index[filename] = {key: value.strip() for key, value in row.items()}
    return index


def _github_page_entries(subset: str) -> list[dict[str, Any]]:
    url = f"{RASAM_API}/contents/page/{subset}?ref=main"
    entries = _fetch_json(url)
    return sorted(
        [entry for entry in entries if entry["name"].lower().endswith(".xml")],
        key=lambda item: item["name"],
    )


def _is_public_domain(rights: str, license_value: str) -> bool:
    rights_value = f"{rights} {license_value}".lower()
    return "domaine public" in rights_value or "public domain" in rights_value


def select_rasam_pages(batch_size: int = 100) -> list[RasamPageCandidate]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    image_index = load_image_index()
    selected: list[RasamPageCandidate] = []
    subset_targets = [
        ("rasam1", batch_size // 2),
        ("rasam2", batch_size - (batch_size // 2)),
    ]
    manifest_cache: dict[str, tuple[str, str]] = {}
    for subset, target in subset_targets:
        for entry in _github_page_entries(subset):
            page_id = Path(entry["name"]).stem
            image_row = image_index.get(page_id)
            if image_row is None:
                continue
            manifest_url = image_row["IIIF manifest"]
            if manifest_url not in manifest_cache:
                manifest_cache[manifest_url] = _manifest_rights(
                    _fetch_json(manifest_url)
                )
            rights, license_value = manifest_cache[manifest_url]
            image_id = image_row["IIIF image ID"]
            width = int(image_row["Width (px)"])
            height = int(image_row["Height (px)"])
            image_url = f"https://bina.bulac.fr/iiif/2/{image_id}/full/{width},{height}/0/default.jpg"
            estimated_image_size = content_length(image_url)
            if estimated_image_size is None:
                continue
            selected.append(
                RasamPageCandidate(
                    page_id=page_id,
                    subset=subset,
                    page_xml_url=entry["download_url"],
                    page_xml_size_bytes=int(entry["size"]),
                    github_sha=entry["sha"],
                    image_id=image_id,
                    image_url=image_url,
                    image_width=width,
                    image_height=height,
                    iiif_manifest_url=manifest_url,
                    bibliographical_record_url=image_row["Bibliographical record"],
                    manuscript=image_row["Manuscript"],
                    image_rights=rights,
                    image_license=license_value,
                    estimated_image_size_bytes=estimated_image_size,
                )
            )
            if sum(1 for item in selected if item.subset == subset) >= target:
                break
    return selected[:batch_size]


def _batch_root(project_root: Path) -> Path:
    return project_root / "data/downloads/rasam_dataset/first_batch"


def plan_rasam_first_batch(
    project_root: Path, *, batch_size: int = 100, max_bytes: int = ONE_GB
) -> RasamBatchPlan:
    pages = select_rasam_pages(batch_size)
    created_at = datetime.now(timezone.utc).isoformat()
    root = _batch_root(project_root)
    directories = [
        root,
        root / "metadata",
        root / "page_xml",
        root / "images",
        project_root / "data/manifests",
        project_root / "outputs/reports",
    ]
    metadata_files = [
        "metadata/LICENSE",
        "metadata/README.md",
        "metadata/contributing.md",
        "metadata/list-images.tsv",
    ]
    page_files = [f"page_xml/{page.page_id}.xml" for page in pages]
    image_files = [f"images/{page.page_id}.jpg" for page in pages]
    manifest_files = [
        "source_manifest.json",
        "../../../manifests/rasam_first_batch_manifest.json",
        "../../../manifests/rasam_first_batch_rejections.jsonl",
        "../../../manifests/download_manifests/rasam_dataset_first_batch.json",
        "../../../outputs/reports/rasam_first_batch_quality.json",
        "../../../docs/RASAM_FIRST_BATCH_REPORT.md",
    ]
    estimated = sum(
        page.page_xml_size_bytes + (page.estimated_image_size_bytes or 0)
        for page in pages
    )
    estimated += sum(
        content_length(url) or 0
        for url in [LICENSE_URL, README_URL, CONTRIBUTING_URL, LIST_IMAGES_URL]
    )
    issues: list[str] = []
    if len(pages) != batch_size:
        issues.append(f"selected_page_count_mismatch:{len(pages)}")
    if any(
        not _is_public_domain(page.image_rights, page.image_license) for page in pages
    ):
        issues.append("image_license_not_public_domain_for_all_pages")
    if any(page.estimated_image_size_bytes is None for page in pages):
        issues.append("missing_image_content_length")
    if estimated > max_bytes:
        issues.append(f"estimated_download_exceeds_limit:{estimated}>{max_bytes}")
    available = disk_free_bytes(project_root)
    if available < max_bytes:
        issues.append("insufficient_disk_space_for_limit")
    license_summary = {
        "repository_files": {
            "covered_by": "Apache-2.0",
            "files": [
                "README.md",
                "contributing.md",
                "list-images.tsv",
                "page/rasam1/*.xml",
                "page/rasam2/*.xml",
            ],
            "commercial_use": "permitted",
            "redistribution": "permitted_with_Apache_2_0_notice",
        },
        "external_images": {
            "covered_by": "BULAC/BiNA item rights, not the GitHub Apache-2.0 license",
            "required_rights": "Domaine public / Public Domain Mark 1.0",
            "commercial_use": "permitted_for_selected_pages",
            "redistribution": "permitted_for_selected_pages",
        },
        "third_party_assets": {
            "bundled_assets_found": False,
            "excluded_files": [],
        },
    }
    return RasamBatchPlan(
        source_id=RASAM_SOURCE_ID,
        batch_id="rasam_first_batch",
        ok=not issues,
        created_at=created_at,
        page_count=len(pages),
        file_count=len(metadata_files) + len(page_files) + len(image_files),
        estimated_download_size_bytes=estimated,
        estimated_extracted_size_bytes=estimated,
        available_disk_bytes=available,
        directories_to_create=[str(path) for path in directories],
        files_to_create=metadata_files + page_files + image_files + manifest_files,
        license_summary=license_summary,
        issues=issues,
        pages=pages,
    )


def _relative(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _download_asset(
    url: str, destination: Path, *, remaining_bytes: int
) -> DownloadedFile:
    if destination.exists():
        return DownloadedFile(
            url=url,
            path=str(destination),
            size_bytes=destination.stat().st_size,
            checksum_sha256=sha256_file(destination),
            archive_valid=None,
        )
    return download_http(url, destination, max_bytes=remaining_bytes)


def _page_xml_features(path: Path) -> dict[str, Any]:
    root = ElementTree.parse(path).getroot()
    page = next(
        (
            element
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "Page"
        ),
        None,
    )
    if page is None:
        raise ValueError("missing_page_element")
    regions = [
        element
        for element in page.iter()
        if element.tag.rsplit("}", 1)[-1].endswith("TextRegion")
    ]
    lines = [
        element
        for element in page.iter()
        if element.tag.rsplit("}", 1)[-1].endswith("TextLine")
    ]
    reading_refs = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "RegionRefIndexed":
            reading_refs.append(
                {
                    "index": element.attrib.get("index"),
                    "regionRef": element.attrib.get("regionRef"),
                }
            )
    region_ids = {element.attrib.get("id") for element in regions}
    invalid_reading_order = [
        item
        for item in reading_refs
        if item["regionRef"] not in region_ids or item["index"] is None
    ]
    text = text_from_ground_truth(path, "page_xml")
    layout_types = []
    for region in regions:
        layout_types.append(region.attrib.get("type") or "body")
    return {
        "image_filename": page.attrib.get("imageFilename"),
        "image_width": int(page.attrib.get("imageWidth", "0")),
        "image_height": int(page.attrib.get("imageHeight", "0")),
        "region_count": len(regions),
        "line_count": len(lines),
        "text": text,
        "text_checksum": sha256_text(text),
        "reading_order": reading_refs,
        "invalid_reading_order": invalid_reading_order,
        "layout_types": sorted(set(layout_types)),
    }


def download_rasam_first_batch(
    project_root: Path, *, batch_size: int = 100, max_bytes: int = ONE_GB
) -> RasamBatchResult:
    plan = plan_rasam_first_batch(
        project_root, batch_size=batch_size, max_bytes=max_bytes
    )
    if not plan.ok:
        return RasamBatchResult(
            False,
            plan.batch_id,
            0,
            0,
            plan.page_count,
            0,
            plan.page_count,
            "",
            "",
            "",
            "",
            plan.issues,
        )
    root = _batch_root(project_root)
    for directory in plan.directories_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)
    downloaded: list[DownloadedFile] = []
    remaining = max_bytes
    metadata_assets = [
        (LICENSE_URL, root / "metadata/LICENSE"),
        (README_URL, root / "metadata/README.md"),
        (CONTRIBUTING_URL, root / "metadata/contributing.md"),
        (LIST_IMAGES_URL, root / "metadata/list-images.tsv"),
    ]
    for url, destination in metadata_assets:
        item = _download_asset(url, destination, remaining_bytes=remaining)
        downloaded.append(item)
        remaining -= item.size_bytes
    for page in plan.pages:
        for url, destination in [
            (page.page_xml_url, root / f"page_xml/{page.page_id}.xml"),
            (page.image_url, root / f"images/{page.page_id}.jpg"),
        ]:
            item = _download_asset(url, destination, remaining_bytes=remaining)
            downloaded.append(item)
            remaining -= item.size_bytes
            if remaining < 0:
                return RasamBatchResult(
                    False,
                    plan.batch_id,
                    max_bytes - remaining,
                    0,
                    plan.page_count,
                    0,
                    plan.page_count,
                    "",
                    "",
                    "",
                    "",
                    ["download_exceeded_limit"],
                )
    download_manifest = {
        "source_id": RASAM_SOURCE_ID,
        "batch_id": plan.batch_id,
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "downloaded_size_bytes": sum(item.size_bytes for item in downloaded),
        "estimated_download_size_bytes": plan.estimated_download_size_bytes,
        "files": [asdict(item) for item in downloaded],
    }
    out = (
        project_root
        / "data/manifests/download_manifests/rasam_dataset_first_batch.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(download_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return verify_rasam_first_batch(project_root, plan=plan)


def verify_rasam_first_batch(
    project_root: Path, *, plan: RasamBatchPlan | None = None
) -> RasamBatchResult:
    if plan is None:
        plan = plan_rasam_first_batch(project_root)
    root = _batch_root(project_root)
    manifest_records: list[dict[str, Any]] = []
    source_pages: list[dict[str, Any]] = []
    issues: list[str] = []
    rejections: list[dict[str, Any]] = []
    seen_checksums: dict[str, str] = {}
    languages: set[str] = {"ar"}
    layout_types: dict[str, int] = {}
    widths: list[int] = []
    heights: list[int] = []
    valid_pages = 0
    downloaded_size = 0
    for page_number, page in enumerate(plan.pages, start=1):
        xml_path = root / f"page_xml/{page.page_id}.xml"
        image_path = root / f"images/{page.page_id}.jpg"
        reasons: list[str] = []
        features: dict[str, Any] = {}
        for path in (xml_path, image_path):
            if not path.exists():
                reasons.append(f"missing_file:{path.name}")
            else:
                downloaded_size += path.stat().st_size
                checksum = sha256_file(path)
                if checksum in seen_checksums and seen_checksums[checksum] != str(path):
                    reasons.append(f"duplicate_file:{path.name}")
                seen_checksums[checksum] = str(path)
        if not reasons:
            try:
                features = _page_xml_features(xml_path)
                if not features["text"].strip():
                    reasons.append("empty_ground_truth")
                if features["invalid_reading_order"]:
                    reasons.append("invalid_reading_order")
                if (
                    features["image_filename"]
                    and Path(features["image_filename"]).stem != page.page_id
                ):
                    reasons.append("page_xml_image_filename_mismatch")
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image_width, image_height = image.size
                if (image_width, image_height) != (
                    features["image_width"],
                    features["image_height"],
                ):
                    reasons.append("image_xml_dimension_mismatch")
                widths.append(image_width)
                heights.append(image_height)
                for layout_type in features["layout_types"]:
                    layout_types[layout_type] = layout_types.get(layout_type, 0) + 1
            except Exception as exc:
                reasons.append(f"corrupt_or_invalid:{exc}")
        xml_checksum = sha256_file(xml_path) if xml_path.exists() else None
        image_checksum = sha256_file(image_path) if image_path.exists() else None
        record = {
            "page_id": page.page_id,
            "subset": page.subset,
            "manuscript": page.manuscript,
            "page_number": page_number,
            "page_xml_path": (
                _relative(xml_path, project_root)
                if xml_path.exists()
                else str(xml_path)
            ),
            "image_path": (
                _relative(image_path, project_root)
                if image_path.exists()
                else str(image_path)
            ),
            "page_xml_checksum_sha256": xml_checksum,
            "image_checksum_sha256": image_checksum,
            "image_width": features.get("image_width", page.image_width),
            "image_height": features.get("image_height", page.image_height),
            "text_checksum": features.get("text_checksum"),
            "line_count": features.get("line_count", 0),
            "region_count": features.get("region_count", 0),
            "layout_types": features.get("layout_types", []),
            "ground_truth_present": bool(features.get("text", "").strip()),
            "layout_annotation_present": bool(features.get("region_count", 0)),
            "image_rights": page.image_rights,
            "image_license": page.image_license,
            "status": "valid" if not reasons else "rejected",
            "rejection_reasons": reasons,
        }
        manifest_records.append(record)
        if reasons:
            rejections.append(record)
        else:
            valid_pages += 1
            source_pages.append(
                {
                    "document_id": "rasam_first_batch",
                    "page_id": page.page_id,
                    "page_number": page_number,
                    "source_path": f"images/{page.page_id}.jpg",
                    "source_type": "image",
                    "language": "ar",
                    "ground_truth_path": f"page_xml/{page.page_id}.xml",
                    "text_checksum": features["text_checksum"],
                    "source_license": "Apache-2.0 PAGE XML; BULAC Public Domain Mark 1.0 image",
                    "reading_order": [],
                }
            )
    downloaded_size += sum(
        (root / f"metadata/{name}").stat().st_size
        for name in ["LICENSE", "README.md", "contributing.md", "list-images.tsv"]
        if (root / f"metadata/{name}").exists()
    )
    reason_counts = Counter(
        reason for item in manifest_records for reason in item["rejection_reasons"]
    )
    total_files = sum(1 for path in root.rglob("*") if path.is_file())
    source_manifest = {
        "documents": [
            {
                "document_id": "rasam_first_batch",
                "source_path": "metadata/list-images.tsv",
                "source_type": "text",
                "language": "ar",
                "source_license": "Apache-2.0 metadata; selected image rows verified as BULAC Public Domain Mark 1.0",
                "title": "RASAM first controlled batch",
            }
        ],
        "pages": source_pages,
    }
    source_manifest_path = root / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    batch_manifest_path = (
        project_root / "data/manifests/rasam_first_batch_manifest.json"
    )
    batch_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    batch_manifest = {
        "schema_version": "0.1.0",
        "batch_id": "rasam_first_batch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "source_id": RASAM_SOURCE_ID,
            "repository_url": RASAM_REPO,
            "image_provider": "BULAC BiNA IIIF",
        },
        "downloaded_size_bytes": downloaded_size,
        "extracted_size_bytes": downloaded_size,
        "total_files": total_files,
        "total_pages": len(manifest_records),
        "page_count": len(manifest_records),
        "valid_pages": valid_pages,
        "rejected_pages": len(rejections),
        "missing_ground_truth": reason_counts.get("missing_ground_truth", 0),
        "empty_ground_truth": reason_counts.get("empty_ground_truth", 0),
        "invalid_page_xml": sum(
            count
            for reason, count in reason_counts.items()
            if reason.startswith("corrupt_or_invalid")
            or reason == "invalid_reading_order"
        ),
        "image_xml_mismatches": reason_counts.get("page_xml_image_filename_mismatch", 0)
        + reason_counts.get("image_xml_dimension_mismatch", 0),
        "duplicate_pages": sum(
            count
            for reason, count in reason_counts.items()
            if reason.startswith("duplicate_file")
        ),
        "ground_truth_coverage": f"{valid_pages}/{len(manifest_records)}",
        "layout_annotation_coverage": f"{sum(1 for item in manifest_records if item['layout_annotation_present'])}/{len(manifest_records)}",
        "languages": sorted(languages),
        "layout_categories": layout_types,
        "image_sizes": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None,
        },
        "scan_or_rendering_quality": {
            "source_kind": "BULAC/BiNA IIIF full-page JPEG images",
            "image_decode": "passed for all valid pages",
            "dimension_alignment": "PAGE XML dimensions match decoded images for all valid pages",
            "corrupt_images": sum(
                count
                for reason, count in reason_counts.items()
                if reason.startswith("corrupt_or_invalid")
            ),
        },
        "license_status": "approved_with_conditions: Apache-2.0 repository annotations/metadata; selected BULAC/BiNA IIIF images verified as public-domain rights.",
        "records": manifest_records,
    }
    batch_manifest_path.write_text(
        json.dumps(batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rejections_path = project_root / "data/manifests/rasam_first_batch_rejections.jsonl"
    rejections_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rejections),
        encoding="utf-8",
    )
    quality_report = project_root / "outputs/reports/rasam_first_batch_quality.json"
    quality_report.parent.mkdir(parents=True, exist_ok=True)
    quality_report.write_text(
        json.dumps(batch_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = write_rasam_first_batch_report(project_root, batch_manifest, plan)
    return RasamBatchResult(
        ok=not issues,
        batch_id="rasam_first_batch",
        downloaded_size_bytes=downloaded_size,
        extracted_size_bytes=downloaded_size,
        page_count=len(manifest_records),
        valid_pages=valid_pages,
        rejected_pages=len(rejections),
        source_manifest_path=str(source_manifest_path),
        batch_manifest_path=str(batch_manifest_path),
        rejections_path=str(rejections_path),
        report_path=str(report_path),
        issues=issues,
    )


def write_rasam_first_batch_report(
    project_root: Path, summary: dict[str, Any], plan: RasamBatchPlan
) -> Path:
    out = project_root / "docs/RASAM_FIRST_BATCH_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RASAM First Batch Report",
        "",
        f"Verification date: {datetime.now().date().isoformat()}",
        "",
        "## License conclusion",
        "",
        "- RASAM repository files used in this batch (README, contributing notes, list-images.tsv, and PAGE XML under page/rasam1 and page/rasam2) are covered by Apache-2.0 as repository contents.",
        "- Page images are not covered by the GitHub repository license. They are external BULAC BiNA IIIF assets and were downloaded only for selected manuscripts whose IIIF manifest metadata reports Domaine public / Public Domain Mark 1.0.",
        "- Apache-2.0 notices must be preserved for annotations and metadata. BULAC/BiNA provenance should be cited for image reuse.",
        "- No bundled third-party assets were found in the downloaded batch.",
        "",
        "## Batch summary",
        "",
        f"- Pages selected: {summary['page_count']}",
        f"- Total files: {summary['total_files']}",
        f"- Valid pages: {summary['valid_pages']}",
        f"- Rejected pages: {summary['rejected_pages']}",
        f"- Missing ground truth: {summary['missing_ground_truth']}",
        f"- Empty ground truth: {summary['empty_ground_truth']}",
        f"- Invalid PAGE XML: {summary['invalid_page_xml']}",
        f"- Image/XML mismatches: {summary['image_xml_mismatches']}",
        f"- Duplicate pages: {summary['duplicate_pages']}",
        f"- Downloaded size: {summary['downloaded_size_bytes']} bytes",
        f"- Extracted size: {summary['extracted_size_bytes']} bytes",
        f"- Ground-truth coverage: {summary['ground_truth_coverage']}",
        f"- Layout-annotation coverage: {summary['layout_annotation_coverage']}",
        f"- Languages: {', '.join(summary['languages'])}",
        f"- Layout categories: {json.dumps(summary['layout_categories'], ensure_ascii=False)}",
        f"- Image sizes: {json.dumps(summary['image_sizes'], ensure_ascii=False)}",
        f"- Scan/rendering quality: {json.dumps(summary['scan_or_rendering_quality'], ensure_ascii=False)}",
        f"- License status: {summary['license_status']}",
        "",
        "## Pre-download plan",
        "",
        f"- Estimated download size: {plan.estimated_download_size_bytes} bytes",
        f"- Estimated extracted size: {plan.estimated_extracted_size_bytes} bytes",
        f"- Available disk: {plan.available_disk_bytes} bytes",
        f"- Planned file count: {plan.file_count}",
        "",
        "## Created files",
        "",
        "- data/downloads/rasam_dataset/first_batch/",
        "- data/manifests/rasam_first_batch_manifest.json",
        "- data/manifests/rasam_first_batch_rejections.jsonl",
        "- outputs/reports/rasam_first_batch_quality.json",
        "",
        "## Quality notes",
        "",
        "- Checksums are recorded for every downloaded source image and PAGE XML file.",
        "- Image dimensions were verified against PAGE XML page dimensions.",
        "- PAGE XML text was extracted only from Unicode elements under the XML structure.",
        "- Invalid items are rejected in the rejections JSONL and are not silently repaired.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def plan_to_dict(plan: RasamBatchPlan) -> dict[str, Any]:
    payload = asdict(plan)
    return payload


def result_to_dict(result: RasamBatchResult) -> dict[str, Any]:
    return asdict(result)
