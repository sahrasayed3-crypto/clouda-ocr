from __future__ import annotations

from pathlib import Path

from clouda_data.ground_truth.checksums import sha256_text

from .schema import LayoutRegion, PageRecord, SourceManifest


def validate_region(region: LayoutRegion) -> None:
    x, y, width, height = region.bbox
    if width <= 0 or height <= 0:
        raise ValueError(f"Region {region.region_id} has non-positive dimensions.")
    if x < 0 or y < 0:
        raise ValueError(f"Region {region.region_id} starts outside the page.")
    if region.reading_order < 0:
        raise ValueError(f"Region {region.region_id} has invalid reading order.")


def validate_page_record(record: PageRecord, project_root: Path | None = None) -> None:
    if not record.document_id or not record.page_id:
        raise ValueError("document_id and page_id are required.")
    if record.page_number < 1:
        raise ValueError("page_number must start at 1.")
    if not record.clean_text and record.source_type != "image":
        raise ValueError("clean_text is required for text-bearing sources.")
    expected = sha256_text(record.clean_text)
    if record.text_checksum != expected:
        raise ValueError("text_checksum does not match clean_text.")
    region_ids = {region.region_id for region in record.layout_regions}
    for region in record.layout_regions:
        validate_region(region)
    for referenced in record.reading_order:
        if referenced not in region_ids:
            raise ValueError(f"reading_order references unknown region: {referenced}")
    if project_root is not None:
        path = (project_root / record.source_path).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise ValueError("source_path must stay inside the project root.") from exc


def validate_source_manifest(manifest: SourceManifest) -> None:
    document_ids: set[str] = set()
    for document in manifest.documents:
        if not document.document_id:
            raise ValueError("Every source document must have document_id.")
        if document.document_id in document_ids:
            raise ValueError(f"Duplicate document_id: {document.document_id}")
        document_ids.add(document.document_id)
        if not document.source_path:
            raise ValueError(f"Document {document.document_id} is missing source_path.")
    page_ids: set[str] = set()
    for page in manifest.pages:
        if not page.page_id:
            raise ValueError("Every source page must have page_id.")
        if page.page_id in page_ids:
            raise ValueError(f"Duplicate page_id: {page.page_id}")
        page_ids.add(page.page_id)
        if page.document_id not in document_ids:
            raise ValueError(
                f"Page {page.page_id} references unknown document_id: {page.document_id}"
            )
        if page.page_number < 1:
            raise ValueError(f"Page {page.page_id} page_number must start at 1.")
        if not page.source_path:
            raise ValueError(f"Page {page.page_id} is missing source_path.")
        if page.clean_text is None and not page.ground_truth_path:
            raise ValueError(
                f"Page {page.page_id} must provide clean_text or ground_truth_path."
            )
