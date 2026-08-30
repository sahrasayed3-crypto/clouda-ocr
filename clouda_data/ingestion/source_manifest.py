from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import SourceDocument, SourceManifest, SourcePage


def _document_from_mapping(data: dict[str, Any]) -> SourceDocument:
    return SourceDocument(
        document_id=data["document_id"],
        source_path=data["source_path"],
        source_type=data["source_type"],
        language=data.get("language", "ar"),
        source_license=data.get("source_license", "unknown"),
        title=data.get("title"),
    )


def _page_from_mapping(data: dict[str, Any]) -> SourcePage:
    return SourcePage(
        document_id=data["document_id"],
        page_id=data["page_id"],
        page_number=int(data["page_number"]),
        source_path=data["source_path"],
        source_type=data["source_type"],
        language=data.get("language", "ar"),
        ground_truth_path=data.get("ground_truth_path"),
        clean_text=data.get("clean_text"),
        text_checksum=data.get("text_checksum"),
        layout_path=data.get("layout_path"),
        source_license=data.get("source_license", "unknown"),
        reading_order=list(data.get("reading_order", [])),
    )


def read_source_manifest(path: str | Path) -> SourceManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source manifest must be a JSON object.")
    documents = [_document_from_mapping(item) for item in payload.get("documents", [])]
    pages = [_page_from_mapping(item) for item in payload.get("pages", [])]
    if not documents:
        raise ValueError("Source manifest must contain at least one document.")
    if not pages:
        raise ValueError("Source manifest must contain at least one page.")
    return SourceManifest(documents=documents, pages=pages)


def write_source_manifest(manifest: SourceManifest, path: str | Path) -> None:
    payload = {
        "documents": [asdict(document) for document in manifest.documents],
        "pages": [asdict(page) for page in manifest.pages],
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
