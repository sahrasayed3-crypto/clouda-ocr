# Data Ingestion Workflow

This stage prepares clean source data for later rendering and distortion. It does not distort pages, train OCR, download model weights, or use AWS.

## Where To Place Future Clean Data

Place future clean data for local registration under:

- `data/raw/incoming/`

After validation, copy-based ingestion writes canonical copies to:

- Source documents: `data/raw/documents/`
- Clean page images or page-level source files: `data/raw/pages/`
- Exact ground truth: `data/raw/ground_truth/`
- Layout annotations: `data/raw/layout_annotations/`

Original files are never moved, deleted, or modified.

## Required Manifest

Create a source ingestion manifest JSON matching `src/ingestion/source_manifest.schema.json`.

Required document fields:

- `document_id`
- `source_path`
- `source_type`

Required page fields:

- `document_id`
- `page_id`
- `page_number`
- `source_path`
- `source_type`
- Either `clean_text` or `ground_truth_path`

Recommended fields:

- `language`
- `source_license`
- `text_checksum`
- `layout_path`
- `reading_order`

Accepted source types are `pdf`, `docx`, `image`, `text`, `ground_truth_json`, `page_xml`, `alto_xml`, and `json_layout`.

## Validation Rules

Ingestion rejects missing source files, unsupported formats, corrupt PDFs/DOCX/images/XML/JSON, missing ground truth, empty text, duplicate distinct files, ground-truth checksum mismatches, missing identifiers, unknown document references, invalid page numbers, and invalid reading order.

## Commands

```powershell
.\.venv\Scripts\python -m clouda_data.pipeline.cli inspect-source data/raw/incoming/example.pdf
.\.venv\Scripts\python -m clouda_data.pipeline.cli validate-source data/raw/incoming/source_manifest.json
.\.venv\Scripts\python -m clouda_data.pipeline.cli ingest data/raw/incoming/source_manifest.json --dry-run
.\.venv\Scripts\python -m clouda_data.pipeline.cli ingest data/raw/incoming/source_manifest.json
.\.venv\Scripts\python -m clouda_data.pipeline.cli list-ingested
.\.venv\Scripts\python -m clouda_data.pipeline.cli find-duplicates
.\.venv\Scripts\python -m clouda_data.pipeline.cli generate-ingestion-report --manifest data/raw/incoming/source_manifest.json
```

The dry-run command performs validation and planning only. The non-dry-run command copies files into canonical raw folders and writes `data/manifests/source_document_manifest.json`, `data/manifests/page_manifest.json`, and `data/manifests/file_registry.json`.

## Current Managed Real Batch

The first controlled RASAM batch source files are preserved at:

- `data/downloads/rasam_dataset/first_batch/images/`
- `data/downloads/rasam_dataset/first_batch/page_xml/`
- `data/downloads/rasam_dataset/first_batch/metadata/`
- `data/downloads/rasam_dataset/first_batch/source_manifest.json`

The managed copy-based ingestion for valid pages writes to:

- `data/raw/pages/`
- `data/raw/ground_truth/`

For RASAM, the PAGE XML files in `data/raw/ground_truth/` carry both exact text and layout annotations. They are registered once so duplicate detection remains meaningful.

Rejected source pages are not repaired or silently altered. They are recorded in `data/manifests/rasam_first_batch_rejections.jsonl`.
