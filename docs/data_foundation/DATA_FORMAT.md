# Data Format

Each future clean page must be represented by a page record with:

- `document_id`
- `page_id`
- `source_path`
- `source_type`
- `language`
- `page_number`
- `clean_text`
- `text_checksum`
- `image_checksum`
- `layout_regions`
- `footnote_regions`
- `margin_regions`
- `title_regions`
- `table_regions`
- `reading_order`
- `source_license`
- `creation_timestamp`

Accepted future source types:

- Digital PDF
- DOCX
- Page image
- Text file
- JSON layout annotation
- PAGE XML
- ALTO XML
- Structured ground-truth JSON

Ground truth and layout metadata must be linked by stable IDs. Source files must remain inside the project root or an explicitly approved storage mount.

## Source Ingestion Manifest

Before page records are created, clean inputs are described by `src/ingestion/source_manifest.schema.json`.

Top-level fields:

- `documents`
- `pages`

Each document requires `document_id`, `source_path`, and `source_type`.

Each page requires `document_id`, `page_id`, `page_number`, `source_path`, `source_type`, and either `clean_text` or `ground_truth_path`.

Each page may include `text_checksum`. When present, ingestion rejects the page unless the checksum matches the extracted or inline ground truth exactly.

Future clean data should first be placed under `data/raw/incoming/`. Successful ingestion copies files into canonical raw folders and writes source-document and page manifests under `data/manifests/`.
