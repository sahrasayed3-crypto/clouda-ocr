# Architecture

The project is organized as a staged pipeline:

1. Ingestion inspects, validates, checksum-registers, duplicate-checks, and copy-registers clean source data and exact ground truth.
2. Rendering creates clean page images from approved source documents.
3. Distortion applies replayable, profile-driven image operations.
4. Validation checks output image quality, metadata, layout preservation, and text checksums.
5. Manifest storage records every generated page as JSON Lines and optionally SQLite.
6. Evaluation compares future OCR output against preserved references with CER and WER.
7. Reporting produces JSON, CSV, and Markdown summaries.

The ingestion preparation stage implements dry-run and copy-based clean-data registration. No real source collection or generated training dataset exists.
