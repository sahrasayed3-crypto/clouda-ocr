# Changelog

## 0.1.0 - 2026-07-22

- Created foundation project structure.
- Added configuration, ingestion, ground-truth, distortion, layout, manifest, validation, evaluation, CLI, test, documentation, and AWS-template scaffolding.
- No datasets, model weights, OCR training, real distortion generation, or AWS resources were used.

## 0.2.0 - 2026-07-22

- Added data-ingestion preparation workflow.
- Added source inspection, source manifest validation, copy-based ingestion, file registry, duplicate detection, and ingestion reports.
- Added CLI commands for ingestion preparation.
- Added tests for tiny synthetic PDF, DOCX, image, text, JSON, PAGE XML, and ALTO XML fixtures.
- Still no real datasets, model weights, OCR training, real distortion generation, or AWS resources.

## 0.3.0 - 2026-07-23

- Added license-aware dataset registry and source reports.
- Added safe sample downloader with resume, checksum, duplicate, size-limit, archive-validation, quarantine, and manifest support.
- Added dataset CLI commands.
- Downloaded a tiny approved RASAM PAGE XML/license sample under the 100 MB test cap.
- Validated the downloaded sample through the existing ingestion dry-run workflow.
- Still no full dataset download, model weights, OCR training, real distortion generation, or AWS resources.
