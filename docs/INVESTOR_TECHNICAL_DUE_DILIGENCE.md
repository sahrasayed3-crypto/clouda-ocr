# Clouda OCR — Technical Due-Diligence Pack

This document records the repository's current, verifiable technical status.
It is evidence-oriented and does not make claims about customers, revenue,
production OCR accuracy, or a hosted product.

## Current status

| Area | Verified status |
| --- | --- |
| Software | Clouda OCR v0.2.1 is the current release. The release DOI recorded in `CITATION.cff` is `10.5281/zenodo.22903260`. |
| Runtime | The supported application path converts trusted born-digital PDF text to editable, RTL-aware DOCX. |
| Scanned pages | Image-only and otherwise unsupported pages route to `pending_ocr_model`; that state is not successful OCR output. |
| Model selection | Complete. The public v1.0 model-selection benchmark is the current benchmark record. |
| Project phase | Selected-model development and training. |
| Not shipped | No final trained production OCR model, integrated production OCR engine, hosted API, or SaaS product. |

The project is model-agnostic at runtime. A benchmark rank does not by itself
authorize, integrate, or establish a production model.

## What is implemented

- Page analysis, a categorical trusted-digital-text gate, per-page routing,
  and explicit review states.
- Editable DOCX export with Arabic Unicode and RTL paragraph handling for the
  trusted born-digital-text route.
- Model registry and extraction-engine boundaries for a future approved OCR
  integration.
- Deterministic data, evaluation, quality-gate, provenance, and training
  planning infrastructure.
- A loopback-only Clouda Lab control plane for local inspection and managed
  workflows. Normal navigation does not download models or datasets.

These capabilities do not imply a released production OCR engine or a
production accuracy figure.

## Current model-selection benchmark

The current public record is the [Clouda OCR Arabic OCR Model Selection
Benchmark v1.0](https://github.com/sahrasayed3-crypto/clouda-ocr-model-selection-benchmark/releases/tag/v1.0.0).
It contains 462 Arabic document pages and 10 OCR/VLM candidates. Its primary
metric is Normalized Arabic CER. The release is quality-only: it intentionally
does not publish runtime, speed, or VRAM comparisons. All 7,198 release
checksums were verified, with zero mismatches and zero missing files.

| Rank | Model | Valid | Normalized CER | Raw CER | Normalized WER | Raw WER |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | amad-iq/amad-vlm6 | 462/462 | 0.405471 | 0.408645 | 0.533982 | 0.536017 |
| 2 | amad-iq/amad-vlm5 | 462/462 | 0.406228 | 1.049789 | 0.542530 | 0.544223 |
| 3 | tencent/HunyuanOCR | 462/462 | 0.434648 | 0.438131 | 0.517221 | 0.517424 |
| 4 | YasserSami/Qari-OCR-0.4.0-VL-4B-Instruct | 462/462 | 2.163387 | 2.140999 | 2.728452 | 2.731869 |
| 5 | MBZUAI/AIN | 462/462 | 2.176870 | 2.500256 | 2.453322 | 2.456118 |

Three runs are recorded as `EARLY_STOPPED_NONCOMPETITIVE` and are unranked:
`context212/alhazen-ocr` (401/462),
`hastyle/olmOCR-arabic-lora-v2` (188/462), and
`AhmedZaky1/DIMI-Arabic-OCR-V2` (132/462). The classification records an
early stop; it does not claim that these models were stopped for poor OCR
quality.

Two candidates did not produce a complete ranked run:

- `loay/Arabic-OCR-DeepSeek-OCR-2` — `FAILED_INCOMPATIBLE`, 0/462 pages.
- `sherif1313/Arabic-GLM-OCR-v2` — `FAILED_SMOKE`, no full run.

## Historical benchmark record

The repository's v0.1.0 benchmark is historical. It used 177 distorted Arabic
pages and six complete model runs; HunyuanOCR-1.5 ranked first by Normalized
Arabic CER for that benchmark. Its DOI is `10.5281/zenodo.22859930`. It must
not be represented as the current model-selection benchmark.

## Engineering evidence and reproducibility

The CI workflow runs on Windows and Ubuntu with Python 3.11. Its required
`validate` job installs the constrained dependency set and runs import/CLI
smoke tests, JavaScript syntax validation, compilation, Ruff, Black, mypy,
the offline pytest coverage gate, synthetic acceptance, repository and
security scans, dependency audit, package build, and demo smoke test.

Use the commands and pinned dependency set in `.github/workflows/ci.yml` for
the result of a particular commit. Test counts and pass counts are snapshots,
not enduring product claims; `docs/TESTING.md` records a dated historical
result and points readers to the release-readiness evidence for the current
suite.

## Boundaries and remaining work

- A final trained OCR model has not been released.
- The application has no integrated production OCR engine and makes no
  production OCR-accuracy claim.
- No hosted API or SaaS offering is documented here.
- Licensing, provenance, and deployment acceptance remain separate gates from
  benchmark ranking.
- Current work is selected-model development and training, followed by any
  future integration and application-route validation.

## Evidence map

See [the companion source map](INVESTOR_TECHNICAL_DUE_DILIGENCE_SOURCES.md)
for the files, release records, and workflow configuration supporting these
claims.
