# Roadmap

## Implemented

- Production digital-PDF extraction and RTL DOCX output.
- Isolated `clouda_data`, versioned contracts, external storage roots.
- Dataset catalog, license gate, manifest migration, and asset reconciliation.
- Training planner and model metadata registry without training or weights.
- Disabled local-model boundary, queue isolation, security and CI checks.
- Real bounded PDF/image rendering and 100+ deterministic pixel operators.
- Versioned YAML profiles, batch resume, validation, quarantine, and previews.
- CER/WER execution and license-gated deterministic training-data export.
- Completed 177-page public benchmark with HunyuanOCR-1.5 ranked first on
  Normalized Arabic CER for that specific benchmark.
- Completed and published the 462-page model-selection benchmark v1.0 (10
  candidate OCR/VLM models; 5 complete ranked runs; 3 partial runs; 2 failed
  runs; Normalized Arabic CER as the primary ranking metric; 7,198 / 7,198
  release checksums verified) in the
  [separate benchmark repository](https://github.com/sahrasayed3-crypto/clouda-ocr-model-selection-benchmark).
- Safe local OCR adapters and mock-verified runtime integration.
- Dry-run-first lifecycle operations and local administration UI.
- Request IDs, rate limiting, Redis TLS hooks, and Prometheus metrics.

## Partially complete / disabled by default

- Dataset, training-preparation, and model-evaluation worker capabilities.
- Real local-model adapters are implemented, but no final production model or
  weight is integrated.
- On the earlier published 177-page benchmark v0.1.0 (historical),
  HunyuanOCR-1.5 ranked first by Normalized Arabic CER (0.391497). This is a
  benchmark-specific historical result, separate from the completed
  model-selection benchmark v1.0 that now anchors model selection.
- OIDC/reverse-proxy boundary and production rate-limit guidance.
- Selected-model development and training are the current technical stage,
  building on the completed model-selection benchmark v1.0; model-agnostic
  runtime integration follows. Dataset and model rights remain governed by the
  fail-closed licensing and provenance process. No final trained production
  model or hosted service exists yet.

## External decisions

- Project B code copyright and license.
- Production authentication provider and repository visibility.
- Final trained/integrated production runtime packaging for the selected
  model: architecture integration, revision pinning, licensing, deployment
  constraints, and GPU platform. (Benchmark model selection itself is complete
  via published model-selection benchmark v1.0; no final trained production
  model exists yet.)
- Commercial permissions for pending datasets.
- A reviewed user-document consent policy (current behavior remains disabled).
