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
- Safe local OCR adapters and mock-verified runtime integration.
- Dry-run-first lifecycle operations and local administration UI.
- Request IDs, rate limiting, Redis TLS hooks, and Prometheus metrics.

## Partially complete / disabled by default

- Dataset, training-preparation, and model-evaluation worker capabilities.
- Real local-model adapters are implemented, but no final production model or
  weight is integrated.
- HunyuanOCR-1.5 is the current leading candidate based on this specific
  177-page benchmark (Normalized Arabic CER 0.391497). Final production/runtime
  selection remains subject to architecture, licensing, deployment constraints,
  integration, and subsequent validation.
- OIDC/reverse-proxy boundary and production rate-limit guidance.
- Model adaptation/training and runtime integration are the next major technical
  stage. Progress is currently limited primarily by access to suitable GPU
  compute; dataset and model rights remain governed by the fail-closed licensing
  and provenance process.

## External decisions

- Project B code copyright and license.
- Production authentication provider and repository visibility.
- Final production/runtime OCR/VLM model, architecture, revision, licensing,
  deployment constraints, and GPU platform.
- Commercial permissions for pending datasets.
- A reviewed user-document consent policy (current behavior remains disabled).
