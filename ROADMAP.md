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
- Safe local OCR adapters and mock-verified runtime integration.
- Dry-run-first lifecycle operations and local administration UI.
- Request IDs, rate limiting, Redis TLS hooks, and Prometheus metrics.

## Partially complete / disabled by default

- Dataset, training-preparation, and model-evaluation worker capabilities.
- Real local-model adapters are implemented; no model or weight is selected.
- OIDC/reverse-proxy boundary and production rate-limit guidance.
- GPU training remains disabled even though training-data preparation is complete.

## External decisions

- Project B code copyright and license.
- Production authentication provider and repository visibility.
- Final OCR/VLM model, revision, license, and GPU platform.
- Commercial permissions for pending datasets.
- A reviewed user-document consent policy (current behavior remains disabled).
