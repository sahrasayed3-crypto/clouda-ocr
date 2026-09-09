# Clouda merged architecture

Clouda is one repository with isolated domains and external state. `pdfword` is
the production PDF-to-DOCX runtime. `clouda_data` prepares and evaluates
licensed OCR datasets. `clouda_contracts` is the dependency-light boundary
between domains. `clouda_training` plans training and executes reproducible
offline mock experiments without model downloads, and
`clouda_models` records model metadata without storing weights.

```text
Streamlit / worker API -> pdfword -> runtime://
                              |
                              v
                       clouda_contracts
                              ^
clouda_data -> dataset://     |     clouda_training -> artifact://
clouda_models -> model:// ----+
```

The code tree contains metadata and implementations only. Runtime files,
datasets, reports, caches, databases, and model files resolve through
`StorageRoots` and remain outside Git. Dataset licensing fails closed:
`pending`, `blocked`, and research-only records cannot enter commercial
training.

The local OCR boundary is feature flagged and disabled by default. Born-digital
PDF extraction remains the production path. Scanned pages remain
`pending_ocr_model` unless an enabled provider returns non-empty text, quality
metadata, and a pinned model revision.

The training experiment layer is filesystem-backed and adapter-neutral. It
validates dataset identity and the protected-holdout boundary before creating a
run, then records immutable resolved configuration, environment/Git provenance,
status history, JSONL metrics, integrity-checked checkpoints, and summaries.
Only the mock adapter is enabled; real GPU training remains fail-closed.

See [system overview](docs/architecture/SYSTEM_OVERVIEW.md),
[data boundaries](docs/architecture/DATA_BOUNDARIES.md), and
[runtime versus training](docs/architecture/RUNTIME_VS_TRAINING.md).
