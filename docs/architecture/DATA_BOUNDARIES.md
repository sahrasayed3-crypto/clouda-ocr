# Data boundaries

Tracked code may contain schemas, redacted examples, catalog metadata, and
checksums. It must not contain real scans, page text manifests, uploads,
databases, generated DOCX, reports with private paths, or model artifacts.

| URI | Purpose |
|---|---|
| `runtime://` | conversions, runtime database, user-owned state |
| `dataset://` | licensed downloads, raw data, manifests |
| `artifact://` | reports, evaluation, training plans |
| `model://` | weights and checkpoints |
| `cache://` | disposable render and download caches |

Path resolution rejects traversal and cross-root database locations. Runtime
workers cannot access dataset or model roots. Dataset licensing is evaluated
before commercial training planning.
