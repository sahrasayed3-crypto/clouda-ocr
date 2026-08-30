# Training data privacy

Status: **code-level privacy gates complete**.

User uploads have no route into the training catalog. Existing two-gate
document consent defaults to false. Training export accepts only external
dataset-state manifests and writes only to the external artifact root.
Commercial export requires catalog approval and an explicit commercial flag.

Ground truth is never normalized or rewritten during visual augmentation.
Restricted, pending, blocked, research-only, and evaluation-only records cannot
enter commercial training. Logs redact common credential fields.

