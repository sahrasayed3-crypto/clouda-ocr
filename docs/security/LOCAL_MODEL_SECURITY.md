# Local model security

Status: **code-level controls complete; model trust remains a human decision**.

Local OCR is disabled by default. Model and processor paths must be local;
Transformers receives `local_files_only=True` and `trust_remote_code=False`.
Images are decoded and pixel-bounded before inference. Output must be non-empty
and confidence-bounded. HTTP destinations default to loopback. Command engines
require an absolute executable, argument arrays, no shell, output capture, and
a timeout.

Model license, weights, revision, malware scanning, GPU isolation, and measured
quality must be approved externally before real activation.

