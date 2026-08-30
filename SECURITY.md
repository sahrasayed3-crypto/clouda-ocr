# Security policy

Clouda PDF is suitable for local development and controlled staging. A public
multi-user deployment requires production identity, TLS termination, trusted
proxy configuration, rate limiting, network isolation, managed secrets,
monitoring, incident response, retention rules, and applicable data/model
license approvals.

Never commit secrets, `.env`, databases, user documents, datasets, logs,
backups, model weights, or checkpoints. Use external storage roots and run:

```text
python -m tools.validation.repository_scan --root .
```

Uploads are bounded by byte and page limits. DOCX and backup archives are
checked for traversal, links, member count, expansion size, and compression
ratio. XML uses `defusedxml`; images have a pixel limit. The worker API requires
a header key, validates hosts, disables docs, and sets baseline security
headers. Configure request limits, rate limiting, TLS, and authentication at a
trusted reverse proxy in production.

User documents are never training data by default. A code-level consent
boundary requires both document-specific consent and an approved policy; no
current runtime path calls it to admit documents to training.

Redis must bind to a private network and accept only isolated workers.
`CLOUDA_WORKER_API_KEY_PREVIOUS` is a temporary rotation hook and must be
removed after cutover. Local OCR remains disabled unless explicitly configured
with an approved, pinned model.

The repository assumes administrators control service configuration and
storage roots. `runtime://`, `dataset://`, `artifact://`, `model://`, and
`cache://` must remain separated. OCR text and document content are untrusted
data, never instructions. Model adapters must not download code or weights at
runtime, and Redis or worker endpoints must not be exposed directly to the
internet.

Security checks run through the normal test suite and CI. They cover upload
and archive bounds, XML/image handling, path containment, worker callbacks,
SSRF defenses, model-root isolation, license and consent gates, output
sanitization, backup/restore safety, secret redaction, dependency auditing, and
repository scanning. Historical pass/fail exercise reports are deliberately
kept outside the public source tree; current test results are the authority.

Report vulnerabilities privately with the affected component, reproduction,
impact, and suggested mitigation. Rotate any exposed credential immediately.
See the [threat model](docs/security/THREAT_MODEL.md),
[data privacy policy](docs/security/DATA_PRIVACY.md), and
[authentication guide](docs/authentication.md).
