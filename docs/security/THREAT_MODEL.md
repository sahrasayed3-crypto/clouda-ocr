# Threat model

## Scope and assets

Primary assets are user documents and results, identities and sessions, worker
credentials, correction data, licensed datasets, model files, runtime
databases, and deployment configuration. Relevant actors include unauthenticated
internet users, malicious authenticated users, compromised workers, untrusted
dataset/model contributors, malicious administrators, compromised dependencies
or CI runners, and local users with limited filesystem access.

## Trust boundaries

| Boundary | Untrusted source | Higher-trust destination | Required controls |
|---|---|---|---|
| Browser/upload | Requests, files, names | Runtime storage | Authentication, ownership, generated names, byte/page limits |
| Worker callback | Worker or queue payload | API and database | Worker key, job ownership, state transition checks, idempotency |
| Documents/archives | PDF, image, XML, ZIP | Parsers and restore roots | Early bounds, safe XML, pixel limits, traversal/link checks |
| Data manifests | Contributor metadata | Dataset/training roots | Schema, contained URIs, checksums, authoritative license catalog |
| Models/providers | Model files or endpoints | Inference process | Approved roots/revisions, timeouts, no remote code, SSRF controls |
| Extracted text | OCR/document content | DOCX, CSV, HTML, logs | Treat as data; sanitize XML, formulas, markup, and log fields |
| Deployment edge | Public network | Local API, Redis, workers | TLS, identity, trusted proxy, rate limits, ACLs, network isolation |

## Threats and controls

Material threats include oversized or malformed documents, archive bombs,
external XML entities, image decompression, path traversal, hostile filenames,
cross-tenant reads, unauthorized or replayed worker calls, command injection,
SSRF, secret leakage, active content in generated outputs, dataset-license
bypass, model/checkpoint substitution, cross-domain storage access, unsafe
backup restoration, and accidental user-document training.

Implemented controls include bounded streaming, archive/XML/image validation,
external and separated storage roots, opaque identifiers, owner-scoped queries,
parameterized SQL, worker authentication with constant-time comparison,
validated state transitions, trusted hosts, security headers, queue
capabilities, license gates, two-part training-consent gates, local-only model
loading, output sanitization, recursive log redaction, and repository scanning.

## Deployment assumptions and residual risk

Administrators control configuration and service accounts. Redis and worker
interfaces remain on private networks. Local OCR is disabled unless an
approved, pinned model is explicitly configured. Public deployment remains
conditional on production identity, TLS, proxy trust, rate limiting, managed
secrets and rotation, monitoring, incident response, backup/retention policy,
and legal approval for datasets and models. Optional cloud, GPU, and external
provider paths require separate validation in their actual environments.
