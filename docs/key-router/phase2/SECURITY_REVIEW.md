# Security Review

Confirmed controls:

- SQLite accepts validated environment secret references, not raw secret values.
- Secret handles and account repr output redact references and never show values.
- Authorization/API-key headers, secret-like fields, signed URL credentials, Base64
  image/document payloads, prompts, content, and document text are redacted from
  operational logging/audit views.
- Operational API output excludes secret references and reservation tokens.
- Internal endpoints reuse constant-time worker internal authentication and ID validation.
- Privacy classifications can block provider accounts.
- Provider messages are sanitized before persistence; raw responses are not stored.

No secret, credential, external request, or live health check was used. Residual risks are
listed separately; the system remains experimental.
