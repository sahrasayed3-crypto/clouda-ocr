# Key Router Security

Secrets policy:

- Database stores `secret_ref` only.
- Secret values are loaded from environment variables through `EnvSecretResolver`.
- `SecretHandle.__repr__` redacts values and exposes only a hash fingerprint suffix.
- Audit metadata is sanitized for secret, token, authorization, password, credential,
  and signed URL fields.
- Internal status API omits `secret_ref` and never returns key material.

Modes:

- `disabled`: no selection and no side effects.
- `shadow`: sanitized audit only, old request path remains authoritative.
- `active`: fails closed if no eligible authorized account exists.

This system must not be used to bypass provider terms, quotas, or billing limits.
