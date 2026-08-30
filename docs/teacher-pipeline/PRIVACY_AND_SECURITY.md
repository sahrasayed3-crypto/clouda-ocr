# Privacy and Security

Confirmed controls:

- All four Phase 3 safety switches default false/disabled and parse fail closed.
- Phase 3 live mode is rejected and no live dispatcher exists.
- Mock mode uses only local deterministic inputs and records zero network requests.
- Secret references must match the environment-reference format; raw secret-shaped values are rejected.
- Assignment and endpoint privacy classifications must match before selection.
- Internal routes reuse constant-time worker authentication.
- Status APIs omit secret references, complete text, images, payloads, and raw responses.
- Audit metadata uses the Phase 2 sanitizer.
- Original images are hashed before and after preparation and never overwritten.

Residual risk: real-provider privacy, retention, terms, MIME limits, and capability behavior remain unverified because live tests are explicitly prohibited.
