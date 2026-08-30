# Circuit Breaker

Persistent SQLite circuits support provider, provider-model, and provider-account levels
with `CLOSED`, `OPEN`, and `HALF_OPEN`. Records include rolling-window failure counts,
cooldown, half-open in-flight probes, successes, version, and timestamps.

Only relevant operational/provider failures feed circuits; ordinary 400/422 payload
errors do not. Provider and model state remain separate. `BEGIN IMMEDIATE` and a guarded
increment enforce the half-open probe limit across workers. A protected manual reset is
audited. Tests use injected timestamps rather than sleep.
