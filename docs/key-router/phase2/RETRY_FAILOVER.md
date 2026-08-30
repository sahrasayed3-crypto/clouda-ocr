# Retry and Failover

Retry state records parent request lineage, deterministic attempt IDs, candidate/model/
provider exclusions, attempt count, account/provider failovers, total deadline, and a
retry budget. Backoff is exponential with deterministic jitter and honors Retry-After.

There are no unlimited or circular retries. Failed candidates are excluded. Quarantined
accounts cannot be selected. Timeout-after-send stops automatically unless a policy
explicitly permits resending. When cloud routes are exhausted, a usable local OCR result
is preserved and existing 90% quality/manual-review behavior is respected.
