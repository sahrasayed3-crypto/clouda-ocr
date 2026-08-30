# Capability Registry

`capability_endpoints` stores provider endpoint evidence separately from provider
accounts. Records include canonical and provider model IDs, endpoint type, declared and
verified vision flags, input formats, MIME types, image/payload limits, context/output
limits, structured/stream/tool features, pricing fields, regions, evidence source,
timestamps, verification state, quality, latency, and enabled state.

Verification states are `declared`, `verified`, `failed_verification`, `stale`, and
`unknown`. Active vision routing requires verified vision by default. Declared-only
vision requires both an explicit relaxed policy and configuration; conflicting active
configuration fails closed.

Registry records may come from SQLite or safe application configuration. No broad model
catalog was hard-coded and no capability was live-verified in this phase.
