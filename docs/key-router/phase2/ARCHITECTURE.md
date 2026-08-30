# Architecture

The active data flow is:

1. Resolve an OCR/task policy.
2. Filter enabled capability endpoints by canonical mapping, verified capabilities,
   MIME/payload constraints, privacy policy, cost/latency constraints, exclusions, and
   circuit state.
3. Score endpoint candidates deterministically.
4. Filter accounts for the selected provider by enabled/state/cooldown/secret/quota
   policy/concurrency/privacy constraints.
5. Reserve quota atomically in SQLite.
6. Translate through the selected provider adapter and dispatch through an explicit
   transport.
7. Settle, release, or mark the reservation uncertain based on the dispatch boundary.
8. Normalize and audit failures, enforce retry budgets, and update circuit/account state.

Trust boundaries are the environment secret resolver, SQLite operational state,
provider transport, internal authenticated API, and untrusted document/image payload.
Canonical model mapping is represented by endpoint records but is not used as proof of
capability. Provider-wide declarations are transport metadata only.

`disabled` bypasses all Phase 2 selection and state mutation. `shadow` evaluates and
audits only. `active` is explicit and fail-closed.
