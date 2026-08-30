# Reservation Lifecycle

Supported states are `PENDING`, `RESERVED`, `DISPATCHING`, `SENT`,
`SETTLED_SUCCESS`, `SETTLED_FAILURE`, `RELEASED`, `UNCERTAIN`, `EXPIRED`, and
`CANCELLED`. Phase 1 `SUCCEEDED`/`FAILED` API names remain aliases and legacy stored
values are readable.

Local validation and timeout-before-send release reserved quota. A successful response
settles actual usage. Timeout-after-send becomes `UNCERTAIN` and is not retried by
default. Provider 400/422 policy can consume request quota independently of token and
monetary quota. Fencing tokens and state predicates prevent stale or double settlement.

Reconciliation uses `BEGIN IMMEDIATE`, fencing, and audited decisions. Expired
pre-dispatch reservations release quota; stale dispatched/sent reservations become
uncertain; old uncertain records remain for manual/provider reconciliation and are never
silently deleted.
