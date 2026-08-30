# Key Router Data Model

Schema version is now `4`. The migration is additive and uses
`CREATE TABLE IF NOT EXISTS`.

Tables:

- `provider_accounts`: provider account metadata and `secret_ref`; never stores key values.
- `account_runtime`: runtime state keyed by `router_id + provider + pool_id`.
- `quota_runtime`: quota windows keyed by `provider + pool_id + account_id + model + window_type + window_started_at`.
- `request_reservations`: pre-request reservations with `reservation_token` fencing.
- `key_router_audit`: sanitized event log.

Compatibility test: `test_database_upgrade_adds_key_router_tables_without_losing_existing_rows`.
