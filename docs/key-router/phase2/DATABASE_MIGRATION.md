# Database Migration

Schema version advances from 4 to 5 using additive changes only.

New tables: `capability_endpoints`, `key_router_circuits`, `key_router_errors`, and
`key_router_attempts`.

Additive account columns: state, organization ID, privacy allow-list JSON, and cooldown.
Additive reservation columns: request/parent/attempt lineage, provider model, policy,
reserved dimensions, independent consumption flags, dispatch/sent timestamps, fencing
token, and uncertainty reason.

No table or column is dropped and existing rows are preserved. Compatibility tests build
a version-4-style database with an existing provider account, initialize version 5, and
verify the row and new schema remain available.
