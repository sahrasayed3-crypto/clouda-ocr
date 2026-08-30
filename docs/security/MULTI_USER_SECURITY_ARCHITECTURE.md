# Multi-User Security Architecture

## Trust Boundaries

- Firebase/Google Identity verifies external identity.
- SQLite is authoritative for internal user ID, role, status, ownership, limits, sessions, and audit events.
- FastAPI dependencies enforce authentication, role checks, CSRF, and owner-scoped access.
- Streamlit presents product surfaces but does not grant authorization.

## Tenant Isolation

Every user-owned conversion receives an immutable `owner_user_id`. User APIs query by owner by default. Admin cross-user reads use explicit admin routes and emit audit events. Guest jobs use a separate `guest_scope_id` and are never treated as authenticated accounts.

## Storage

Storage paths are server-constructed:

```text
users/<internal_user_uuid>/uploads/
users/<internal_user_uuid>/processing/
users/<internal_user_uuid>/outputs/
users/<internal_user_uuid>/temporary/
guests/<guest_scope_id>/uploads/
guests/<guest_scope_id>/outputs/
```

Email addresses and client-supplied IDs are not used as directory names. Paths are canonicalized and checked for containment before use.

## Guest Trial

Guest trial defaults are PDF-only, 10 MiB, 5 pages, one active job, local/free provider policy, no paid dispatch, no history, unguessable IDs, and short TTL cleanup.

## Deferred Production Work

Public production still requires HTTPS, DNS, reverse proxy, production Firebase, real secrets provisioning, production database/backups, optional Redis hardening, monitoring, incident response, and staging security validation.
