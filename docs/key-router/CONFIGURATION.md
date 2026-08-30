# Key Router Configuration

Default:

```text
KEY_ROUTER_MODE=disabled
KEY_ROUTER_ID=clouda-local
KEY_ROUTER_FREE_ONLY_DEFAULT=true
KEY_ROUTER_ALLOW_PAID_DEFAULT=false
```

Account records use `secret_ref`, for example `OPENROUTER_API_KEY_PRIMARY`. The actual
value must be placed in the local environment or an untracked `.env`, not in Git.

To test shadow mode:

1. Add an authorized account record to `provider_accounts`.
2. Set the referenced environment variable locally.
3. Set `KEY_ROUTER_MODE=shadow`.
4. Verify sanitized events under `/internal/key-router/status`.

To activate later:

1. Confirm provider terms and account authorization.
2. Configure budgets and quotas.
3. Run provider-specific tests without exposing secrets.
4. Set `KEY_ROUTER_MODE=active`.

Rollback is setting `KEY_ROUTER_MODE=disabled`.
