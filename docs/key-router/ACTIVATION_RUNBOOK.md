# Activation Runbook

Current state: dormant. Do not activate without real account ownership, provider terms
review, and test credentials.

Shadow activation:

```powershell
$env:KEY_ROUTER_MODE = "shadow"
$env:OPENROUTER_API_KEY_PRIMARY = "<local secret only>"
```

Active activation:

```powershell
$env:KEY_ROUTER_MODE = "active"
```

Active mode requires:

- enabled account row,
- present environment secret,
- supported modality,
- known price for free-only requests,
- budget for paid requests,
- non-open circuit,
- available concurrency,
- `ACTIVE` runtime state.

Rollback:

```powershell
$env:KEY_ROUTER_MODE = "disabled"
```
