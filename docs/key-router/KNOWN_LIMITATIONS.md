# Known Limitations

- Provider adapters are not credential-tested.
- No real network health checks were executed.
- Provider-specific usage parsing and dynamic price ingestion are configuration-ready,
  not production-verified.
- Active mode is fail-closed but should remain disabled until real provider credentials,
  budgets, and account policies are reviewed.
- No Streamlit administration page was added; status is available through the protected
  internal API only.
- The system does not and must not rotate accounts to bypass platform terms or quotas.
