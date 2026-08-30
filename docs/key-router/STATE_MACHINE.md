# Account State Machine

Supported states:

- `ACTIVE`
- `DRAINING`
- `CLOSED`
- `SWITCHING`
- `DISABLED`
- `ERROR`

Normal transition:

`ACTIVE -> DRAINING -> CLOSED -> SWITCHING -> ACTIVE`

Illegal transitions are rejected by `validate_transition`. Runtime updates use optimistic
version checks in `KeyRouterRepository.compare_and_set_runtime`; stale switch completion
is rejected when `switch_id` or `account_epoch` do not match.

Regression tests cover legal switching, stale switch rejection, and illegal transition
rejection.
