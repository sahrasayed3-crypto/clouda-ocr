from __future__ import annotations

from .enums import AccountState
from .exceptions import InvalidStateTransition

ALLOWED_ACCOUNT_TRANSITIONS: dict[AccountState, set[AccountState]] = {
    AccountState.ACTIVE: {
        AccountState.DRAINING,
        AccountState.DISABLED,
        AccountState.ERROR,
        AccountState.QUARANTINED,
        AccountState.COOLDOWN,
    },
    AccountState.DRAINING: {
        AccountState.CLOSED,
        AccountState.ERROR,
        AccountState.QUARANTINED,
    },
    AccountState.CLOSED: {AccountState.SWITCHING, AccountState.DISABLED},
    AccountState.SWITCHING: {
        AccountState.ACTIVE,
        AccountState.ERROR,
        AccountState.CLOSED,
    },
    AccountState.DISABLED: {AccountState.SWITCHING, AccountState.ACTIVE},
    AccountState.ERROR: {
        AccountState.CLOSED,
        AccountState.DISABLED,
        AccountState.SWITCHING,
        AccountState.QUARANTINED,
    },
    AccountState.QUARANTINED: {AccountState.DISABLED, AccountState.ACTIVE},
    AccountState.COOLDOWN: {
        AccountState.ACTIVE,
        AccountState.DISABLED,
        AccountState.QUARANTINED,
    },
}


def validate_transition(
    old_state: AccountState | str, new_state: AccountState | str
) -> None:
    old = (
        old_state
        if isinstance(old_state, AccountState)
        else AccountState(str(old_state))
    )
    new = (
        new_state
        if isinstance(new_state, AccountState)
        else AccountState(str(new_state))
    )
    if old == new:
        return
    if new not in ALLOWED_ACCOUNT_TRANSITIONS.get(old, set()):
        raise InvalidStateTransition(
            f"Invalid account transition: {old.value} -> {new.value}"
        )
