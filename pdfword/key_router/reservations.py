from __future__ import annotations

from .models import RequestReservation


def owns_reservation(reservation: RequestReservation, token: str) -> bool:
    return bool(token) and reservation.reservation_token == token
