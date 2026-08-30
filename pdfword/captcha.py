from __future__ import annotations

from typing import Protocol


class CaptchaVerifier(Protocol):
    def verify(self, token: str | None, *, risk_score: int = 0) -> bool: ...


class FakeCaptchaVerifier:
    def verify(self, token: str | None, *, risk_score: int = 0) -> bool:
        return risk_score < 3 or token == "test-captcha-ok"
