from __future__ import annotations

from typing import Any

import requests

from .base import ProviderRequest, ProviderResponse


class TransportFailure(RuntimeError):
    def __init__(self, message: str, *, request_sent: bool) -> None:
        super().__init__(message)
        self.request_sent = request_sent


class RequestsTransport:
    """Synchronous production transport; tests inject deterministic fake transports."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def __call__(self, request: ProviderRequest) -> ProviderResponse:
        try:
            response = self.session.request(
                request.method,
                request.url,
                headers=request.headers,
                json=request.json_body,
                timeout=request.timeout_seconds,
            )
        except (requests.ConnectTimeout, requests.ConnectionError) as exc:
            raise TransportFailure(
                "Provider connection failed before a response was received",
                request_sent=False,
            ) from exc
        except requests.ReadTimeout as exc:
            raise TransportFailure(
                "Provider response timed out after dispatch",
                request_sent=True,
            ) from exc
        except requests.RequestException as exc:
            raise TransportFailure(
                "Provider transport failed",
                request_sent=exc.response is not None,
            ) from exc
        try:
            body: Any = response.json()
        except ValueError:
            body = {"error": {"message": response.text[:500]}}
        return ProviderResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            json_body=body,
        )
