from dataclasses import dataclass


@dataclass(frozen=True)
class UiSystemStatus:
    server: bool = False
    redis: bool = False
    worker_state: str = "offline"
    cloud: bool | None = False
    worker_count: int = 0
    service_status_outdated: bool = False

    def as_dict(self) -> dict:
        return {
            "server": self.server,
            "redis": self.redis,
            "worker_state": self.worker_state,
            "cloud": self.cloud,
            "worker_count": self.worker_count,
            "service_status_outdated": self.service_status_outdated,
        }


def parse_health_payload(
    payload: dict,
    *,
    legacy_status: bool = False,
    redis_available: bool | None = None,
) -> UiSystemStatus:
    workers = payload.get("workers") or []
    server_available = payload.get("status") == "ok"
    redis_is_available = (
        bool(redis_available)
        if redis_available is not None
        else bool(payload.get("redis_available", workers))
    )
    states = {str(worker.get("state", "ready")).lower() for worker in workers}
    if server_available and bool(payload.get("local_processing_enabled")):
        worker_state = "ready"
    elif legacy_status and server_available and redis_is_available:
        worker_state = "ready"
    elif not workers:
        worker_state = "offline"
    elif "busy" in states or "working" in states:
        worker_state = "busy"
    else:
        worker_state = "ready"
    return UiSystemStatus(
        server=server_available,
        redis=redis_is_available,
        worker_state=worker_state,
        cloud=None if legacy_status else bool(payload.get("cloud_available")),
        worker_count=max(1, len(workers)) if worker_state != "offline" else 0,
        service_status_outdated=legacy_status,
    )


def fetch_system_status(
    base_url: str,
    worker_api_key: str,
    requester,
    timeout: int = 5,
    redis_available: bool | None = None,
) -> UiSystemStatus:
    if not base_url:
        return UiSystemStatus()
    headers = {"X-Worker-API-Key": worker_api_key} if worker_api_key else {}
    try:
        response = None
        if worker_api_key:
            try:
                response = requester.get(
                    f"{base_url.rstrip('/')}/internal/health",
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
            except Exception:
                response = None
        if response is None:
            response = requester.get(
                f"{base_url.rstrip('/')}/health",
                timeout=timeout,
            )
            response.raise_for_status()
            return parse_health_payload(
                response.json(),
                redis_available=redis_available,
            )
        legacy_status = False
        try:
            status_response = requester.get(
                f"{base_url.rstrip('/')}/internal/workers/status",
                headers=headers,
                timeout=timeout,
            )
            legacy_status = status_response.status_code == 404
        except Exception:
            pass
        return parse_health_payload(
            response.json(),
            legacy_status=legacy_status,
            redis_available=redis_available,
        )
    except Exception:
        return UiSystemStatus()
