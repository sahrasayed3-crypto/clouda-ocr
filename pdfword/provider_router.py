import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from .constants import SECRETS_DIR

PROVIDER_ROUTER_STORE = os.path.join(SECRETS_DIR, "provider_router_stats.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_store() -> dict:
    return {
        "updated_at": _utc_now(),
        "providers": {},
        "signatures": {},
    }


def _safe_load(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return _default_store()
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return _default_store()
        payload.setdefault("providers", {})
        payload.setdefault("signatures", {})
        return payload
    except Exception:
        return _default_store()


def _safe_write(path: str, payload: dict) -> None:
    os.makedirs(SECRETS_DIR, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass(frozen=True)
class ProviderChoice:
    name: str
    reason: str


class ProviderRouter:
    def __init__(self, store_path: str = PROVIDER_ROUTER_STORE) -> None:
        self.store_path = store_path
        self.store = _safe_load(store_path)

    def save(self) -> None:
        self.store["updated_at"] = _utc_now()
        _safe_write(self.store_path, self.store)

    def available_providers(self, primary_api_key: str) -> list[str]:
        providers: list[str] = []
        if (primary_api_key or "").strip():
            providers.append("openrouter")
        if os.getenv("TOGETHER_API_KEY", "").strip():
            providers.append("together")
        if os.getenv("FIREWORKS_API_KEY", "").strip():
            providers.append("fireworks")
        return providers

    def choose_provider(
        self, candidates: list[str], signature: str = "default"
    ) -> ProviderChoice:
        if not candidates:
            return ProviderChoice(name="openrouter", reason="no_candidates")
        if len(candidates) == 1:
            return ProviderChoice(name=candidates[0], reason="single_candidate")

        best_name = candidates[0]
        best_score = float("-inf")
        for name in candidates:
            score = self._score_provider(name=name, signature=signature)
            if score > best_score:
                best_score = score
                best_name = name
        return ProviderChoice(name=best_name, reason=f"score={best_score:.2f}")

    def choose_backup_provider(
        self, current: str, candidates: list[str], signature: str = "default"
    ) -> str | None:
        pool = [p for p in candidates if p != current]
        if not pool:
            return None
        return self.choose_provider(pool, signature=signature).name

    def record_success(
        self,
        provider: str,
        latency_ms: float,
        quality_score: float | None,
        signature: str = "default",
    ) -> None:
        row = self._provider_row(provider)
        row["ok"] = int(row.get("ok", 0)) + 1
        row["latency_ms_sum"] = float(row.get("latency_ms_sum", 0.0)) + max(
            0.0, float(latency_ms)
        )
        if quality_score is not None:
            row["quality_sum"] = float(row.get("quality_sum", 0.0)) + max(
                0.0, min(100.0, float(quality_score))
            )
            row["quality_n"] = int(row.get("quality_n", 0)) + 1
        self._record_signature(provider=provider, signature=signature, ok=True)

    def record_failure(
        self, provider: str, error_kind: str, signature: str = "default"
    ) -> None:
        row = self._provider_row(provider)
        row["fail"] = int(row.get("fail", 0)) + 1
        key = (error_kind or "unknown").strip().lower() or "unknown"
        fail_by_kind = row.setdefault("fail_by_kind", {})
        fail_by_kind[key] = int(fail_by_kind.get(key, 0)) + 1
        self._record_signature(provider=provider, signature=signature, ok=False)

    def _provider_row(self, provider: str) -> dict:
        providers = self.store.setdefault("providers", {})
        row = providers.setdefault(
            provider,
            {
                "ok": 0,
                "fail": 0,
                "latency_ms_sum": 0.0,
                "quality_sum": 0.0,
                "quality_n": 0,
                "fail_by_kind": {},
            },
        )
        return row

    def _record_signature(self, provider: str, signature: str, ok: bool) -> None:
        sigs = self.store.setdefault("signatures", {})
        sig = sigs.setdefault(signature or "default", {})
        row = sig.setdefault(provider, {"ok": 0, "fail": 0})
        if ok:
            row["ok"] = int(row.get("ok", 0)) + 1
        else:
            row["fail"] = int(row.get("fail", 0)) + 1

    def _score_provider(self, name: str, signature: str) -> float:
        row = self.store.get("providers", {}).get(name, {})
        ok = int(row.get("ok", 0))
        fail = int(row.get("fail", 0))
        attempts = ok + fail

        if attempts <= 0:
            base = 55.0
        else:
            success_rate = ok / max(1, attempts)
            avg_latency = float(row.get("latency_ms_sum", 0.0)) / max(1, ok)
            avg_quality = float(row.get("quality_sum", 0.0)) / max(
                1, int(row.get("quality_n", 0))
            )
            latency_component = max(0.0, 25.0 - min(25.0, avg_latency / 130.0))
            base = (success_rate * 50.0) + latency_component + (avg_quality * 0.25)

        sig_row = (
            self.store.get("signatures", {})
            .get(signature or "default", {})
            .get(name, {})
        )
        sig_ok = int(sig_row.get("ok", 0))
        sig_fail = int(sig_row.get("fail", 0))
        if sig_ok + sig_fail > 0:
            sig_rate = sig_ok / max(1, sig_ok + sig_fail)
            base += (sig_rate - 0.5) * 12.0

        return base
