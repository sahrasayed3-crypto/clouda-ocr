import difflib
import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any

from .constants import LEARNING_STORE_FILE, SECRETS_DIR

_TOKEN_PATTERN = re.compile(r"[\u0600-\u06FFA-Za-z0-9]+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_store() -> dict:
    return {
        "version": 1,
        "updated_at": _utc_now_iso(),
        "stats": {"CLEAR": {}, "COMPLEX": {}, "UNKNOWN": {}},
        "corrections": {},
        "ai_suggestions": {},
        "runtime_errors": {"by_key": {}, "history": []},
        "history": [],
    }


def _safe_load_json(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return _default_store()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_store()
        data.setdefault("stats", {"CLEAR": {}, "COMPLEX": {}, "UNKNOWN": {}})
        data.setdefault("corrections", {})
        data.setdefault("ai_suggestions", {})
        data.setdefault("history", [])
        data.setdefault("runtime_errors", {"by_key": {}, "history": []})
        return data
    except Exception:
        return _default_store()


def _safe_write_json(path: str, payload: dict) -> None:
    os.makedirs(SECRETS_DIR, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text or "")


def _is_valid_pair(src: str, dst: str) -> bool:
    if not src or not dst or src == dst:
        return False
    if len(src) < 2 or len(dst) < 2:
        return False
    if src.isdigit() or dst.isdigit():
        return False
    if abs(len(src) - len(dst)) > 4:
        return False
    return True


def _classify_runtime_error(error_text: str) -> str:
    text = (error_text or "").lower()
    if not text:
        return "unknown"
    if "payment required" in text or "402" in text:
        return "payment_required"
    if "rate limit" in text or "429" in text:
        return "rate_limit"
    if "no endpoints found" in text or "غير متاح" in text:
        return "model_endpoint_unavailable"
    if "empty response" in text or "did not include usable text content" in text:
        return "empty_response"
    if "400 client error" in text or "bad request" in text:
        return "bad_request"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(code in text for code in ("500", "502", "503", "504", "transient http")):
        return "server_transient"
    if any(mark in text for mark in ("connection", "network", "dns", "ssl")):
        return "network"
    return "unknown"


class SelfLearningEngine:
    def __init__(self, path: str = LEARNING_STORE_FILE) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._store = _safe_load_json(path)

    def save(self) -> None:
        with self._lock:
            self._store["updated_at"] = _utc_now_iso()
            _safe_write_json(self.path, self._store)

    def get_prompt_hints(self, max_items: int = 10, min_count: int = 2) -> str:
        with self._lock:
            items = []
            for key, count in self._store.get("corrections", {}).items():
                if count < min_count or "\t" not in key:
                    continue
                src, dst = key.split("\t", 1)
                if _is_valid_pair(src, dst):
                    items.append((src, dst, int(count)))
            items.sort(key=lambda x: x[2], reverse=True)
            items = items[:max_items]

        if not items:
            return ""
        lines = [
            "Previously confirmed OCR corrections (apply only if they match image context exactly):"
        ]
        for src, dst, count in items:
            lines.append(f"- {src} -> {dst} (confidence {count})")
        return "\n".join(lines)

    def get_preferred_attempt(
        self, quality_label: str, allowed_models: set[str] | None = None
    ) -> dict | None:
        q = (quality_label or "UNKNOWN").upper()
        with self._lock:
            bucket = self._store.get("stats", {}).get(q, {})
            best = None
            best_score = -1.0
            for key, info in bucket.items():
                try:
                    model, dpi_str, aggressive_str = key.split("|")
                    if allowed_models and model not in allowed_models:
                        continue
                    count = int(info.get("count", 0))
                    if count < 2:
                        continue
                    avg_score = float(info.get("score_sum", 0.0)) / max(1, count)
                    blended = avg_score + min(6.0, count * 0.4)
                    if blended > best_score:
                        best_score = blended
                        best = {
                            "model": model,
                            "dpi": max(300, min(1200, int(dpi_str))),
                            "aggressive": aggressive_str == "1",
                        }
                except Exception:
                    continue
        return best

    def record_page_result(
        self,
        *,
        quality_label: str,
        model: str,
        dpi: int,
        aggressive: bool,
        score: float | None,
        page_no: int,
    ) -> None:
        q = (quality_label or "UNKNOWN").upper()
        q = q if q in {"CLEAR", "COMPLEX"} else "UNKNOWN"
        stat_key = f"{model}|{int(dpi)}|{1 if aggressive else 0}"

        with self._lock:
            stats = self._store.setdefault("stats", {}).setdefault(q, {})
            row = stats.setdefault(
                stat_key, {"count": 0, "score_sum": 0.0, "best": 0.0}
            )
            row["count"] = int(row.get("count", 0)) + 1
            if score is not None:
                s = float(max(0.0, min(100.0, score)))
                row["score_sum"] = float(row.get("score_sum", 0.0)) + s
                row["best"] = max(float(row.get("best", 0.0)), s)

            history = self._store.setdefault("history", [])
            history.append(
                {
                    "ts": _utc_now_iso(),
                    "quality": q,
                    "model": model,
                    "dpi": int(dpi),
                    "aggressive": bool(aggressive),
                    "score": None if score is None else float(score),
                    "page_no": int(page_no),
                }
            )
            if len(history) > 600:
                del history[: len(history) - 600]

    def record_runtime_error(
        self,
        *,
        quality_label: str,
        model: str,
        dpi: int,
        aggressive: bool,
        error_text: str,
        page_no: int | None = None,
    ) -> str:
        q = (quality_label or "UNKNOWN").upper()
        q = q if q in {"CLEAR", "COMPLEX"} else "UNKNOWN"
        signature = _classify_runtime_error(error_text)
        key = f"{signature}|{q}|{model}"

        with self._lock:
            runtime = self._store.setdefault(
                "runtime_errors", {"by_key": {}, "history": []}
            )
            by_key = runtime.setdefault("by_key", {})
            row = by_key.setdefault(
                key,
                {
                    "count": 0,
                    "signature": signature,
                    "quality": q,
                    "model": model,
                    "last_error": "",
                    "last_dpi": 0,
                    "last_aggressive": False,
                },
            )
            row["count"] = int(row.get("count", 0)) + 1
            row["last_error"] = (error_text or "")[:500]
            row["last_dpi"] = int(dpi)
            row["last_aggressive"] = bool(aggressive)
            row["updated_at"] = _utc_now_iso()

            history = runtime.setdefault("history", [])
            history.append(
                {
                    "ts": _utc_now_iso(),
                    "signature": signature,
                    "quality": q,
                    "model": model,
                    "dpi": int(dpi),
                    "aggressive": bool(aggressive),
                    "page_no": None if page_no is None else int(page_no),
                }
            )
            if len(history) > 800:
                del history[: len(history) - 800]
        return signature

    def get_adaptive_profile(
        self, quality_label: str, allowed_models: set[str] | None = None
    ) -> dict:
        q = (quality_label or "UNKNOWN").upper()
        q = q if q in {"CLEAR", "COMPLEX"} else "UNKNOWN"
        profile: dict[str, Any] = {
            "max_width_scale": 1.0,
            "jpeg_quality_delta": 0,
            "force_aggressive": False,
            "avoid_models": [],
            "notes": [],
        }
        with self._lock:
            runtime = self._store.get("runtime_errors", {})
            by_key = runtime.get("by_key", {})
            sig_counts: dict[str, int] = {}
            model_sig_counts: dict[tuple[str, str], int] = {}

            for row in by_key.values():
                try:
                    row_q = str(row.get("quality", "UNKNOWN")).upper()
                    if row_q not in {q, "UNKNOWN"}:
                        continue
                    sig = str(row.get("signature", "unknown"))
                    model = str(row.get("model", ""))
                    count = int(row.get("count", 0))
                    if count <= 0:
                        continue
                    sig_counts[sig] = sig_counts.get(sig, 0) + count
                    if model:
                        model_sig_counts[(model, sig)] = (
                            model_sig_counts.get((model, sig), 0) + count
                        )
                except Exception:
                    continue

        req_reduce = sig_counts.get("rate_limit", 0) + sig_counts.get("timeout", 0)
        if req_reduce >= 3:
            profile["max_width_scale"] = min(profile["max_width_scale"], 0.86)
            profile["jpeg_quality_delta"] = min(profile["jpeg_quality_delta"], -8)
            profile["notes"].append("reduce_upload_load")

        req_compact = sig_counts.get("bad_request", 0) + sig_counts.get(
            "empty_response", 0
        )
        if req_compact >= 2:
            profile["max_width_scale"] = min(profile["max_width_scale"], 0.78)
            profile["jpeg_quality_delta"] = min(profile["jpeg_quality_delta"], -12)
            profile["force_aggressive"] = True
            profile["notes"].append("compact_payload")

        for (model, sig), count in model_sig_counts.items():
            if sig in {"model_endpoint_unavailable", "payment_required"} and count >= 2:
                if allowed_models and model not in allowed_models:
                    continue
                profile["avoid_models"].append(model)

        profile["avoid_models"] = sorted(set(profile["avoid_models"]))
        return profile

    def learn_from_revision(self, before_text: str, after_text: str) -> None:
        """Legacy trusted-revision collector; callers must provide human-approved text."""
        before_tokens = _tokenize(before_text)
        after_tokens = _tokenize(after_text)
        if not before_tokens or not after_tokens:
            return

        sm = difflib.SequenceMatcher(a=before_tokens, b=after_tokens, autojunk=False)
        updates: dict[str, int] = {}
        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            if tag != "replace":
                continue
            left = before_tokens[a0:a1]
            right = after_tokens[b0:b1]
            if len(left) != len(right) or len(left) > 6:
                continue
            for src, dst in zip(left, right):
                if not _is_valid_pair(src, dst):
                    continue
                key = f"{src}\t{dst}"
                updates[key] = updates.get(key, 0) + 1

        if not updates:
            return
        with self._lock:
            corr = self._store.setdefault("corrections", {})
            for key, count in updates.items():
                corr[key] = int(corr.get(key, 0)) + int(count)
            if len(corr) > 3000:
                ranked = sorted(corr.items(), key=lambda x: int(x[1]), reverse=True)[
                    :2500
                ]
                self._store["corrections"] = {k: int(v) for k, v in ranked}

    def record_ai_suggestion(self, before_text: str, after_text: str) -> None:
        """Record AI revisions for diagnostics only; never expose them as trusted hints."""
        before_tokens = _tokenize(before_text)
        after_tokens = _tokenize(after_text)
        matcher = difflib.SequenceMatcher(
            a=before_tokens, b=after_tokens, autojunk=False
        )
        with self._lock:
            suggestions = self._store.setdefault("ai_suggestions", {})
            for tag, a0, a1, b0, b1 in matcher.get_opcodes():
                if tag != "replace":
                    continue
                left, right = before_tokens[a0:a1], after_tokens[b0:b1]
                if len(left) != len(right) or len(left) > 6:
                    continue
                for src, dst in zip(left, right):
                    if _is_valid_pair(src, dst):
                        key = f"{src}\t{dst}"
                        suggestions[key] = int(suggestions.get(key, 0)) + 1

    def apply_auto_corrections(
        self, text: str, min_count: int = 5, max_changes: int = 30
    ) -> str:
        original = text or ""
        if not original:
            return original

        with self._lock:
            candidates = []
            for key, count in self._store.get("corrections", {}).items():
                if count < min_count or "\t" not in key:
                    continue
                src, dst = key.split("\t", 1)
                if _is_valid_pair(src, dst):
                    candidates.append((src, dst, int(count)))
            candidates.sort(key=lambda x: x[2], reverse=True)

        updated = original
        changes = 0
        for src, dst, _ in candidates:
            if changes >= max_changes:
                break
            pattern = re.compile(
                rf"(?<![\w\u0600-\u06FF]){re.escape(src)}(?![\w\u0600-\u06FF])"
            )
            updated, n = pattern.subn(dst, updated)
            changes += n

        return updated

    def get_summary(self) -> dict:
        with self._lock:
            stats = self._store.get("stats", {})
            total_history = len(self._store.get("history", []))
            total_corr = len(self._store.get("corrections", {}))
            runtime = self._store.get("runtime_errors", {})
            runtime_by_key = runtime.get("by_key", {})
            runtime_history = runtime.get("history", [])
            last_update = self._store.get("updated_at", "")
            return {
                "history_events": total_history,
                "corrections": total_corr,
                "ai_suggestions": len(self._store.get("ai_suggestions", {})),
                "runtime_error_keys": len(runtime_by_key),
                "runtime_error_events": len(runtime_history),
                "last_update": last_update,
                "stats_buckets": {k: len(v) for k, v in stats.items()},
            }


def load_learning_summary(path: str = LEARNING_STORE_FILE) -> dict:
    engine = SelfLearningEngine(path=path)
    return engine.get_summary()
