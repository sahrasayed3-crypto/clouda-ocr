import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone

from .constants import (
    MODEL_ROUTER_CLAUDE_SONNET,
    MODEL_ROUTER_GEMINI_PRO,
    MODEL_ROUTER_GPT5,
    MODEL_ROUTER_GPT5_MINI,
    MODEL_ROUTER_STORE_FILE,
    SECRETS_DIR,
    VISION_MODEL_PRIORITY,
)


@dataclass(frozen=True)
class PageSignals:
    page_no: int
    total_pages: int
    born_digital: bool
    scanned_page: bool
    complexity: float
    layout_complexity: float
    arabic_ratio: float
    english_ratio: float
    has_tables: bool
    has_images: bool
    has_equations: bool
    ocr_confidence_hint: float


@dataclass(frozen=True)
class ModelAttemptPlan:
    model: str
    reason: str
    target_score: float
    dpi: int
    aggressive: bool
    max_tokens: int
    max_width: int
    jpeg_quality: int
    force_judge: bool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_store() -> dict:
    return {
        "updated_at": _utc_now_iso(),
        "signature_cache": {},
        "model_stats": {},
        "history": [],
    }


def _safe_load(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return _default_store()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_store()
        data.setdefault("signature_cache", {})
        data.setdefault("model_stats", {})
        data.setdefault("history", [])
        return data
    except Exception:
        return _default_store()


def _safe_write(path: str, payload: dict) -> None:
    os.makedirs(SECRETS_DIR, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class AIModelRouter:
    MIN_SAMPLES_BEFORE_REORDER = 20

    def __init__(
        self,
        store_path: str = MODEL_ROUTER_STORE_FILE,
        preferred_models: list[str] | None = None,
    ) -> None:
        self.store_path = store_path
        self.store = _safe_load(store_path)
        self.dynamic_preferred = bool(preferred_models)
        defaults = list(VISION_MODEL_PRIORITY)
        self.priority_models = list(dict.fromkeys((preferred_models or []) + defaults))

    def save(self) -> None:
        self.store["updated_at"] = _utc_now_iso()
        _safe_write(self.store_path, self.store)

    def build_attempts(
        self, signals: PageSignals, doc_signature: str, mode: str
    ) -> list[ModelAttemptPlan]:
        target = self._target_score(signals, mode)
        signature = self._signature_key(doc_signature=doc_signature, signals=signals)

        cached_row = self.store.get("signature_cache", {}).get(signature, {})
        cached_best = (
            cached_row.get("best_model", "")
            if int(cached_row.get("samples", 0)) >= self.MIN_SAMPLES_BEFORE_REORDER
            else ""
        )
        if cached_best and random.random() < 0.10:
            cached_best = ""
        initial = self._select_initial_model(signals, cached_best)
        chain = self._escalation_chain(initial)

        attempts: list[ModelAttemptPlan] = []
        for idx, model in enumerate(chain):
            complexity_boost = int(min(220, max(0, signals.complexity * 260)))
            is_heavy = model in {
                MODEL_ROUTER_GEMINI_PRO,
                MODEL_ROUTER_CLAUDE_SONNET,
                MODEL_ROUTER_GPT5,
            }
            layout_bias = (
                90
                if (
                    signals.layout_complexity >= 0.72
                    and model == MODEL_ROUTER_CLAUDE_SONNET
                )
                else 0
            )
            eq_bias = (
                80 if (signals.has_equations and model == MODEL_ROUTER_GPT5) else 0
            )
            attempts.append(
                ModelAttemptPlan(
                    model=model,
                    reason=self._reason_for_model(model, signals, cached_best),
                    target_score=min(
                        96.0, target + (idx * 1.9 if not is_heavy else idx * 1.2)
                    ),
                    dpi=620
                    + complexity_boost
                    + (90 if is_heavy else 0)
                    + layout_bias
                    + eq_bias,
                    aggressive=signals.scanned_page
                    or signals.complexity >= 0.55
                    or signals.has_equations,
                    max_tokens=4096,
                    max_width=1500 + int(min(280, signals.complexity * 260)),
                    jpeg_quality=84 if signals.complexity < 0.55 else 80,
                    force_judge=(signals.complexity >= 0.58 or idx > 0 or is_heavy),
                )
            )

        self._push_history(
            {
                "ts": _utc_now_iso(),
                "event": "route_decision",
                "signature": signature,
                "page_no": signals.page_no,
                "selected_model": (
                    attempts[0].model if attempts else MODEL_ROUTER_GPT5_MINI
                ),
                "reason": attempts[0].reason if attempts else "default",
                "target": target,
            }
        )
        return attempts

    def record_outcome(
        self,
        doc_signature: str,
        signals: PageSignals,
        model: str,
        score: float | None,
        success: bool,
    ) -> None:
        signature = self._signature_key(doc_signature=doc_signature, signals=signals)
        stats = self.store.setdefault("model_stats", {}).setdefault(
            model, {"ok": 0, "fail": 0, "score_sum": 0.0, "score_n": 0}
        )
        if success:
            stats["ok"] = int(stats.get("ok", 0)) + 1
        else:
            stats["fail"] = int(stats.get("fail", 0)) + 1
        if score is not None:
            stats["score_sum"] = float(stats.get("score_sum", 0.0)) + float(
                max(0.0, min(100.0, score))
            )
            stats["score_n"] = int(stats.get("score_n", 0)) + 1

        cache = self.store.setdefault("signature_cache", {}).setdefault(
            signature,
            {"best_model": model, "best_score": 0.0, "samples": 0},
        )
        cache["samples"] = int(cache.get("samples", 0)) + 1
        if (
            score is not None
            and success
            and float(score) >= float(cache.get("best_score", 0.0))
        ):
            cache["best_model"] = model
            cache["best_score"] = float(score)

    def _target_score(self, signals: PageSignals, mode: str) -> float:
        base = 78.0
        m = (mode or "").lower()
        if m == "max_accuracy":
            base = 88.0
        elif m in {"balanced", "hyper"}:
            base = 82.0
        if signals.scanned_page:
            base += 2.0
        if signals.has_equations:
            base += 4.0
        base += signals.layout_complexity * 5.0
        base += signals.complexity * 6.0
        return max(72.0, min(96.0, base))

    def _select_initial_model(self, signals: PageSignals, cached_best: str) -> str:
        if cached_best in self.priority_models:
            return cached_best
        if signals.has_equations:
            return MODEL_ROUTER_GPT5
        if signals.layout_complexity >= 0.74:
            return MODEL_ROUTER_CLAUDE_SONNET
        if signals.arabic_ratio >= 0.55 and (
            signals.scanned_page or signals.ocr_confidence_hint < 68
        ):
            return MODEL_ROUTER_GPT5
        return MODEL_ROUTER_GPT5_MINI

    def _escalation_chain(self, initial: str) -> list[str]:
        if initial in self.priority_models:
            index = self.priority_models.index(initial)
            return self.priority_models[index:] + self.priority_models[:index]
        if initial == MODEL_ROUTER_GPT5_MINI:
            return [
                MODEL_ROUTER_GPT5_MINI,
                MODEL_ROUTER_GPT5,
                MODEL_ROUTER_GEMINI_PRO,
                MODEL_ROUTER_CLAUDE_SONNET,
            ]
        if initial == MODEL_ROUTER_GPT5:
            return [
                MODEL_ROUTER_GPT5,
                MODEL_ROUTER_GEMINI_PRO,
                MODEL_ROUTER_CLAUDE_SONNET,
            ]
        if initial == MODEL_ROUTER_GEMINI_PRO:
            return [
                MODEL_ROUTER_GEMINI_PRO,
                MODEL_ROUTER_GPT5,
                MODEL_ROUTER_CLAUDE_SONNET,
            ]
        if initial == MODEL_ROUTER_CLAUDE_SONNET:
            return [MODEL_ROUTER_CLAUDE_SONNET]
        return [
            MODEL_ROUTER_GPT5_MINI,
            MODEL_ROUTER_GPT5,
            MODEL_ROUTER_GEMINI_PRO,
            MODEL_ROUTER_CLAUDE_SONNET,
        ]

    def _reason_for_model(
        self, model: str, signals: PageSignals, cached_best: str
    ) -> str:
        if cached_best and model == cached_best:
            return "cache_hit_similar_document"
        if model == MODEL_ROUTER_GPT5_MINI:
            return "default_quality_cost_balanced"
        if model == MODEL_ROUTER_GPT5:
            return "arabic_difficult_or_formula_page"
        if model == MODEL_ROUTER_GEMINI_PRO:
            return "text_fidelity_fallback_for_complex_scan"
        if model == MODEL_ROUTER_CLAUDE_SONNET and signals.layout_complexity >= 0.74:
            return "complex_layout_resolution"
        return "final_fallback_for_hard_page"

    def _signature_key(self, doc_signature: str, signals: PageSignals) -> str:
        c = f"{int(signals.complexity * 10)}"
        lc = f"{int(signals.layout_complexity * 10)}"
        ar = f"{int(signals.arabic_ratio * 10)}"
        en = f"{int(signals.english_ratio * 10)}"
        bd = "bd1" if signals.born_digital else "bd0"
        sc = "sc1" if signals.scanned_page else "sc0"
        t = "t1" if signals.has_tables else "t0"
        i = "i1" if signals.has_images else "i0"
        e = "e1" if signals.has_equations else "e0"
        conf = f"o{int(signals.ocr_confidence_hint // 10)}"
        return f"{doc_signature}|{bd}|{sc}|c{c}|lc{lc}|ar{ar}|en{en}|{t}|{i}|{e}|{conf}"

    def _push_history(self, row: dict) -> None:
        history = self.store.setdefault("history", [])
        history.append(row)
        if len(history) > 1200:
            del history[: len(history) - 1200]
