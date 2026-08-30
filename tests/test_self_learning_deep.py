import json

from pdfword.self_learning import (
    SelfLearningEngine,
    _classify_runtime_error,
    _safe_load_json,
)


def test_safe_load_json_handles_missing_invalid_and_non_dict(tmp_path):
    missing = tmp_path / "missing.json"
    assert _safe_load_json(str(missing))["version"] == 1

    broken = tmp_path / "broken.json"
    broken.write_text("{bad", encoding="utf-8")
    assert _safe_load_json(str(broken))["stats"]

    scalar = tmp_path / "scalar.json"
    scalar.write_text(json.dumps(["not", "dict"]), encoding="utf-8")
    assert _safe_load_json(str(scalar))["corrections"] == {}


def test_learning_prompt_preferred_attempt_save_and_summary(tmp_path):
    store = tmp_path / "learning.json"
    engine = SelfLearningEngine(path=str(store))
    engine.learn_from_revision("hello wurld\nقلم", "hello world\nعلم")
    engine.learn_from_revision("hello wurld", "hello world")
    engine.record_page_result(
        quality_label="clear",
        model="model-a",
        dpi=500,
        aggressive=True,
        score=97.5,
        page_no=2,
    )
    engine.record_page_result(
        quality_label="clear",
        model="model-a",
        dpi=500,
        aggressive=True,
        score=88.0,
        page_no=3,
    )

    hints = engine.get_prompt_hints(min_count=2)
    assert "wurld -> world" in hints
    preferred = engine.get_preferred_attempt("clear", allowed_models={"model-a"})
    assert preferred == {"model": "model-a", "dpi": 500, "aggressive": True}
    corrected = engine.apply_auto_corrections("hello wurld", min_count=2)
    assert corrected == "hello world"

    engine.save()
    reloaded = SelfLearningEngine(path=str(store))
    summary = reloaded.get_summary()
    assert summary["history_events"] == 2
    assert summary["corrections"] >= 1


def test_ai_suggestions_are_diagnostic_only(tmp_path):
    engine = SelfLearningEngine(path=str(tmp_path / "learning.json"))
    engine.record_ai_suggestion("bad token", "good token")
    assert engine.get_prompt_hints(min_count=1) == ""
    assert engine.get_summary()["ai_suggestions"] == 1


def test_runtime_error_classification_and_adaptive_profile(tmp_path):
    cases = {
        "402 payment required": "payment_required",
        "429 rate limit": "rate_limit",
        "no endpoints found": "model_endpoint_unavailable",
        "empty response": "empty_response",
        "400 client error": "bad_request",
        "request timed out": "timeout",
        "503 transient http": "server_transient",
        "ssl connection failed": "network",
        "": "unknown",
    }
    for text, expected in cases.items():
        assert _classify_runtime_error(text) == expected

    engine = SelfLearningEngine(path=str(tmp_path / "learning.json"))
    for _ in range(2):
        engine.record_runtime_error(
            quality_label="complex",
            model="model-b",
            dpi=1200,
            aggressive=False,
            error_text="400 client error empty response",
        )
    profile = engine.get_adaptive_profile("complex")
    assert profile["force_aggressive"] is True
    assert "compact_payload" in profile["notes"]
