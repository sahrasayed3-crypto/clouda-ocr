from __future__ import annotations

from pdfword.ocr_self_review import run_cuda_smoke


def test_cuda_smoke_skips_without_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CLOUDA_CUDA_SMOKE", raising=False)

    result = run_cuda_smoke()

    assert result.state == "skipped"
    assert result.reason == "not_enabled"
