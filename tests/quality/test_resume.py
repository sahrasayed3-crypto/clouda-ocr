"""Resume / run-state identity tests."""

from __future__ import annotations

import pytest

from clouda_data.quality.run_state import (
    QualityRunState,
    StaleResumeError,
    checkpoint,
    start_or_resume,
)


class TestResume:
    def test_stale_resume_refused(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        initial = QualityRunState("a" * 64, 1, "config-a", {"gate": "1"})
        start_or_resume(tmp_path, initial)
        changed = QualityRunState("b" * 64, 1, "config-a", {"gate": "1"})
        with pytest.raises(StaleResumeError, match="manifest_sha256"):
            start_or_resume(tmp_path, changed)

    def test_resume_state_matches_identity(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        expected = QualityRunState("a" * 64, 2, "config-a", {"gate": "1"})
        state, resumed = start_or_resume(tmp_path, expected)
        assert state == expected and resumed is False
        checkpoint(tmp_path, "near", {"row": 2}, 2)
        resumed_state, resumed = start_or_resume(tmp_path, expected)
        assert resumed is True
        assert resumed_state.stage == "near"
        assert resumed_state.processed_count == 2
