from pdfword.release_self_test import run_release_self_test


def test_release_self_test_exercises_categorical_pipeline() -> None:
    result = run_release_self_test()

    assert result.ok
    assert {"digital_text", "blank_page", "pending_ocr_model"} <= set(result.states)
    assert result.page_boundaries_preserved
    assert result.user_facing_percentages is False
