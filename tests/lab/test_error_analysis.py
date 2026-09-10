"""Tests: OCR Error Analysis Engine (Phases 2-3)."""

from __future__ import annotations

import pytest

from clouda_lab.error_analysis import analyze_sample, normalized_cer
from clouda_lab.error_taxonomy import classify_char_error, classify_word_error
from clouda_lab.batch_analysis import analyze_batch, export_batch_csv
from clouda_lab.models import BatchReport, OCRSample


def _sample(sample_id: str, gt: str, pred: str, **meta) -> OCRSample:
    return OCRSample(sample_id=sample_id, ground_truth=gt, prediction=pred, **meta)


class TestExactMatch:
    def test_identical_text(self):
        analysis = analyze_sample(_sample("s1", "مرحبا بالعالم", "مرحبا بالعالم"))
        assert analysis.cer == 0.0
        assert analysis.wer == 0.0
        assert analysis.normalized_cer == 0.0
        assert analysis.exact_match is True
        assert all(r.operation == "match" for r in analysis.char_records)
        assert analysis.error_type_counts == {}
        assert analysis.substitutions == ()

    def test_counts(self):
        analysis = analyze_sample(_sample("s1", "مرحبا", "مرحبا"))
        assert analysis.character_count == 5
        assert analysis.word_count == 1


class TestBasicEdits:
    def test_insertion(self):
        analysis = analyze_sample(_sample("s1", "كتاب", "كتابا"))
        assert analysis.cer == pytest.approx(1 / 4)
        insertions = [r for r in analysis.char_records if r.operation == "insertion"]
        assert len(insertions) == 1
        assert insertions[0].predicted == "ا"
        assert insertions[0].gt is None

    def test_deletion(self):
        analysis = analyze_sample(_sample("s1", "كتاب", "كتا"))
        assert analysis.cer == pytest.approx(1 / 4)
        deletions = [r for r in analysis.char_records if r.operation == "deletion"]
        assert len(deletions) == 1
        assert deletions[0].gt == "ب"
        assert deletions[0].predicted is None

    def test_substitution(self):
        # كتاب vs كتابن: the aligner resolves the extra trailing char as an
        # insertion (deterministic tie-break), not a substitution.
        analysis = analyze_sample(_sample("s1", "كتاب", "كتابن"))
        insertions = [r for r in analysis.char_records if r.operation == "insertion"]
        assert len(insertions) == 1 and insertions[0].predicted == "ن"
        # A pure substitution mid-word: ب -> ن
        analysis = analyze_sample(_sample("s1", "كتاب", "كتان"))
        subs = [r for r in analysis.char_records if r.operation == "substitution"]
        assert len(subs) == 1
        assert subs[0].gt == "ب" and subs[0].predicted == "ن"
        assert subs[0].error_type == "character_substitution"
        assert subs[0].position == 3
        assert subs[0].context
        assert analysis.cer == pytest.approx(1 / 4)

    def test_deterministic_alignment(self):
        gt, pred = "اللغة العربية جميلة", "اللغه العربيه جميله"
        first = analyze_sample(_sample("s1", gt, pred))
        for _ in range(3):
            again = analyze_sample(_sample("s1", gt, pred))
            assert [r.to_dict() for r in again.char_records] == [
                r.to_dict() for r in first.char_records
            ]


class TestArabicClassification:
    def test_whitespace_error(self):
        analysis = analyze_sample(_sample("s1", "مرحبا بالعالم", "مرحبا  بالعالم"))
        types = analysis.error_type_counts
        assert types.get("whitespace", 0) >= 1

    def test_digit_system_confusion(self):
        analysis = analyze_sample(_sample("s1", "سنة 2024", "سنة ٢٠٢٤"))
        assert analysis.error_type_counts.get("digit_system_confusion", 0) >= 1
        assert analysis.normalized_cer == 0.0  # digits fold under N-CER policy

    def test_arabic_digit_substitution(self):
        assert classify_char_error("٣", "٤") == "arabic_digit"
        assert classify_char_error("٣", "3") == "digit_system_confusion"

    def test_latin_digit(self):
        assert classify_char_error("5", None) == "latin_digit"
        assert classify_char_error(None, "7") == "latin_digit"

    def test_diacritic(self):
        # Removing a fatha: one char-level deletion + one word-level
        # diacritic classification in the folded summary counts.
        analysis = analyze_sample(_sample("s1", "مَكتب", "مكتب"))
        assert analysis.error_type_counts.get("diacritic", 0) == 2
        char_deletions = [
            r
            for r in analysis.char_records
            if r.operation == "deletion" and r.error_type == "diacritic"
        ]
        assert len(char_deletions) == 1
        assert analysis.normalized_cer == 0.0
        # Substituting one diacritic for another is also classified diacritic
        assert classify_char_error("َ", "ُ") == "diacritic"

    def test_hamza_alef_variants(self):
        assert classify_char_error("أ", "ا") == "hamza_alef_variant"
        assert classify_char_error("إ", "ا") == "hamza_alef_variant"
        assert classify_char_error("آ", "ٱ") == "hamza_alef_variant"
        analysis = analyze_sample(_sample("s1", "أحمد", "احمد"))
        assert analysis.error_type_counts.get("hamza_alef_variant", 0) >= 1

    def test_ta_marbuta_ha(self):
        assert classify_char_error("ة", "ه") == "ta_marbuta_ha"

    def test_ya_alef_maqsura(self):
        assert classify_char_error("ى", "ي") == "ya_alef_maqsura"

    def test_script_confusion(self):
        assert classify_char_error("ن", "n") == "arabic_latin_script"
        assert classify_char_error("a", "ب") == "arabic_latin_script"

    def test_punctuation(self):
        assert classify_char_error("،", ".") == "punctuation"
        assert classify_char_error(None, "؟") == "punctuation"

    def test_word_categories(self):
        assert classify_word_error("كتاب", None) == "missing_word"
        assert classify_word_error(None, "كتاب") == "extra_word"
        assert classify_word_error("مَكتب", "مكتب") == "diacritic"
        assert classify_word_error("123", "٤٥٦") == "digit_word"
        assert classify_word_error("review", "مراجعة") == "mixed_script_word"
        assert classify_word_error("كتاب", "قلم") == "word_substitution"

    def test_missing_word_summary(self):
        analysis = analyze_sample(
            _sample("s1", "مرحبا بالعالم الجميل", "مرحبا بالعالم")
        )
        assert analysis.error_type_counts.get("missing_word", 0) == 1

    def test_unknown_fallback(self):
        assert classify_char_error("§", "€") in {"character_substitution", "unknown"}
        assert classify_word_error("§", "€") == "word_substitution"


class TestNormalizedCER:
    def test_alef_folding(self):
        assert normalized_cer("أحمد", "احمد") == 0.0

    def test_diacritic_removal(self):
        assert normalized_cer("مَكْتَب", "مكتب") == 0.0

    def test_digit_folding(self):
        assert normalized_cer("٣٥", "35") == 0.0

    def test_real_difference_survives(self):
        assert normalized_cer("كتاب", "قلم") > 0.0


class TestSummaryOutput:
    def test_summary_counts_and_rates(self):
        analysis = analyze_sample(_sample("s1", "أحمد مش في البيت", "احمد مش في البيت"))
        assert analysis.error_type_counts.get("hamza_alef_variant") == 1
        total = analysis.character_count
        for label, count in analysis.error_type_counts.items():
            assert analysis.error_type_rates[label] == pytest.approx(
                count / total, abs=1e-6
            )

    def test_substitution_pairs_counted(self):
        analysis = analyze_sample(_sample("s1", "أأأ", "ااا"))
        assert analysis.substitutions == (("أ", "ا", 3),)

    def test_to_dict_is_json_safe(self):
        import json

        analysis = analyze_sample(_sample("s1", "مرحبا", "مراحبا"))
        payload = json.dumps(analysis.to_dict(), ensure_ascii=False)
        assert "page_id" in payload


class TestBatch:
    def _batch(self):
        return [
            _sample(
                "p1",
                "مرحبا بالعالم",
                "مرحبا بالعالم",
                model_id="m1",
                metadata={"profile": "clean", "split": "test", "document_type": "book"},
            ),
            _sample(
                "p2",
                "مرحبا بالعالم الجميل",
                "مرحبا بالعالم",
                model_id="m1",
                metadata={
                    "profile": "bad_scan_heavy",
                    "split": "test",
                    "distortion": "gaussian_blur",
                    "document_type": "book",
                },
            ),
            _sample(
                "p3",
                "الرقم ٢٠٢٤ كبير",
                "الرقم 2024 كبير",
                model_id="m2",
                metadata={
                    "profile": "clean",
                    "split": "train",
                    "document_type": "form",
                },
            ),
        ]

    def test_aggregation_by_model(self):
        report = analyze_batch(self._batch())
        assert report.total_samples == 3
        assert report.groups["model_id"]["m1"].count == 2
        assert report.groups["model_id"]["m2"].count == 1
        assert report.overall.count == 3
        m1 = report.groups["model_id"]["m1"]
        # p1 is a perfect match (CER 0), p2 loses one word (CER 0.35)
        assert m1.cer_mean == pytest.approx(0.35 / 2)

    def test_aggregation_by_profile_split_source(self):
        report = analyze_batch(self._batch())
        assert report.groups["profile"]["clean"].count == 2
        assert report.groups["split"]["train"].count == 1
        assert report.groups["distortion"]["gaussian_blur"].count == 1
        assert report.groups["document_type"]["form"].count == 1

    def test_worst_and_best_pages(self):
        report = analyze_batch(self._batch())
        assert report.worst_pages[0].key == "p2"  # deletion -> highest CER
        assert report.best_pages[0].key in {"p1", "p3"}
        assert report.worst_pages[0].cer_mean >= report.best_pages[0].cer_mean

    def test_percentiles(self):
        report = analyze_batch(self._batch())
        overall = report.overall
        assert overall.cer_p50 <= overall.cer_p90 <= overall.cer_p95 <= overall.cer_max

    def test_error_type_distribution(self):
        report = analyze_batch(self._batch())
        # p2 loses one word; p3 substitutes all 4 Arabic-Indic digits for Latin
        assert report.error_type_counts.get("digit_system_confusion") == 4
        assert report.error_type_counts.get("missing_word") == 1

    def test_export_json_and_csv(self, tmp_path):
        import json

        report = analyze_batch(self._batch())
        out_json = tmp_path / "batch.json"
        out_csv = tmp_path / "batch.csv"
        out_json.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        export_batch_csv(report, out_csv)
        assert json.loads(out_json.read_text(encoding="utf-8"))["total_samples"] == 3
        assert "cer_mean" in out_csv.read_text(encoding="utf-8")

    def test_empty_batch(self):
        report = analyze_batch([])
        assert isinstance(report, BatchReport)
        assert report.total_samples == 0
        assert report.overall.count == 0
