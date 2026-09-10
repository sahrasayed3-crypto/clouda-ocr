"""Tests: Failure Analysis + Buckets + Hard Examples + Active Learning
(Phases 6-10)."""

from __future__ import annotations

import pytest

from clouda_lab.active_learning import recommend_next_batch
from clouda_lab.failure_analysis import (
    DEFAULT_THRESHOLDS,
    SampleMetrics,
    compare_failure,
    summarize_comparison,
)
from clouda_lab.failure_buckets import assign_buckets, bucket_sample, primary_bucket
from clouda_lab.hard_examples import rank_hard_examples, select_hard_examples
from clouda_lab.models import IMPROVED, NEWLY_FAILED, RECOVERED, REGRESSED, UNCHANGED


def _m(sample_id: str, cer: float, wer: float = 0.0, ncer: float = 0.0, **kw) -> SampleMetrics:
    return SampleMetrics(sample_id=sample_id, cer=cer, wer=wer, ncer=ncer, **kw)


class TestFailureClassification:
    def test_improved(self):
        report = compare_failure(
            [_m("s1", 0.40)], [_m("s1", 0.20)],
            baseline_id="base", candidate_id="cand",
        )
        assert report.comparisons[0].classification == IMPROVED
        assert report.comparisons[0].cer_delta == pytest.approx(-0.20)

    def test_regressed(self):
        report = compare_failure(
            [_m("s1", 0.20)], [_m("s1", 0.40)],
            baseline_id="base", candidate_id="cand",
        )
        assert report.comparisons[0].classification == REGRESSED

    def test_unchanged(self):
        report = compare_failure(
            [_m("s1", 0.30)], [_m("s1", 0.32)],
            baseline_id="base", candidate_id="cand",
        )
        assert report.comparisons[0].classification == UNCHANGED

    def test_newly_failed(self):
        report = compare_failure(
            [_m("s1", 0.30)], [_m("s1", 0.55)],
            baseline_id="base", candidate_id="cand",
        )
        assert report.comparisons[0].classification == NEWLY_FAILED

    def test_recovered(self):
        report = compare_failure(
            [_m("s1", 0.60)], [_m("s1", 0.30)],
            baseline_id="base", candidate_id="cand",
        )
        assert report.comparisons[0].classification == RECOVERED

    def test_configurable_thresholds(self):
        report = compare_failure(
            [_m("s1", 0.30)], [_m("s1", 0.34)],
            baseline_id="base", candidate_id="cand",
            thresholds={"improve": 0.01, "regress": 0.01},
        )
        assert report.comparisons[0].classification == REGRESSED  # 0.04 >= 0.01

    def test_unknown_threshold_rejected(self):
        with pytest.raises(ValueError, match="Unknown thresholds"):
            compare_failure(
                [_m("s1", 0.1)], [_m("s1", 0.1)],
                baseline_id="a", candidate_id="b",
                thresholds={"bogus": 1.0},
            )

    def test_counts_and_rollup(self):
        report = compare_failure(
            [_m("a", 0.5), _m("b", 0.2), _m("c", 0.3), _m("d", 0.6)],
            [_m("a", 0.2), _m("b", 0.4), _m("c", 0.3), _m("d", 0.7)],
            baseline_id="base", candidate_id="cand",
        )
        # a: 0.5 -> 0.2 crosses the failure line downward = RECOVERED
        # b: 0.2 -> 0.4 = REGRESSED (0.2 delta >= threshold)
        # c: unchanged; d: 0.6 -> 0.7 stays failed, regressed
        assert report.counts[RECOVERED] == 1
        assert report.counts[REGRESSED] == 2
        assert report.counts[UNCHANGED] == 1
        assert report.counts[IMPROVED] == 0  # a was a recovery, not plain improved
        summary = summarize_comparison(report)
        assert summary["common_samples"] == 4
        assert "b" in summary["worst_regressions"]

    def test_persistent_failures(self):
        report = compare_failure(
            [_m("keep", 0.9), _m("fix", 0.9)],
            [_m("keep", 0.8), _m("fix", 0.2)],
            baseline_id="base", candidate_id="cand",
        )
        assert report.persistent_failures == ("keep",)

    def test_error_type_deltas(self):
        report = compare_failure(
            [_m("s1", 0.4, error_types={"whitespace": 2, "diacritic": 1})],
            [_m("s1", 0.2, error_types={"whitespace": 0, "diacritic": 3})],
            baseline_id="base", candidate_id="cand",
        )
        deltas = report.comparisons[0].error_type_deltas
        assert deltas == {"whitespace": -2, "diacritic": 2}
        assert report.error_type_delta_totals == {"diacritic": 2, "whitespace": -2}

    def test_duplicate_sample_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            compare_failure([_m("s1", 0.1), _m("s1", 0.2)], [], baseline_id="a", candidate_id="b")

    def test_report_to_dict(self):
        report = compare_failure([_m("s1", 0.4)], [_m("s1", 0.1)],
                                 baseline_id="base", candidate_id="cand")
        payload = report.to_dict()
        assert payload["baseline_id"] == "base"
        assert payload["thresholds"] == dict(DEFAULT_THRESHOLDS)


class TestFailureBuckets:
    def test_high_cer(self):
        buckets = bucket_sample(cer=0.7, wer=0.8)
        labels = {b.bucket for b in buckets}
        assert "high_cer" in labels and "high_wer" in labels
        assert primary_bucket(buckets) == "high_cer"  # fixed rule order

    def test_error_mix_buckets(self):
        buckets = bucket_sample(
            cer=0.2,
            error_type_counts={"whitespace": 5, "character_substitution": 1},
        )
        labels = {b.bucket for b in buckets}
        assert "whitespace_heavy" in labels
        assert primary_bucket(buckets) == "whitespace_heavy"

    def test_digit_and_punctuation_buckets(self):
        digit = bucket_sample(error_type_counts={"arabic_digit": 4, "whitespace": 1})
        assert "digit_heavy" in {b.bucket for b in digit}
        punct = bucket_sample(error_type_counts={"punctuation": 4, "whitespace": 1})
        assert "punctuation_heavy" in {b.bucket for b in punct}

    def test_deletion_insertion_buckets(self):
        deletions = bucket_sample(error_type_counts={"missing_word": 3, "whitespace": 1})
        assert "deletion_heavy" in {b.bucket for b in deletions}
        insertions = bucket_sample(error_type_counts={"extra_word": 3, "whitespace": 1})
        assert "insertion_heavy" in {b.bucket for b in insertions}

    def test_metadata_buckets_only_from_present_metadata(self):
        blur = bucket_sample(metadata={"distortion": "gaussian_blur"})
        assert "distorted_blur" in {b.bucket for b in blur}
        skew = bucket_sample(metadata={"distortion": ["skew"]})
        assert "distorted_skew" in {b.bucket for b in skew}
        table = bucket_sample(metadata={"document_type": "invoice"})
        assert "table_form" in {b.bucket for b in table}
        # no inference without metadata:
        assert bucket_sample(cer=0.1) == [bucket_sample(cer=0.1)[0]]
        assert bucket_sample(cer=0.1)[0].bucket == "unknown"

    def test_unknown_fallback(self):
        buckets = bucket_sample(cer=0.1, wer=0.1)
        assert buckets[0].bucket == "unknown"

    def test_assign_buckets_many(self):
        result = assign_buckets(
            [
                {"sample_id": "a", "cer": 0.8, "wer": 0.9},
                {"sample_id": "b", "cer": 0.1},
            ]
        )
        assert {b.bucket for b in result["a"]} >= {"high_cer"}
        assert result["b"][0].bucket == "unknown"

    def test_custom_thresholds(self):
        buckets = bucket_sample(cer=0.25, thresholds={"high_cer": 0.2})
        assert buckets[0].bucket == "high_cer"


class TestHardExamples:
    def _rows(self):
        return [
            {"sample_id": "easy", "cer": 0.02, "wer": 0.05, "ncer": 0.0},
            {"sample_id": "hard1", "cer": 0.55, "wer": 0.65, "ncer": 0.50,
             "error_type_counts": {"whitespace": 2, "diacritic": 1, "arabic_digit": 1}},
            {"sample_id": "hard2", "cer": 0.60, "wer": 0.70, "ncer": 0.55,
             "error_type_counts": {"character_substitution": 3}},
            {"sample_id": "hard3", "cer": 0.52, "wer": 0.60, "ncer": 0.48,
             "error_type_counts": {"missing_word": 2}},
        ]

    def test_ranking_order(self):
        ranked = rank_hard_examples(self._rows())
        # hard1 wins on combined signals: more error diversity than hard2
        # (3 categories vs 1) despite slightly lower raw CER.
        assert ranked[0].sample_id == "hard1"
        assert ranked[-1].sample_id == "easy"
        assert ranked[0].score > ranked[1].score > ranked[2].score > ranked[3].score
        assert [item.rank for item in ranked] == [1, 2, 3, 4]

    def test_transparent_scoring(self):
        ranked = rank_hard_examples(self._rows())
        top = ranked[0]
        # CER/WER signals are the raw rates clipped to [0, 1].
        assert top.signals["wer"] == pytest.approx(0.65)
        assert top.weights  # weights recorded
        expected = sum(top.weights[k] * top.signals.get(k, 0.0) for k in top.weights)
        assert top.score == pytest.approx(expected, abs=1e-5)

    def test_custom_weights_change_ranking(self):
        custom = {"cer": 1.0, "wer": 0.0, "ncer": 0.0}
        ranked = rank_hard_examples(
            [
                {"sample_id": "a", "cer": 0.9, "wer": 0.0},
                {"sample_id": "b", "cer": 0.5, "wer": 0.9},
            ],
            weights=custom,
        )
        assert ranked[0].sample_id == "a"

    def test_unknown_signal_rejected(self):
        with pytest.raises(ValueError, match="Unknown hard-example signal"):
            rank_hard_examples(self._rows(), weights={"magic": 1.0})

    def test_deterministic(self):
        first = rank_hard_examples(self._rows())
        for _ in range(3):
            assert [i.to_dict() for i in rank_hard_examples(self._rows())] == [
                i.to_dict() for i in first
            ]

    def test_select_top_n_and_percentile(self):
        ranked = self._rows()
        top1 = select_hard_examples(ranked, top_n=1)
        assert len(top1) == 1 and top1[0].sample_id == "hard1"
        half = select_hard_examples(ranked, percentile=50)
        assert len(half) == 2
        floor = select_hard_examples(ranked, min_score=0.3)
        assert all(item.score >= 0.3 for item in floor)

    def test_cross_model_and_persistent_signals(self):
        rows = [
            {"sample_id": "chronic", "cer": 0.6, "wer": 0.7, "ncer": 0.5,
             "persistent_failure_count": 4, "regression_magnitude": 0.3,
             "model_id": "m1"},
            {"sample_id": "fresh", "cer": 0.55, "wer": 0.65, "ncer": 0.5,
             "model_id": "m2"},
        ]
        ranked = rank_hard_examples(rows)
        chronic = next(i for i in ranked if i.sample_id == "chronic")
        fresh = next(i for i in ranked if i.sample_id == "fresh")
        assert chronic.signals["persistent_failure_count"] == 1.0
        assert fresh.signals["persistent_failure_count"] == 0.0
        assert chronic.score > fresh.score


class TestActiveLearning:
    def _rows(self):
        return [
            {"sample_id": f"blur-{i}", "cer": 0.6 - i * 0.01, "wer": 0.7,
             "ncer": 0.5, "profile": "bad_scan_heavy", "distortion": "gaussian_blur",
             "error_type_counts": {"character_substitution": 2}, "model_id": "m1"}
            for i in range(6)
        ] + [
            {"sample_id": f"skew-{i}", "cer": 0.5, "wer": 0.6, "ncer": 0.4,
             "profile": "phone_photo_medium", "distortion": "skew",
             "error_type_counts": {"whitespace": 2}, "model_id": "m1"}
            for i in range(6)
        ]

    def test_hardest_only(self):
        rec = recommend_next_batch(self._rows(), strategy="hardest_only", batch_size=4, seed=1)
        ids = [s.sample_id for s in rec.selections]
        assert len(ids) == 4
        assert ids[0] == "blur-0"  # highest CER
        assert rec.strategy == "hardest_only"

    def test_balanced_hard_respects_quotas(self):
        rec = recommend_next_batch(self._rows(), strategy="balanced_hard", batch_size=4, seed=1)
        assert rec.balance["profile"]["bad_scan_heavy"] == 2
        assert rec.balance["profile"]["phone_photo_medium"] == 2

    def test_avoids_duplicates_with_history(self):
        rows = self._rows()
        rec = recommend_next_batch(rows, strategy="hardest_only", batch_size=4,
                                   seed=1, history=("blur-0", "blur-1", "skew-0"))
        ids = [s.sample_id for s in rec.selections]
        assert "blur-0" not in ids and "blur-1" not in ids and "skew-0" not in ids
        assert "blur-0" in rec.excluded
        assert len(set(ids)) == len(ids)

    def test_diversity_first_covers_categories(self):
        rec = recommend_next_batch(self._rows(), strategy="diversity_first", batch_size=6, seed=1)
        ids = [s.sample_id for s in rec.selections]
        # first picks must come from different dominant categories
        first_two = ids[:2]
        assert {sid.split("-")[0] for sid in first_two} == {"blur", "skew"}
        assert all(rec.rationale.get(sid) for sid in ids)

    def test_rationale_recorded(self):
        rec = recommend_next_batch(self._rows(), strategy="balanced_hard", batch_size=4, seed=1)
        for selection in rec.selections:
            assert selection.sample_id in rec.rationale
            assert "profile" in rec.rationale[selection.sample_id]

    def test_deterministic_random(self):
        first = recommend_next_batch(self._rows(), strategy="deterministic_random",
                                     batch_size=5, seed=9)
        second = recommend_next_batch(self._rows(), strategy="deterministic_random",
                                      batch_size=5, seed=9)
        other = recommend_next_batch(self._rows(), strategy="deterministic_random",
                                     batch_size=5, seed=10)
        assert [s.sample_id for s in first.selections] == [s.sample_id for s in second.selections]
        assert [s.sample_id for s in first.selections] != [s.sample_id for s in other.selections]

    def test_mixed_curriculum(self):
        rec = recommend_next_batch(self._rows(), strategy="mixed_curriculum", batch_size=6, seed=1)
        assert len(rec.selections) == 6

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            recommend_next_batch(self._rows(), strategy="magic")

    def test_duplicate_input_rejected(self):
        rows = self._rows()
        with pytest.raises(ValueError, match="Duplicate"):
            recommend_next_batch(rows + rows, strategy="hardest_only")
