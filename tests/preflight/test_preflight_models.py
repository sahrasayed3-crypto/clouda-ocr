"""Unit tests for the preflight domain model (models.py).

Unique basename ``test_preflight_models`` per WAVE1_BRIEF (no collection
clashes). Pure domain-model tests: no torch / transformers imports.
"""

from __future__ import annotations

from clouda_training.preflight import (
    PreflightCheck,
    PreflightContext,
    PreflightFinalStatus,
    PreflightReport,
    PreflightSection,
    PreflightSeverity,
    PreflightStatus,
    TrainingBlocker,
    TrainingPlanSummary,
    TrainingWarning,
)


def check(
    name: str,
    status: PreflightStatus,
    *,
    blocker: bool = False,
    detail: str = "d",
) -> PreflightCheck:
    return PreflightCheck(name=name, status=status, detail=detail, blocker=blocker)


def report(*checks: PreflightCheck) -> PreflightReport:
    return PreflightReport(sections=(PreflightSection(name="s", checks=tuple(checks)),))


class TestEnums:
    def test_status_values(self) -> None:
        assert {s.value for s in PreflightStatus} == {
            "PASS",
            "WARN",
            "FAIL",
            "SKIP",
            "UNAVAILABLE",
        }

    def test_severity_values(self) -> None:
        assert {s.value for s in PreflightSeverity} == {"INFO", "WARNING", "ERROR"}

    def test_final_status_values(self) -> None:
        assert {s.value for s in PreflightFinalStatus} == {
            "READY",
            "READY_WITH_WARNINGS",
            "NOT_READY",
        }


class TestCheckAndSection:
    def test_check_to_dict_uses_status_value(self) -> None:
        c = check("x", PreflightStatus.FAIL, blocker=True, detail="boom")
        d = c.to_dict()
        assert d == {
            "name": "x",
            "status": "FAIL",
            "detail": "boom",
            "blocker": True,
        }

    def test_section_worst_status_ordering(self) -> None:
        assert (
            PreflightSection(
                name="s",
                checks=(
                    check("a", PreflightStatus.PASS),
                    check("b", PreflightStatus.WARN),
                ),
            ).worst_status()
            is PreflightStatus.WARN
        )
        assert (
            PreflightSection(
                name="s",
                checks=(
                    check("a", PreflightStatus.UNAVAILABLE),
                    check("b", PreflightStatus.SKIP),
                    check("c", PreflightStatus.PASS),
                ),
            ).worst_status()
            is PreflightStatus.UNAVAILABLE
        )

    def test_section_worst_status_empty(self) -> None:
        assert PreflightSection(name="empty").worst_status() is None

    def test_section_to_dict(self) -> None:
        d = PreflightSection(
            name="cfg", checks=(check("a", PreflightStatus.PASS),)
        ).to_dict()
        assert d["name"] == "cfg"
        assert d["worst_status"] == "PASS"
        assert d["checks"][0]["status"] == "PASS"


class TestFinalStatusAggregation:
    def test_empty_report_is_ready(self) -> None:
        assert PreflightReport().final_status() is PreflightFinalStatus.READY

    def test_all_pass_is_ready(self) -> None:
        assert (
            report(check("a", PreflightStatus.PASS)).final_status()
            is PreflightFinalStatus.READY
        )

    def test_warn_only_is_ready_with_warnings(self) -> None:
        assert (
            report(check("a", PreflightStatus.WARN)).final_status()
            is PreflightFinalStatus.READY_WITH_WARNINGS
        )

    def test_fail_blocker_is_not_ready(self) -> None:
        r = report(check("a", PreflightStatus.FAIL, blocker=True))
        assert r.final_status() is PreflightFinalStatus.NOT_READY

    def test_fail_blocker_dominates_warn(self) -> None:
        r = report(
            check("w", PreflightStatus.WARN),
            check("f", PreflightStatus.FAIL, blocker=True),
        )
        assert r.final_status() is PreflightFinalStatus.NOT_READY

    def test_fail_blocker_dominates_pass(self) -> None:
        r = report(
            check("ok", PreflightStatus.PASS),
            check("f", PreflightStatus.FAIL, blocker=True),
        )
        assert r.final_status() is PreflightFinalStatus.NOT_READY

    def test_non_blocker_fail_is_ready_with_warnings_not_not_ready(self) -> None:
        r = report(check("info", PreflightStatus.FAIL, blocker=False))
        assert r.final_status() is PreflightFinalStatus.READY_WITH_WARNINGS
        assert not r.blockers

    def test_unavailable_only_is_ready_with_warnings(self) -> None:
        # Decision: UNAVAILABLE alone never blocks -> READY_WITH_WARNINGS.
        r = report(check("quality-gate", PreflightStatus.UNAVAILABLE))
        assert r.final_status() is PreflightFinalStatus.READY_WITH_WARNINGS
        assert not r.blockers

    def test_unavailable_plus_pass_is_ready_with_warnings(self) -> None:
        r = report(
            check("ok", PreflightStatus.PASS),
            check("u", PreflightStatus.UNAVAILABLE),
        )
        assert r.final_status() is PreflightFinalStatus.READY_WITH_WARNINGS

    def test_skip_only_is_ready(self) -> None:
        # SKIP is neutral: neither blocks nor warns.
        r = report(check("skipped", PreflightStatus.SKIP))
        assert r.final_status() is PreflightFinalStatus.READY

    def test_skip_with_warn_is_ready_with_warnings(self) -> None:
        r = report(
            check("skipped", PreflightStatus.SKIP),
            check("w", PreflightStatus.WARN),
        )
        assert r.final_status() is PreflightFinalStatus.READY_WITH_WARNINGS

    def test_non_blocker_fail_plus_warn_not_not_ready(self) -> None:
        r = report(
            check("info", PreflightStatus.FAIL, blocker=False),
            check("w", PreflightStatus.WARN),
        )
        assert r.final_status() is PreflightFinalStatus.READY_WITH_WARNINGS

    def test_blocker_flag_only_matters_on_fail(self) -> None:
        r = report(
            check("w_blocked", PreflightStatus.WARN, blocker=True),
            check("u_blocked", PreflightStatus.UNAVAILABLE, blocker=True),
        )
        assert r.final_status() is PreflightFinalStatus.READY_WITH_WARNINGS
        assert not r.blockers

    def test_aggregation_across_sections(self) -> None:
        r = PreflightReport(
            sections=(
                PreflightSection(name="a", checks=(check("a1", PreflightStatus.PASS),)),
                PreflightSection(
                    name="b",
                    checks=(
                        check("b1", PreflightStatus.FAIL, blocker=True),
                        check("b2", PreflightStatus.WARN),
                    ),
                ),
            )
        )
        assert r.final_status() is PreflightFinalStatus.NOT_READY


class TestBlockersAndWarnings:
    def test_blockers_explicit_list(self) -> None:
        r = report(
            check("f1", PreflightStatus.FAIL, blocker=True, detail="bad"),
            check("f2", PreflightStatus.FAIL, blocker=False),
        )
        assert r.blockers == (TrainingBlocker(check_name="f1", reason="bad"),)

    def test_blocker_to_dict_severity_value(self) -> None:
        b = TrainingBlocker(
            check_name="f", reason="r", severity=PreflightSeverity.ERROR
        ).to_dict()
        assert b == {"check_name": "f", "reason": "r", "severity": "ERROR"}

    def test_warnings_categorize_non_blocking_concerns(self) -> None:
        r = report(
            check("w", PreflightStatus.WARN, detail="watch out"),
            check("nf", PreflightStatus.FAIL, blocker=False, detail="minor"),
            check("u", PreflightStatus.UNAVAILABLE, detail="missing gate"),
        )
        by_name = {w.check_name: w for w in r.warnings}
        assert by_name["w"].message == "watch out"
        assert by_name["w"].severity is PreflightSeverity.WARNING
        assert "non-blocking failure" in by_name["nf"].message
        assert by_name["u"].message == "missing gate"
        assert by_name["u"].severity is PreflightSeverity.INFO

    def test_unavailable_default_message(self) -> None:
        r = report(check("u", PreflightStatus.UNAVAILABLE, detail=""))
        assert r.warnings[0].message == "capability unavailable"

    def test_no_warnings_when_all_pass(self) -> None:
        assert report(check("a", PreflightStatus.PASS)).warnings == ()


class TestSummaryDataclasses:
    def test_training_plan_summary_to_dict(self) -> None:
        s = TrainingPlanSummary(
            effective_batch_size=8,
            micro_batches_per_epoch=4,
            optimizer_steps_per_epoch=2,
            planned_optimizer_steps=2,
            estimated_checkpoint_count=0,
            world_size=1,
            notes=("max_steps override",),
        )
        d = s.to_dict()
        assert d["effective_batch_size"] == 8
        assert d["notes"] == ["max_steps override"]

    def test_training_plan_summary_defaults_are_none(self) -> None:
        assert TrainingPlanSummary().planned_optimizer_steps is None

    def test_training_warning_to_dict(self) -> None:
        d = TrainingWarning(check_name="c", message="m").to_dict()
        assert d == {
            "check_name": "c",
            "message": "m",
            "severity": "WARNING",
        }

    def test_preflight_context_defaults(self) -> None:
        c = PreflightContext()
        assert c.model_id is None
        assert c.adapter_type is None
        assert c.generated_at  # timestamp populated


class TestReportToDict:
    def test_full_dict_roundtrip_fields(self) -> None:
        r = PreflightReport(
            sections=(
                PreflightSection(
                    name="sys",
                    checks=(
                        check("dev", PreflightStatus.PASS),
                        check("f", PreflightStatus.FAIL, blocker=True, detail="x"),
                        check("u", PreflightStatus.UNAVAILABLE, detail="missing"),
                    ),
                ),
            ),
            context=PreflightContext(model_id="m", device="cpu"),
            training_plan=TrainingPlanSummary(effective_batch_size=2),
            adapter_identity="hunyuanocr15_sft@abc",
            dataset_identity="ds@sha256:deadbeef",
            checkpoint_identity=None,
        )
        d = r.to_dict()
        assert d["final_status"] == "NOT_READY"
        assert d["ready"] is False
        assert d["context"]["model_id"] == "m"
        assert [b["check_name"] for b in d["blockers"]] == ["f"]
        assert {w["check_name"] for w in d["warnings"]} == {"u"}
        assert d["training_plan"]["effective_batch_size"] == 2
        assert d["adapter_identity"] == "hunyuanocr15_sft@abc"
        assert d["dataset_identity"] == "ds@sha256:deadbeef"
        assert d["checkpoint_identity"] is None
        assert d["sections"][0]["name"] == "sys"

    def test_empty_report_to_dict(self) -> None:
        d = PreflightReport().to_dict()
        assert d["final_status"] == "READY"
        assert d["ready"] is True
        assert d["blockers"] == []
        assert d["warnings"] == []
        assert d["context"] is None
        assert d["training_plan"] is None

    def test_to_dict_is_json_serializable(self) -> None:
        import json

        r = report(
            check("a", PreflightStatus.WARN),
            check("b", PreflightStatus.SKIP),
        )
        payload = json.dumps(r.to_dict())
        assert json.loads(payload)["final_status"] == "READY_WITH_WARNINGS"


class TestFailClosedGuarantees:
    def test_every_fail_blocker_combo_is_not_ready(self) -> None:
        for other in PreflightStatus:
            r = report(
                check("f", PreflightStatus.FAIL, blocker=True),
                check("o", other),
            )
            assert r.final_status() is PreflightFinalStatus.NOT_READY, other

    def test_unavailable_never_produces_blocker(self) -> None:
        r = report(
            check("u1", PreflightStatus.UNAVAILABLE, blocker=True),
            check("u2", PreflightStatus.UNAVAILABLE),
        )
        assert not r.blockers
        assert r.final_status() is PreflightFinalStatus.READY_WITH_WARNINGS

    def test_all_status_combinations_classified(self) -> None:
        expected: dict[tuple[PreflightStatus, ...], PreflightFinalStatus] = {
            (PreflightStatus.PASS,): PreflightFinalStatus.READY,
            (PreflightStatus.SKIP,): PreflightFinalStatus.READY,
            (PreflightStatus.UNAVAILABLE,): PreflightFinalStatus.READY_WITH_WARNINGS,
            (PreflightStatus.WARN,): PreflightFinalStatus.READY_WITH_WARNINGS,
            (
                PreflightStatus.FAIL,
            ): PreflightFinalStatus.READY_WITH_WARNINGS,  # non-blocker
        }
        for statuses, want in expected.items():
            r = report(*(check(f"c{i}", s) for i, s in enumerate(statuses)))
            assert r.final_status() is want, statuses
