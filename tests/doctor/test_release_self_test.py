from clouda_data.doctor.report import collect_report


def test_deep_doctor_reports_pdfword_release_self_test() -> None:
    report = collect_report(deep=True)

    assert any(
        check.id == "deep.pdfword-release-self-test"
        for section in report.sections
        for check in section.checks
    )
