import io
from pathlib import Path

import pytest
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from fastapi.testclient import TestClient

from pdfword.accuracy import compute_accuracy_metrics
from pdfword.correction_learning import (
    apply_correction_rules,
    atomic_save_revision,
    compare_docx,
    file_sha256,
    is_sensitive_text,
    validate_docx_bytes,
)
from pdfword.database import Database, utc_now
from pdfword.worker_api import app


def docx_bytes(
    paragraphs: list[str], *, rtl: bool = False, right: bool = False
) -> bytes:
    document = Document()
    for text in paragraphs:
        paragraph = document.add_paragraph(text)
        if right:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if rtl:
            for run in paragraph.runs:
                run.font.rtl = True
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def create_conversion(
    database: Database, root: Path, job_id: str = "job-1", username: str = "alice"
):
    root.mkdir(parents=True, exist_ok=True)
    original = root / "output.docx"
    original.write_bytes(docx_bytes(["هذا نص خاطئ", "فقرة ثانية"]))
    database.create_conversion(
        {
            "job_id": job_id,
            "username": username,
            "original_pdf_name": "input.pdf",
            "stored_pdf_path": str(root / "input.pdf"),
            "output_docx_name": "output.docx",
            "stored_docx_path": str(original),
            "status": "completed",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    return original


def test_revision_files_are_versioned_and_not_overwritten(tmp_path: Path):
    directory = tmp_path / "corrected"
    one = atomic_save_revision(docx_bytes(["الأولى"]), directory, 1)
    two = atomic_save_revision(docx_bytes(["الثانية"]), directory, 2)
    assert one.name == "corrected_v1.docx"
    assert two.name == "corrected_v2.docx"
    assert one.read_bytes() != two.read_bytes()


def test_invalid_docx_is_rejected():
    with pytest.raises(ValueError):
        validate_docx_bytes(b"not-a-docx")


def test_revision_ownership_and_multiple_versions(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    create_conversion(database, tmp_path / "job")
    p1 = docx_bytes(["نسخة أولى"])
    p2 = docx_bytes(["نسخة ثانية"])
    r1 = database.create_correction_revision(
        "job-1", "alice", "v1.docx", file_sha256(p1), len(p1)
    )
    r2 = database.create_correction_revision(
        "job-1", "alice", "v2.docx", file_sha256(p2), len(p2)
    )
    assert [r1["version"], r2["version"]] == [1, 2]
    with pytest.raises(PermissionError):
        database.create_correction_revision("job-1", "mallory", "x.docx", "x", 1)


def test_atomic_save_cannot_escape_revision_directory(tmp_path: Path):
    target = atomic_save_revision(docx_bytes(["آمن"]), tmp_path / "safe", 1)
    assert target.parent == (tmp_path / "safe").resolve()


def test_comparison_extracts_replace_insert_delete_and_arabic(tmp_path: Path):
    original = tmp_path / "original.docx"
    corrected = tmp_path / "corrected.docx"
    original.write_bytes(docx_bytes(["هذا نض خاطئ", "احذف هذه الفقرة"]))
    corrected.write_bytes(docx_bytes(["هذا نص صحيح", "فقرة مضافة"]))
    result = compare_docx(original, corrected)
    types = {row["change_type"] for row in result["examples"]}
    assert result["examples"]
    assert any(row["language"] == "ar" for row in result["examples"])
    assert types & {
        "replacement",
        "arabic_character",
        "paragraph_insertion",
        "paragraph_deletion",
    }


def test_comparison_detects_rtl_and_alignment(tmp_path: Path):
    original = tmp_path / "a.docx"
    corrected = tmp_path / "b.docx"
    original.write_bytes(docx_bytes(["مرحبا"]))
    corrected.write_bytes(docx_bytes(["مرحبا"], rtl=True, right=True))
    changes = compare_docx(original, corrected)["formatting_changes"]
    properties = {row["property"] for row in changes}
    assert "alignment" in properties
    assert "rtl" in properties


@pytest.mark.parametrize(
    "value",
    [
        "01012345678",
        "2026-07-04",
        "name@example.com",
        "https://example.com",
        "12345678901234",
        "42",
    ],
)
def test_sensitive_values(value: str):
    assert is_sensitive_text(value)


def test_sensitive_example_never_becomes_general_rule(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    create_conversion(database, tmp_path / "job")
    revision = database.create_correction_revision(
        "job-1", "alice", "v.docx", "hash", 10, "learning"
    )
    database.replace_revision_examples(
        revision["id"],
        "job-1",
        "alice",
        [
            {
                "wrong_text": "01000000000",
                "correct_text": "01111111111",
                "change_type": "replacement",
                "language": "ar",
                "is_sensitive": True,
            }
        ],
    )
    example = database.list_correction_examples(revision["id"])[0]
    database.review_correction_example(example["id"], "approved", "alice")
    assert database.list_memory_rules() == []


def test_rule_is_not_auto_approved_and_requires_sources(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    for index in range(3):
        job_id = f"job-{index}"
        create_conversion(database, tmp_path / job_id, job_id)
        revision = database.create_correction_revision(
            job_id, "alice", f"{index}.docx", f"h{index}", 10, "learning"
        )
        database.replace_revision_examples(
            revision["id"],
            job_id,
            "alice",
            [
                {
                    "wrong_text": "نض",
                    "correct_text": "نص",
                    "change_type": "arabic_character",
                    "language": "ar",
                    "is_sensitive": False,
                }
            ],
        )
        example = database.list_correction_examples(revision["id"])[0]
        database.review_correction_example(example["id"], "approved", "alice")
    rule = database.list_memory_rules()[0]
    assert rule["source_count"] == 3
    assert rule["approved"] == 0 and rule["enabled"] == 0
    assert database.active_memory_rules() == []
    database.set_memory_rule_state(rule["id"], True, True, "alice")
    assert len(database.active_memory_rules()) == 1


def test_review_only_revision_does_not_feed_memory(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    create_conversion(database, tmp_path / "job")
    revision = database.create_correction_revision(
        "job-1", "alice", "review.docx", "review-hash", 10, "review_only"
    )
    database.replace_revision_examples(
        revision["id"],
        "job-1",
        "alice",
        [
            {
                "wrong_text": "نض",
                "correct_text": "نص",
                "change_type": "arabic_character",
                "language": "ar",
                "is_sensitive": False,
            }
        ],
    )
    example = database.list_correction_examples(revision["id"])[0]
    database.review_correction_example(example["id"], "approved", "alice")
    assert database.list_memory_rules() == []


def test_worker_rule_application_and_conflict_resolution():
    rules = [
        {
            "id": 1,
            "wrong_text": "نض",
            "correct_text": "نص",
            "confidence": 0.95,
            "threshold": 0.9,
            "approved": 1,
            "enabled": 1,
            "is_sensitive": 0,
            "scope": "global",
        },
        {
            "id": 2,
            "wrong_text": "نض",
            "correct_text": "نص آخر",
            "confidence": 0.91,
            "threshold": 0.9,
            "approved": 1,
            "enabled": 1,
            "is_sensitive": 0,
            "scope": "global",
        },
        {
            "id": 3,
            "wrong_text": "قديم",
            "correct_text": "جديد",
            "confidence": 0.99,
            "threshold": 0.9,
            "approved": 0,
            "enabled": 1,
            "is_sensitive": 0,
            "scope": "global",
        },
    ]
    result = apply_correction_rules("هذا نض قديم", rules)
    assert result.text == "هذا نص قديم"
    assert [row["rule_id"] for row in result.applications] == [1]
    assert [row["rule_id"] for row in result.conflicts] == [2]


def test_accuracy_before_after_is_measurable():
    reference = "هذا نص صحيح"
    before = compute_accuracy_metrics(reference, "هذا نض صحيح")
    after = compute_accuracy_metrics(reference, "هذا نص صحيح")
    assert after["word_accuracy"] > before["word_accuracy"]


def test_dataset_exports_only_approved_non_sensitive_examples(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    create_conversion(database, tmp_path / "job")
    revision = database.create_correction_revision(
        "job-1", "alice", "v.docx", "hash", 10, "learning"
    )
    database.replace_revision_examples(
        revision["id"],
        "job-1",
        "alice",
        [
            {
                "wrong_text": "نض",
                "correct_text": "نص",
                "context_before": "هذا",
                "context_after": "صحيح",
                "change_type": "arabic_character",
                "language": "ar",
                "is_sensitive": False,
            }
        ],
    )
    example = database.list_correction_examples(revision["id"])[0]
    assert database.export_approved_dataset() == ""
    database.review_correction_example(example["id"], "approved", "alice")
    exported = database.export_approved_dataset()
    assert '"wrong_text": "نض"' in exported
    assert "alice" not in exported and "job-1" not in exported


def test_snapshot_contains_only_active_trusted_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database_path = tmp_path / "db.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    monkeypatch.setenv("WORKER_API_KEY", "secret")
    monkeypatch.setenv("CORRECTION_MIN_SOURCE_FILES", "3")
    monkeypatch.setenv("CORRECTION_AUTO_APPLY_THRESHOLD", "0.90")
    database = Database(database_path)
    with database.transaction() as connection:
        for wrong, approved, sensitive in [
            ("نض", 1, 0),
            ("غير معتمد", 0, 0),
            ("01000000000", 1, 1),
        ]:
            connection.execute(
                """
                INSERT INTO correction_memory_rules(
                    wrong_text,correct_text,scope,source_count,occurrence_count,
                    accepted_count,confidence,is_sensitive,approved,enabled,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    wrong,
                    "نص",
                    "global",
                    3,
                    3,
                    3,
                    0.95,
                    sensitive,
                    approved,
                    approved,
                    utc_now(),
                ),
            )
    response = TestClient(app).get(
        "/internal/corrections/snapshot", headers={"X-Worker-API-Key": "secret"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["checksum"] and payload["version"]
    assert [rule["wrong_text"] for rule in payload["rules"]] == ["نض"]


def test_evaluation_split_and_summary(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    database.record_correction_evaluation(
        1,
        "job-eval",
        "evaluation",
        {"word_accuracy": 70, "char_accuracy": 80},
        {"word_accuracy": 90, "char_accuracy": 92},
    )
    summary = database.correction_evaluation_summary()
    assert summary["samples"] == 1
    assert summary["improvement"] == 20
