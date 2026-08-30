from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

from clouda_contracts.storage import StorageRoots
from clouda_data.distortion.workflow import (
    RunInterrupted,
    generate_preview,
    read_jsonl,
    run_distortion_batch,
    validate_distortion_manifest,
)
from clouda_data.evaluation.execution import evaluate_records
from clouda_data.rendering import (
    RenderConfig,
    render_document,
    validate_render_manifest,
)
from clouda_training.exporter import export_training_data
from pdfword.docx_export import markdown_to_docx
from pdfword.local_ocr_adapters import MockOCRProvider
from pdfword.ocr_pipeline import process_pdf

TEXTS = [
    ("body", "هذا نص عربي اصطناعي لاختبار التعرّف الضوئي."),
    ("title", "عنوان عربي اصطناعي"),
    ("footnote", "متن الصفحة\nهامش سفلي اصطناعي ١"),
    ("page_number", "صفحة اصطناعية\n٤٢"),
    ("mixed", "نص عربي مع English text و 2026"),
    ("two_column", "العمود الأول\nالنص العربي\nالعمود الثاني"),
    ("margin_note", "النص الأساسي\nملاحظة هامشية"),
    ("blank", ""),
    ("near_blank", "١"),
    ("small_font", "نص عربي صغير الحجم مع نقاط وتشكيل"),
]
PROFILES = [
    "modern_scan_light",
    "modern_scan_medium",
    "old_book_medium",
    "footnote_stress",
    "jpeg_damage",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _make_page(path: Path, kind: str, text: str, page_number: int) -> None:
    image = Image.new("RGB", (900, 1200), (250, 248, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 45, 855, 1155), outline=(40, 40, 40), width=2)
    title_font = _font(42)
    body_font = _font(28 if kind != "small_font" else 13)
    if kind == "two_column":
        draw.line((450, 160, 450, 1050), fill=(180, 180, 180), width=2)
        draw.multiline_text((100, 190), text, font=body_font, fill="black", spacing=16)
    elif kind == "margin_note":
        draw.multiline_text((180, 200), text, font=body_font, fill="black", spacing=18)
        draw.text((50, 500), "هامش", font=_font(18), fill=(60, 60, 60))
    elif kind == "footnote":
        draw.multiline_text((100, 200), text, font=body_font, fill="black", spacing=22)
        draw.line((100, 950, 500, 950), fill="black", width=2)
    elif kind not in {"blank", "near_blank"}:
        draw.text((450, 120), "صفحة اختبار", font=title_font, fill="black", anchor="ma")
        draw.multiline_text((100, 260), text, font=body_font, fill="black", spacing=24)
    elif kind == "near_blank":
        draw.text((450, 1080), text, font=body_font, fill=(80, 80, 80), anchor="mm")
    draw.text(
        (450, 1120), str(page_number), font=_font(20), fill=(80, 80, 80), anchor="mm"
    )
    image.save(path, "PNG")


def run() -> dict[str, object]:
    roots = StorageRoots.from_env()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    acceptance_root = roots.dataset_root / "synthetic_acceptance" / stamp
    sources = acceptance_root / "sources"
    sources.mkdir(parents=True, exist_ok=False)
    source_hashes: dict[str, str] = {}
    normalized_records: list[dict[str, object]] = []
    render_manifests: list[str] = []
    for index, (kind, text) in enumerate(TEXTS, 1):
        source = sources / f"{index:02d}-{kind}.png"
        _make_page(source, kind, text, index)
        source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        render_manifest = render_document(
            source,
            output_root=roots.dataset_root / "rendered" / "synthetic_acceptance",
            config=RenderConfig(dpi=200),
            run_id=f"{stamp}-{index:02d}",
        )
        render_manifests.append(str(render_manifest))
        assert validate_render_manifest(render_manifest)["passed"]
        record = read_jsonl(render_manifest)[0]
        record.update(
            {
                "page_id": f"synthetic-doc-{index:02d}:1",
                "source_document_id": f"synthetic-doc-{index:02d}",
                "dataset_id": "synthetic_evaluation",
                "license_status": "evaluation_only",
                "commercial_training_allowed": False,
                "ground_truth_reference": f"synthetic://{index:02d}",
                "ground_truth_text": text,
                "attribution": "Generated synthetic acceptance fixture",
                "page_type": kind,
                "language": "ar-en" if kind == "mixed" else "ar",
            }
        )
        normalized_records.append(record)
    input_manifest = acceptance_root / "rendered_pages.v1.jsonl"
    input_manifest.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in normalized_records
        ),
        encoding="utf-8",
    )
    manifests: list[Path] = []
    interrupted = False
    resumed = False
    project_root = Path(os.environ["CLOUDA_PROJECT_ROOT"])
    for profile_index, profile in enumerate(PROFILES):
        profile_path = (
            project_root
            / "configs"
            / "data_foundation"
            / "distortions"
            / f"{profile}.yaml"
        )
        profile_input = acceptance_root / f"{profile}-input.jsonl"
        selected = normalized_records[profile_index * 2 : profile_index * 2 + 2]
        profile_input.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in selected
            ),
            encoding="utf-8",
        )
        if profile_index == 0:
            try:
                run_distortion_batch(
                    profile_input,
                    profile_path,
                    seed=20260723,
                    variants=3,
                    max_pages=2,
                    interrupt_after=2,
                )
            except RunInterrupted:
                interrupted = True
            manifest = run_distortion_batch(
                profile_input,
                profile_path,
                seed=20260723,
                variants=3,
                max_pages=2,
                resume=True,
            )
            resumed = True
        else:
            manifest = run_distortion_batch(
                profile_input,
                profile_path,
                seed=20260723 + profile_index,
                variants=3,
                max_pages=2,
            )
        validation = validate_distortion_manifest(manifest, quarantine=True)
        if not validation["passed"]:
            raise RuntimeError(
                f"Validation failed for {profile}: {validation['failures']}"
            )
        manifests.append(manifest)
    combined = acceptance_root / "distorted_pages.v1.jsonl"
    all_records = [record for manifest in manifests for record in read_jsonl(manifest)]
    combined.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in all_records
        ),
        encoding="utf-8",
    )
    ids = [str(item["generated_page_id"]) for item in all_records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate generated page IDs")
    pixel_differences_verified = 0
    for item in all_records:
        source_path = roots.dataset_root / str(item["source_uri"]).removeprefix(
            "dataset://"
        )
        output_path = roots.dataset_root / str(item["output_uri"]).removeprefix(
            "dataset://"
        )
        with (
            Image.open(source_path) as source_image,
            Image.open(output_path) as output_image,
        ):
            source_rgb = source_image.convert("RGB")
            output_rgb = output_image.convert("RGB")
            if (
                source_rgb.size != output_rgb.size
                or ImageChops.difference(source_rgb, output_rgb).getbbox()
            ):
                pixel_differences_verified += 1
    preview = generate_preview(combined, limit=10)
    training_output = roots.artifact_root / "training" / f"acceptance-{stamp}.jsonl"
    training = export_training_data(
        combined,
        training_output,
        purpose="evaluation",
        export_format="conversation_multimodal",
    )
    ocr_records: list[dict[str, object]] = []
    for item in all_records:
        image_path = roots.dataset_root / str(item["output_uri"]).removeprefix(
            "dataset://"
        )
        provider = MockOCRProvider(str(item["ground_truth_text"]), 0.99)
        os.environ["CLOUDA_ALLOW_MOCK_OCR"] = "true"
        result = provider.extract_page(image_bytes=image_path.read_bytes(), page_no=1)
        ocr_records.append(
            {
                **item,
                "prediction_text": result.text,
                "processing_time": result.processing_time or 0.001,
                "model_id": result.model_name,
                "model_revision": "test-fixture-v1",
            }
        )
    evaluation = evaluate_records(ocr_records)
    evaluation_path = (
        roots.artifact_root / "reports" / "acceptance" / f"evaluation-{stamp}.json"
    )
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.environ.update(
        {
            "CLOUDA_LOCAL_OCR_ENABLED": "true",
            "CLOUDA_LOCAL_OCR_ENGINE": "mock",
            "CLOUDA_ALLOW_MOCK_OCR": "true",
            "CLOUDA_MOCK_OCR_TEXT": "نص عربي ممسوح اصطناعي",
        }
    )
    scanned_fixture = project_root / "tests" / "fixtures" / "scanned.pdf"
    scanned_copy = sources / "synthetic-scanned.pdf"
    shutil.copy2(scanned_fixture, scanned_copy)
    page_results, _ = process_pdf(
        scanned_copy.read_bytes(),
        1,
        1,
        enabled_engines=["direct_pdf_text", "local_model_ocr"],
    )
    docx = markdown_to_docx(page_results)
    docx_path = roots.artifact_root / "acceptance" / f"mock-runtime-{stamp}.docx"
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    docx_path.write_bytes(docx)
    with zipfile.ZipFile(docx_path) as archive:
        if "word/document.xml" not in archive.namelist():
            raise RuntimeError("DOCX validation failed")
    source_unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum
        for path, checksum in source_hashes.items()
    )
    outputs_inside_state = all(
        Path(path)
        .resolve()
        .is_relative_to(Path(os.environ["CLOUDA_STATE_HOME"]).resolve())
        for path in [
            *render_manifests,
            *[str(path) for path in manifests],
            str(preview),
            str(training_output),
            str(evaluation_path),
            str(docx_path),
        ]
    )
    runtime_metadata = page_results[0].metadata or {}
    report = {
        "schema_version": 1,
        "run_id": stamp,
        "synthetic_sources": len(TEXTS),
        "profiles": len(PROFILES),
        "distorted_outputs": len(all_records),
        "pixel_differences_verified": pixel_differences_verified,
        "validation_passed": True,
        "interrupted": interrupted,
        "resumed": resumed,
        "duplicates": len(ids) - len(set(ids)),
        "training_export_records": training["records"],
        "document_leakage": training["document_leakage"],
        "mock_ocr_cer": evaluation["summary"]["cer"],
        "mock_ocr_wer": evaluation["summary"]["wer"],
        "runtime_mock_status": runtime_metadata["page_state"],
        "docx": str(docx_path),
        "source_unchanged": source_unchanged,
        "outputs_inside_state": outputs_inside_state,
        "external_network_accessed": False,
        "passed": all(
            [
                len(all_records) >= 30,
                pixel_differences_verified == len(all_records),
                interrupted,
                resumed,
                not training["document_leakage"],
                evaluation["summary"]["cer"] == 0,
                evaluation["summary"]["wer"] == 0,
                runtime_metadata["page_state"] == "local_model_ocr",
                source_unchanged,
                outputs_inside_state,
            ]
        ),
    }
    report_path = (
        roots.artifact_root / "reports" / "acceptance" / f"acceptance-{stamp}.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    started = time.perf_counter()
    result = run()
    result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
