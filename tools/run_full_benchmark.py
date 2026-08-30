# ruff: noqa: E402

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdfword.accuracy import (
    compute_accuracy_metrics,
    evaluate_pages_against_references,
)  # noqa: E402
from pdfword.auto_eval import estimate_ai_fidelity_score  # noqa: E402
from pdfword.key_store import load_saved_api_key  # noqa: E402
from pdfword.ocr_pipeline import process_pdf  # noqa: E402
from pdfword.openrouter_client import resolve_models  # noqa: E402
from pdfword.model_registry import ModelRegistry  # noqa: E402
from tools.generate_test_pdfs import (  # noqa: E402
    MIN_SAMPLES_PER_CATEGORY,
    REQUIRED_CATEGORIES,
    main as generate_test_manifest,
    validate_manifest_cases,
)


class _NoopProgress:
    def progress(self, *args, **kwargs):
        return None


class _NoopStatus:
    def info(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _reference_is_valid(text: str) -> bool:
    sample = (text or "").strip()
    if len(sample) < 20:
        return False
    question_ratio = sample.count("?") / max(1, len(sample))
    if question_ratio > 0.08:
        return False
    letter_count = len(__import__("re").findall(r"[A-Za-z\u0600-\u06FF]", sample))
    return letter_count >= 15


def _case_pages(pdf_bytes: bytes, from_manifest: list[int] | None) -> list[int]:
    if from_manifest:
        return sorted(set(int(p) for p in from_manifest if int(p) > 0))
    total = len(PdfReader(__import__("io").BytesIO(pdf_bytes)).pages)
    return list(range(1, total + 1))


def _build_category_summary(cases: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        grouped[c.get("category", "unknown")].append(c)

    rows: list[dict] = []
    for category, rows_in_cat in sorted(grouped.items()):
        valid_word = [
            r["word_accuracy"]
            for r in rows_in_cat
            if r.get("word_accuracy") is not None
        ]
        valid_char = [
            r["char_accuracy"]
            for r in rows_in_cat
            if r.get("char_accuracy") is not None
        ]
        valid_visual = [
            r["visual_score"] for r in rows_in_cat if r.get("visual_score") is not None
        ]
        avg_time = sum(r.get("elapsed_sec", 0.0) for r in rows_in_cat) / max(
            1, len(rows_in_cat)
        )
        ok_cases = sum(
            1 for r in rows_in_cat if r.get("status") in ("ok", "expected_fail_ok")
        )
        rows.append(
            {
                "category": category,
                "count": len(rows_in_cat),
                "avg_word_accuracy": (
                    (sum(valid_word) / len(valid_word)) if valid_word else None
                ),
                "avg_char_accuracy": (
                    (sum(valid_char) / len(valid_char)) if valid_char else None
                ),
                "avg_visual_score": (
                    (sum(valid_visual) / len(valid_visual)) if valid_visual else None
                ),
                "avg_elapsed_sec": avg_time,
                "success_rate": (ok_cases / max(1, len(rows_in_cat))) * 100.0,
            }
        )
    return rows


def _weakest_categories(category_summary: list[dict], limit: int = 3) -> list[dict]:
    scored = []
    for r in category_summary:
        score = (
            r["avg_word_accuracy"]
            if r.get("avg_word_accuracy") is not None
            else r.get("avg_char_accuracy")
        )
        if score is None:
            continue
        scored.append((float(score), r))
    scored.sort(key=lambda item: item[0])
    return [item[1] for item in scored[:limit]]


def _performance_comparison(category_summary: list[dict]) -> dict:
    if not category_summary:
        return {
            "fastest_category": None,
            "slowest_category": None,
            "time_gap_sec": None,
        }
    by_time = sorted(category_summary, key=lambda r: r.get("avg_elapsed_sec", 0.0))
    fastest = by_time[0]
    slowest = by_time[-1]
    return {
        "fastest_category": fastest["category"],
        "fastest_avg_elapsed_sec": fastest["avg_elapsed_sec"],
        "slowest_category": slowest["category"],
        "slowest_avg_elapsed_sec": slowest["avg_elapsed_sec"],
        "time_gap_sec": slowest["avg_elapsed_sec"] - fastest["avg_elapsed_sec"],
    }


def _manifest_file_errors(root: Path, cases: list[dict]) -> list[str]:
    errors: list[str] = []
    for c in cases:
        case_id = c.get("id", "<unknown>")
        pdf_rel = c.get("pdf")
        if not pdf_rel or not (root / pdf_rel).exists():
            errors.append(f"missing pdf for case: {case_id}")
        ref_rel = c.get("reference")
        if ref_rel and not (root / ref_rel).exists():
            errors.append(f"missing reference for case: {case_id}")
    return errors


def _validate_selected_categories(
    cases: list[dict], *, require_all_categories: bool
) -> list[str]:
    if not require_all_categories:
        return []
    counts: dict[str, int] = {}
    for c in cases:
        cat = str(c.get("category", "")).strip()
        counts[cat] = counts.get(cat, 0) + 1
    errors: list[str] = []
    for cat in REQUIRED_CATEGORIES:
        count = counts.get(cat, 0)
        if count == 0:
            errors.append(f"category skipped by selection: {cat}")
        elif count < MIN_SAMPLES_PER_CATEGORY:
            errors.append(f"category has too few selected samples ({count}): {cat}")
    return errors


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _category_execution_plan(
    category: str, default_mode: str, fast_model: str, accurate_model: str
) -> dict:
    cat = (category or "").strip().lower()
    hard_categories = {
        "old_archived_scans",
        "low_quality_scans",
        "blurry_docs",
        "fax_style_pdfs",
        "tilted_rotated_scans",
        "mixed_multi_page_docs",
    }
    easy_categories = {"born_digital_modern", "clean_scans"}

    if cat in hard_categories:
        return {
            "mode": "max_accuracy",
            "fast_model": accurate_model or fast_model,
            "accurate_model": accurate_model or fast_model,
            "policy": "hard_category_force_accurate",
        }
    if cat in easy_categories:
        chosen_mode = (
            "turbo" if default_mode in {"turbo", "balanced", "hyper"} else default_mode
        )
        return {
            "mode": chosen_mode,
            "fast_model": fast_model,
            "accurate_model": accurate_model,
            "policy": "easy_category_fast_route",
        }
    return {
        "mode": default_mode,
        "fast_model": fast_model,
        "accurate_model": accurate_model,
        "policy": "default",
    }


def _markdown_report(results: dict) -> str:
    lines = []
    lines.append("# Benchmark Report")
    lines.append("")
    lines.append(f"- Generated at: {results['generated_at']}")
    lines.append(f"- Mode: `{results['mode']}`")
    lines.append(f"- Fast model: `{results['fast_model']}`")
    lines.append(f"- Accurate model: `{results['accurate_model']}`")
    lines.append(f"- Total cases: {results['summary']['total_cases']}")
    lines.append(f"- Success: {results['summary']['success_cases']}")
    lines.append(f"- Fail: {results['summary']['failed_cases']}")
    lines.append(
        f"- Expected failures passed: {results['summary']['expected_failures_ok']}"
    )
    lines.append("")
    lines.append("## Case Results")
    lines.append("")
    lines.append(
        "| Case | Category | Mode | Policy | Status | Time(s) | WER% | CER% | Word Acc% | Char Acc% | Visual% | Manual Review | Route | Models | Ref |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for c in results["cases"]:
        wa = f"{c['word_accuracy']:.2f}" if c.get("word_accuracy") is not None else "-"
        ca = f"{c['char_accuracy']:.2f}" if c.get("char_accuracy") is not None else "-"
        wer = f"{c['wer']:.2f}" if c.get("wer") is not None else "-"
        cer = f"{c['cer']:.2f}" if c.get("cer") is not None else "-"
        vs = f"{c['visual_score']:.2f}" if c.get("visual_score") is not None else "-"
        models = ", ".join(c.get("models_used", [])) or "-"
        routes = ", ".join(c.get("route_used", [])) or "-"
        ref_ok = c.get("reference_quality", "-")
        lines.append(
            f"| `{c['id']}` | {c['category']} | {c.get('processing_mode', '-')} | {c.get('model_policy', '-')} | {c['status']} | {c['elapsed_sec']:.2f} | {wer} | {cer} | {wa} | {ca} | {vs} | {bool(c.get('manual_review'))} | {routes} | {models} | {ref_ok} |"
        )

    referenced = [c for c in results["cases"] if c.get("reference_quality") == "valid"]
    if referenced:
        lines.append("")
        lines.append("## Reference-Based Accuracy Details")
        lines.append("")
        for c in referenced:
            lines.append(f"### {c['id']}")
            lines.append("")
            lines.append(f"- Route used: `{', '.join(c.get('route_used', [])) or '-'}`")
            lines.append(
                f"- Engine/model used: `{', '.join(c.get('models_used', [])) or '-'}`"
            )
            lines.append(f"- Manual review: `{bool(c.get('manual_review'))}`")
            lines.append(f"- CER: `{c.get('cer')}`")
            lines.append(f"- WER: `{c.get('wer')}`")
            lines.append("")
            lines.append("Expected text:")
            lines.append("")
            lines.append("```text")
            lines.append((c.get("expected_text") or "")[:3000])
            lines.append("```")
            lines.append("")
            lines.append("Extracted text:")
            lines.append("")
            lines.append("```text")
            lines.append((c.get("extracted_text") or "")[:3000])
            lines.append("```")

    category_summary = results.get("category_summary", [])
    if category_summary:
        lines.append("")
        lines.append("## Per-Type Accuracy and Performance")
        lines.append("")
        lines.append(
            "| Category | Count | Success% | Avg Time(s) | Word Acc% | Char Acc% | Visual% |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in category_summary:
            wa = (
                f"{r['avg_word_accuracy']:.2f}"
                if r.get("avg_word_accuracy") is not None
                else "-"
            )
            ca = (
                f"{r['avg_char_accuracy']:.2f}"
                if r.get("avg_char_accuracy") is not None
                else "-"
            )
            vs = (
                f"{r['avg_visual_score']:.2f}"
                if r.get("avg_visual_score") is not None
                else "-"
            )
            lines.append(
                f"| {r['category']} | {r['count']} | {r['success_rate']:.2f} | {r['avg_elapsed_sec']:.2f} | {wa} | {ca} | {vs} |"
            )

    weakest = results.get("weakest_categories", [])
    if weakest:
        lines.append("")
        lines.append("## Weakest Categories")
        lines.append("")
        for w in weakest:
            wa = (
                f"{w['avg_word_accuracy']:.2f}"
                if w.get("avg_word_accuracy") is not None
                else "-"
            )
            ca = (
                f"{w['avg_char_accuracy']:.2f}"
                if w.get("avg_char_accuracy") is not None
                else "-"
            )
            lines.append(
                f"- {w['category']}: word={wa} char={ca} success={w['success_rate']:.2f}%"
            )

    perf = results.get("performance_comparison", {})
    if perf and perf.get("fastest_category") and perf.get("slowest_category"):
        lines.append("")
        lines.append("## Performance Comparison")
        lines.append("")
        lines.append(
            f"- Fastest: `{perf['fastest_category']}` ({perf['fastest_avg_elapsed_sec']:.2f}s avg)"
        )
        lines.append(
            f"- Slowest: `{perf['slowest_category']}` ({perf['slowest_avg_elapsed_sec']:.2f}s avg)"
        )
        lines.append(f"- Gap: {perf['time_gap_sec']:.2f}s")

    failed = [
        c for c in results["cases"] if c["status"] not in ("ok", "expected_fail_ok")
    ]
    if failed:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for c in failed:
            lines.append(f"- `{c['id']}`: {c.get('error', 'unknown error')}")

    return "\n".join(lines) + "\n"


def passes_text_acceptance(row: dict) -> bool:
    if int(row.get("pages_below_90") or 0) > 0:
        return False
    if row.get("reference_quality") == "valid":
        char_accuracy = row.get("char_accuracy")
        word_accuracy = row.get("word_accuracy")
        return (
            char_accuracy is not None
            and word_accuracy is not None
            and float(char_accuracy) >= 90.0
            and float(word_accuracy) >= 90.0
        )
    return True


def run_benchmark(
    root: Path,
    manifest_path: Path,
    mode: str,
    max_cases: int,
    language_filter: str,
    max_parallel_pages: int,
    free_only: bool = False,
    skip_visual_judge: bool = False,
) -> int:
    payload = _load_manifest(manifest_path)
    all_cases = payload.get("cases", [])
    require_all_categories = language_filter == "all" and max_cases <= 0
    selected_validation = _validate_selected_categories(
        all_cases, require_all_categories=require_all_categories
    )
    if selected_validation:
        print("ERROR: benchmark selection invalid:")
        for msg in selected_validation:
            print(f" - {msg}")
        return 2
    if language_filter != "all":
        all_cases = [c for c in all_cases if c.get("language") == language_filter]
    if max_cases > 0:
        all_cases = all_cases[:max_cases]

    api_key = load_saved_api_key() or __import__("os").getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is missing.")
        return 2

    fast_model, accurate_model, used_fallback = resolve_models()
    free_models: list[str] = []
    if free_only:
        free_models = [model.id for model in ModelRegistry().ranked()["free_vision"]]
        if not free_models:
            print("ERROR: no free Vision models are currently available.")
            return 2
        fast_model = free_models[0]
        accurate_model = free_models[min(1, len(free_models) - 1)]
    print(
        f"Models: fast={fast_model} accurate={accurate_model} fallback={used_fallback}"
    )
    print(f"Running {len(all_cases)} cases...")

    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "fast_model": fast_model,
        "accurate_model": accurate_model,
        "cases": [],
        "summary": {},
    }

    noop_progress = _NoopProgress()
    noop_status = _NoopStatus()

    for idx, case in enumerate(all_cases, start=1):
        case_id = case["id"]
        category = case.get("category", "unknown")
        expect_failure = bool(case.get("expect_failure", False))
        pdf_path = root / case["pdf"]
        ref_path = root / case["reference"] if case.get("reference") else None

        start = time.time()
        row = {
            "id": case_id,
            "category": category,
            "expect_failure": expect_failure,
            "status": "fail",
            "elapsed_sec": 0.0,
            "word_accuracy": None,
            "char_accuracy": None,
            "visual_score": None,
            "estimated_text_quality": None,
            "pages_below_90": 0,
            "models_used": [],
            "route_used": [],
            "engine_used": [],
            "processing_mode": mode,
            "model_policy": "default",
            "reference_quality": "-",
            "expected_text": "",
            "extracted_text": "",
            "cer": None,
            "wer": None,
            "per_page_metrics": [],
            "document_metrics": None,
            "manual_review": False,
            "error": "",
        }

        print(f"[{idx}/{len(all_cases)}] {case_id} ({category})")
        try:
            pdf_bytes = pdf_path.read_bytes()
            pages = _case_pages(pdf_bytes, case.get("pages"))
            plan = _category_execution_plan(
                category=category,
                default_mode=mode,
                fast_model=fast_model,
                accurate_model=accurate_model,
            )
            row["processing_mode"] = plan["mode"]
            row["model_policy"] = plan["policy"]
            page_results, full_markdown = process_pdf(
                pdf_bytes=pdf_bytes,
                from_page=None,
                to_page=None,
                api_key=api_key,
                fast_model=plan["fast_model"],
                accurate_model=plan["accurate_model"],
                progress_bar=noop_progress,
                status_placeholder=noop_status,
                speed_mode=plan["mode"],
                page_numbers=pages,
                max_parallel_pages=max_parallel_pages,
                doc_category=category,
                enabled_models=free_models or None,
            )

            row["models_used"] = sorted(set(p.model_used for p in page_results))
            row["engine_used"] = row["models_used"]
            row["route_used"] = sorted(
                set(p.route_used or "" for p in page_results if p.route_used)
            )
            row["manual_review"] = any(p.requires_manual_review for p in page_results)
            text_scores = [
                float(page.text_quality_score)
                for page in page_results
                if page.text_quality_score is not None
            ]
            row["estimated_text_quality"] = (
                sum(text_scores) / len(text_scores) if text_scores else 0.0
            )
            row["pages_below_90"] = sum(
                1
                for page in page_results
                if page.text_quality_score is None
                or float(page.text_quality_score) < 90.0
            )
            if not skip_visual_judge:
                row["visual_score"] = estimate_ai_fidelity_score(
                    api_key=api_key,
                    pdf_bytes=pdf_bytes,
                    page_results=page_results,
                    judge_model=accurate_model or fast_model,
                )

            if ref_path and ref_path.exists():
                ref_text = _read_text(ref_path)
                if _reference_is_valid(ref_text):
                    metrics = compute_accuracy_metrics(ref_text, full_markdown)
                    reference_eval = evaluate_pages_against_references(
                        page_results, [ref_text]
                    )
                    row["word_accuracy"] = metrics["word_accuracy"]
                    row["char_accuracy"] = metrics["char_accuracy"]
                    row["cer"] = metrics["cer"]
                    row["wer"] = metrics["wer"]
                    row["expected_text"] = metrics["reference_text"]
                    row["extracted_text"] = metrics["extracted_text"]
                    row["per_page_metrics"] = reference_eval["pages"]
                    row["document_metrics"] = reference_eval["document"]
                    row["reference_quality"] = "valid"
                else:
                    row["reference_quality"] = "invalid"
            elif ref_path:
                row["reference_quality"] = "missing"

            quality_passed = passes_text_acceptance(row)
            if expect_failure:
                row["status"] = "expected_fail_but_succeeded"
            else:
                row["status"] = "ok" if quality_passed else "fail_text_quality"
                if not quality_passed:
                    row["error"] = "Text quality below mandatory 90% threshold"
        except Exception as exc:
            row["error"] = str(exc)
            if expect_failure:
                row["status"] = "expected_fail_ok"
            else:
                row["status"] = "fail"
        finally:
            row["elapsed_sec"] = time.time() - start
            results["cases"].append(row)
            print(
                f"  -> {row['status']} | t={row['elapsed_sec']:.2f}s | "
                f"word={row['word_accuracy'] if row['word_accuracy'] is not None else '-'} | "
                f"char={row['char_accuracy'] if row['char_accuracy'] is not None else '-'} | "
                f"visual={row['visual_score'] if row['visual_score'] is not None else '-'}"
            )

    total = len(results["cases"])
    success = sum(1 for c in results["cases"] if c["status"] == "ok")
    expected_fail_ok = sum(
        1 for c in results["cases"] if c["status"] == "expected_fail_ok"
    )
    failed = sum(
        1 for c in results["cases"] if c["status"] not in ("ok", "expected_fail_ok")
    )
    avg_word = [
        c["word_accuracy"]
        for c in results["cases"]
        if c.get("word_accuracy") is not None
    ]
    avg_char = [
        c["char_accuracy"]
        for c in results["cases"]
        if c.get("char_accuracy") is not None
    ]

    results["summary"] = {
        "total_cases": total,
        "success_cases": success,
        "expected_failures_ok": expected_fail_ok,
        "failed_cases": failed,
        "avg_word_accuracy": (sum(avg_word) / len(avg_word)) if avg_word else None,
        "avg_char_accuracy": (sum(avg_char) / len(avg_char)) if avg_char else None,
    }
    category_summary = _build_category_summary(results["cases"])
    results["category_summary"] = category_summary
    results["weakest_categories"] = _weakest_categories(category_summary)
    results["performance_comparison"] = _performance_comparison(category_summary)

    runtime_validation = _validate_selected_categories(
        results["cases"], require_all_categories=require_all_categories
    )
    if runtime_validation:
        print("ERROR: benchmark runtime category validation failed:")
        for msg in runtime_validation:
            print(f" - {msg}")
        failed += len(runtime_validation)
        results["summary"]["failed_cases"] = results["summary"]["failed_cases"] + len(
            runtime_validation
        )

    out_dir = manifest_path.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_report_{stamp}.json"
    md_path = out_dir / f"benchmark_report_{stamp}.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(_markdown_report(results), encoding="utf-8")

    print(f"Report JSON: {json_path}")
    print(f"Report MD:   {md_path}")
    print(
        f"Summary: total={total} ok={success} expected_fail_ok={expected_fail_ok} fail={failed} "
        f"avg_word={results['summary']['avg_word_accuracy']} avg_char={results['summary']['avg_char_accuracy']}"
    )

    return 1 if failed > 0 else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end OCR benchmark over mixed PDF types."
    )
    parser.add_argument(
        "--manifest",
        default="samples/generated/cases_manifest.json",
        help="Path to JSON manifest.",
    )
    parser.add_argument(
        "--mode",
        default="hyper",
        choices=["hyper", "max_accuracy", "balanced", "turbo"],
    )
    parser.add_argument(
        "--mode-sequence",
        default="",
        help="Comma-separated modes, e.g. turbo,max_accuracy",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap to first N cases (0 = all).",
    )
    parser.add_argument("--language", default="all", choices=["all", "ar", "en"])
    parser.add_argument(
        "--max-parallel-pages",
        type=int,
        default=8,
        help="Parallel pages for process_pdf.",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=0,
        help="If >0, repeat benchmark rounds until timeout.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=int,
        default=0,
        help="Pause between rounds when duration mode is active.",
    )
    parser.add_argument(
        "--free-only",
        action="store_true",
        help="Use only currently discovered free Vision models.",
    )
    parser.add_argument(
        "--skip-visual-judge",
        action="store_true",
        help="Skip the additional visual judge request.",
    )
    parser.add_argument(
        "--auto-generate-missing-categories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate benchmark PDFs/manifest if required categories are missing.",
    )
    args = parser.parse_args()

    root = ROOT
    manifest_path = (root / args.manifest).resolve()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        raise SystemExit(2)
    payload = _load_manifest(manifest_path)
    cases = payload.get("cases", [])
    coverage_errors = validate_manifest_cases(cases)
    file_errors = _manifest_file_errors(root, cases)
    if (coverage_errors or file_errors) and args.auto_generate_missing_categories:
        print(
            "Manifest coverage/files incomplete; regenerating deterministic benchmark suite..."
        )
        generate_test_manifest()
        payload = _load_manifest(manifest_path)
        cases = payload.get("cases", [])
        coverage_errors = validate_manifest_cases(cases)
        file_errors = _manifest_file_errors(root, cases)
    if coverage_errors or file_errors:
        print("ERROR: benchmark manifest validation failed.")
        for msg in coverage_errors + file_errors:
            print(f" - {msg}")
        raise SystemExit(2)

    modes = [m.strip() for m in (args.mode_sequence or "").split(",") if m.strip()]
    if not modes:
        modes = [args.mode]

    deadline = (
        None
        if args.duration_minutes <= 0
        else (time.time() + (args.duration_minutes * 60))
    )
    final_code = 0
    round_no = 0

    while True:
        round_no += 1
        print(f"=== ROUND {round_no} START ===")
        for mode in modes:
            if deadline is not None and time.time() >= deadline:
                break
            print(f"=== MODE {mode} ===")
            code = run_benchmark(
                root=root,
                manifest_path=manifest_path,
                mode=mode,
                max_cases=args.max_cases,
                language_filter=args.language,
                max_parallel_pages=args.max_parallel_pages,
                free_only=args.free_only,
                skip_visual_judge=args.skip_visual_judge,
            )
            final_code = max(final_code, code)

        if deadline is None or time.time() >= deadline:
            break
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    raise SystemExit(final_code)


if __name__ == "__main__":
    main()
