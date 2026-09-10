"""Clouda Lab CLI — minimal backend-exercise commands.

Follows the existing ``clouda-training`` CLI conventions (argparse
subcommands, ``--json`` machine-readable flag, ``Path`` args). These commands
exercise the backend services only; there is no web/UI layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .active_learning import recommend_next_batch
from .batch_analysis import analyze_batch, export_batch_csv
from .dataset_selection import (
    SelectionCriteria,
    select_samples,
    validate_derived_manifest_for_training,
    write_selection_manifest,
)
from .error_analysis import analyze_sample
from .failure_analysis import SampleMetrics, compare_failure
from .hard_examples import select_hard_examples
from .io import export_json, load_samples_jsonl


def _load_scores(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in payload.items()}


def _criteria_from_args(args: argparse.Namespace) -> SelectionCriteria:
    return SelectionCriteria(
        dataset_id=args.dataset_id,
        split=args.split,
        document_type=args.document_type,
        profile=args.profile,
        distortion=args.distortion,
        source=args.source,
        cer_min=args.cer_min,
        cer_max=args.cer_max,
        random_sample_size=args.random_n,
        top_n=args.top_n,
        top_n_count=args.top_n_count,
        limit=args.limit,
    )


def _cmd_analysis_page(args: argparse.Namespace) -> int:
    samples = load_samples_jsonl(args.input)
    chosen = [s for s in samples if s.sample_id == args.sample_id]
    if not chosen:
        print(f"sample not found: {args.sample_id}", file=sys.stderr)
        return 1
    analysis = analyze_sample(chosen[0])
    payload = analysis.to_dict()
    if args.output:
        export_json(payload, args.output)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_analysis_batch(args: argparse.Namespace) -> int:
    samples = load_samples_jsonl(args.input)
    report = analyze_batch(samples, worst_n=args.top, best_n=args.top)
    if args.output:
        export_json(report.to_dict(), args.output)
        if args.csv:
            export_batch_csv(report, args.csv)
    else:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_analysis_compare(args: argparse.Namespace) -> int:
    def _metrics(path: str) -> list[SampleMetrics]:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("samples", [])
        return [
            SampleMetrics(
                sample_id=str(r["sample_id"]),
                cer=float(r.get("cer", 0.0)),
                wer=float(r.get("wer", 0.0)),
                ncer=float(r.get("ncer", 0.0)),
                error_types=r.get("error_type_counts"),
            )
            for r in records
        ]

    report = compare_failure(
        _metrics(args.baseline),
        _metrics(args.candidate),
        baseline_id=args.baseline_label or "baseline",
        candidate_id=args.candidate_label or "candidate",
    )
    payload = {"summary": None, "report": report.to_dict()}
    from .failure_analysis import summarize_comparison

    payload["summary"] = summarize_comparison(report)
    if args.output:
        export_json(payload, args.output)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_dataset_select(args: argparse.Namespace) -> int:
    criteria = _criteria_from_args(args)
    result = select_samples(
        args.manifest, criteria, seed=args.seed, scores=_load_scores(args.scores)
    )
    if args.output:
        write_selection_manifest(result, args.output)
        summary = result.to_dict()
        summary["derived_manifest"] = str(args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _cmd_dataset_validate(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            validate_derived_manifest_for_training(args.manifest),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_hard_examples(args: argparse.Namespace) -> int:
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("samples", [])
    selected = select_hard_examples(
        rows,
        top_n=args.top_n,
        percentile=args.percentile,
        min_score=args.min_score,
    )
    payload = [item.to_dict() for item in selected]
    if args.output:
        export_json(payload, args.output)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_recommend_next(args: argparse.Namespace) -> int:
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("samples", [])
    history = []
    if args.history:
        history_payload = json.loads(Path(args.history).read_text(encoding="utf-8"))
        history = list(history_payload.get("used_sample_ids", []))
    recommendation = recommend_next_batch(
        rows,
        strategy=args.strategy,
        batch_size=args.batch_size,
        seed=args.seed,
        history=history,
    )
    payload = recommendation.to_dict()
    if args.output:
        export_json(payload, args.output)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_distortion_experiment(args: argparse.Namespace) -> int:
    from .distortion_experiments import (
        plan_distortion_experiment,
        run_distortion_experiment,
    )

    plan = plan_distortion_experiment(
        source_manifest=args.manifest,
        source_sample_ids=args.samples.split(","),
        profile=args.profile,
        distortions=args.distortions.split(",") if args.distortions else None,
        seed=args.seed,
        variants_per_sample=args.variants,
    )
    if args.output:
        # Plan-only by design in the CLI unless --run-images is provided;
        # generation needs source image paths (sample_id=path pairs).
        images: dict[str, str] = {}
        if args.run_images:
            for pair in args.run_images.split(","):
                sample_id, _, path = pair.partition("=")
                images[sample_id] = path
        result = run_distortion_experiment(plan, source_images=images, output_dir=args.output)
        print(json.dumps({"experiment_id": result["experiment_id"],
                          "rows": len(result["rows"]),
                          "output_dir": result["output_dir"]},
                         ensure_ascii=False, indent=2))
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clouda-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    page = sub.add_parser("analysis-page", help="Analyze one OCR sample.")
    page.add_argument("--input", required=True, type=Path)
    page.add_argument("--sample-id", required=True)
    page.add_argument("--output", type=Path)
    page.set_defaults(func=_cmd_analysis_page)

    batch = sub.add_parser("analysis-batch", help="Batch analysis over samples.")
    batch.add_argument("--input", required=True, type=Path)
    batch.add_argument("--output", type=Path)
    batch.add_argument("--csv", type=Path)
    batch.add_argument("--top", type=int, default=10)
    batch.set_defaults(func=_cmd_analysis_batch)

    compare = sub.add_parser(
        "analysis-compare", help="Compare baseline vs candidate metrics."
    )
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--baseline-label", default="baseline")
    compare.add_argument("--candidate-label", default="candidate")
    compare.add_argument("--output", type=Path)
    compare.set_defaults(func=_cmd_analysis_compare)

    select = sub.add_parser("dataset-select", help="Select a reproducible subset.")
    select.add_argument("--manifest", required=True, type=Path)
    select.add_argument("--output", type=Path, help="Derived manifest path")
    select.add_argument("--seed", type=int, default=20260723)
    select.add_argument("--dataset-id")
    select.add_argument("--split")
    select.add_argument("--document-type")
    select.add_argument("--profile")
    select.add_argument("--distortion")
    select.add_argument("--source")
    select.add_argument("--cer-min", type=float)
    select.add_argument("--cer-max", type=float)
    select.add_argument("--random-n", type=int)
    select.add_argument("--top-n", choices=["hardest", "easiest"])
    select.add_argument("--top-n-count", type=int)
    select.add_argument("--limit", type=int)
    select.add_argument("--scores", type=Path, help="JSON {sample_id: cer}")
    select.set_defaults(func=_cmd_dataset_select)

    validate = sub.add_parser(
        "dataset-validate", help="Validate a derived manifest for training safety."
    )
    validate.add_argument("--manifest", required=True, type=Path)
    validate.set_defaults(func=_cmd_dataset_validate)

    hard = sub.add_parser("dataset-hard-examples", help="Rank hard examples.")
    hard.add_argument("--input", required=True, type=Path)
    hard.add_argument("--output", type=Path)
    hard.add_argument("--top-n", type=int)
    hard.add_argument("--percentile", type=float)
    hard.add_argument("--min-score", type=float)
    hard.set_defaults(func=_cmd_hard_examples)

    recommend = sub.add_parser(
        "dataset-recommend-next", help="Recommend the next training batch."
    )
    recommend.add_argument("--input", required=True, type=Path)
    recommend.add_argument("--output", type=Path)
    recommend.add_argument("--strategy", default="balanced_hard")
    recommend.add_argument("--batch-size", type=int, default=32)
    recommend.add_argument("--seed", type=int, default=20260723)
    recommend.add_argument("--history", type=Path)
    recommend.set_defaults(func=_cmd_recommend_next)

    distortion = sub.add_parser(
        "distortion-experiment", help="Plan/run a tiny distortion experiment."
    )
    distortion.add_argument("--manifest", required=True, type=Path)
    distortion.add_argument("--samples", required=True, help="comma-separated ids")
    distortion.add_argument("--profile")
    distortion.add_argument("--distortions", help="comma-separated atomic names")
    distortion.add_argument("--variants", type=int, default=1)
    distortion.add_argument("--seed", type=int, default=20260831)
    distortion.add_argument("--run-images", help="sample_id=path pairs for generation")
    distortion.add_argument("--output", type=Path)
    distortion.set_defaults(func=_cmd_distortion_experiment)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
