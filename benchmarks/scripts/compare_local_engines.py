from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_results(records: Iterable[dict]) -> dict:
    rows = list(records)
    by_category_engine: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error") or row.get("cer") is None:
            continue
        by_category_engine[(row["category"], row["engine"])].append(row)
        by_category[row["category"]].append(row)

    engine_summary = []
    for (category, engine), group in sorted(by_category_engine.items()):
        cer_values = [float(row["cer"]) for row in group]
        wer_values = [float(row["wer"]) for row in group]
        elapsed = [float(row["elapsed_seconds"]) for row in group]
        accepted = [row for row in group if row.get("decision") == "accepted"]
        manual = [row for row in group if row.get("decision") == "manual_review"]
        engine_summary.append(
            {
                "category": category,
                "engine": engine,
                "samples": len(group),
                "avg_cer": statistics.fmean(cer_values),
                "median_cer": statistics.median(cer_values),
                "p90_cer": percentile(cer_values, 0.9),
                "avg_wer": statistics.fmean(wer_values),
                "median_wer": statistics.median(wer_values),
                "p90_wer": percentile(wer_values, 0.9),
                "avg_elapsed_seconds": statistics.fmean(elapsed),
                "accepted_rate": len(accepted) / len(group) if group else 0.0,
                "manual_review_rate": len(manual) / len(group) if group else 0.0,
                "failures": 0,
            }
        )

    best_by_category = {}
    for category, group in by_category.items():
        candidates = [
            row
            for row in group
            if row.get("decision") == "accepted" and row.get("cer") is not None
        ]
        if not candidates:
            candidates = [row for row in group if row.get("cer") is not None]
        if candidates:
            best = min(
                candidates,
                key=lambda row: (
                    float(row["cer"]),
                    float(row["wer"]),
                    float(row["elapsed_seconds"]),
                ),
            )
            best_by_category[category] = {
                "engine": best["engine"],
                "cer": best["cer"],
                "wer": best["wer"],
                "elapsed_seconds": best["elapsed_seconds"],
            }

    return {
        "engine_summary": engine_summary,
        "best_by_category": best_by_category,
    }


def write_summary_files(records_path: str | Path, output_dir: str | Path) -> dict:
    records = json.loads(Path(records_path).read_text(encoding="utf-8"))
    summary = summarize_results(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "engine_summary.json"
    csv_path = output / "engine_summary.csv"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = [
        "category",
        "engine",
        "samples",
        "avg_cer",
        "median_cer",
        "p90_cer",
        "avg_wer",
        "median_wer",
        "p90_wer",
        "avg_elapsed_seconds",
        "accepted_rate",
        "manual_review_rate",
        "failures",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["engine_summary"])
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "records", nargs="?", default="benchmarks/results/local_benchmark_results.json"
    )
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()
    print(
        json.dumps(
            write_summary_files(args.records, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
