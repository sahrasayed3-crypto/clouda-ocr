# Methodology

## Benchmark definition

The canonical cohort contains 177 distorted pages derived from 100 clean source
records. Every rankable model was evaluated on the same 177 `distorted_id`
values in `benchmark_manifest.jsonl`. The SHA-256 of that exact manifest is:

`2a499ed0268a5c583a1b174ef241e05affb9c4ba757593eb9ee3fc6f3e4fb893`

The public release contains metadata and hashes only. Image assets,
ground-truth text, raw predictions, runtime state, and forensic chronology are
retained privately.

## Metrics

The restored benchmark implementation computes Levenshtein edit counts with
deterministic tie-breaking.

- CER is character edit distance divided by the number of characters in the
  raw reference text.
- WER is word-token edit distance divided by the number of whitespace-split
  reference words.
- Normalized Arabic CER applies deterministic Arabic normalization to both
  reference and prediction, then divides normalized character edit distance by
  normalized reference length.

Arabic normalization performs NFC Unicode normalization, maps Arabic alef
variants (`أ`, `إ`, `آ`, `ٱ`) to `ا`, maps `ى` to `ي`, removes tatweel and
Arabic diacritics, collapses whitespace, and trims leading/trailing whitespace.
It does not overwrite the retained raw strings.

The public leaderboard values are arithmetic means of per-page metrics across
successful rows for complete runs. Insertions can make edit distance larger
than reference length, so CER, WER, and Normalized Arabic CER may legitimately
exceed 1.0.

Normalized Arabic CER is the primary ranking metric and lower is better.

## Result selection

A run is rankable only when it has a completion record covering all 177 common
pages. Six runs meet that rule. `dots.mocr` has 30 rows but no completion
summary and is `PARTIAL`; PaddleOCR-VL-1.6 failed its smoke test before a full
run and is `FAILED_SMOKE`. Neither appears in the ranked leaderboard.

Raw result evidence is identified by stable `raw-evidence:` identifiers in
`results.csv`; machine-local paths are intentionally excluded.

## Hardware interpretation

Accuracy metrics are comparable across the common page cohort. Runtime is not
a controlled cross-GPU comparison. AIN-7B used a 95.6 GB RTX PRO 6000
Blackwell system, while the other complete runs used NVIDIA L4 hardware.

## Provenance and rights

`source_manifest.csv` joins each of the 100 clean-source identifiers to its
precise dataset identifier, source item, hashes, derivative count, and separate
permission dimensions. `source_summary.csv` aggregates those records without
collapsing the 14 precise dataset identifiers.

Rights decisions are fail-closed. Evaluation permission and commercial
training permission do not imply permission to redistribute source or derived
assets. Model licenses do not imply permission to redistribute raw outputs or
use those outputs as training labels. Private evidence controls classifications
when package-local labels are incomplete or conflicting.

No dataset or benchmark asset is distributed by this metadata release.
