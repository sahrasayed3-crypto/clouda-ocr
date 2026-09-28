# Roadmap

## Next 0–2 months

- Maintain the direct-text conversion path and metadata contract.
- Continue selected-model development and training work following the completed model-selection benchmark v1.0, preserving the fail-closed licensing and provenance process.
- Plan adaptation and integration work for candidate OCR/VLM models while
  preserving the fail-closed licensing and provenance process.
- Define application-route acceptance criteria for Arabic, English,
  mixed-direction, old, and degraded documents.

## Months 2–4

- Prepare model-agnostic runtime integration for the selected model, satisfying the applicable licensing, and deployment requirements; adapt or train further only if route-level evaluation demonstrates a gap that existing models do not close.
- Measure application-route accuracy, latency, memory, and failure behavior on
  representative hardware.
- Decide whether an optional CPU and/or AMD-compatible deployment path is viable.

## Months 4–6

- Integrate only the selected, validated engine as an optional component.
- Publish reproducible evaluation results and hardware/software details.
- Improve editable DOCX structure while retaining page provenance and manual-review routes.

The model-selection benchmark v1.0 is complete and published: 462 Arabic
document pages, 10 candidate OCR/VLM models, 5 complete ranked runs, 3 partial
runs, and 2 failed runs, ranked by Normalized Arabic CER with 7,198 / 7,198
release checksums verified
([clouda-ocr-model-selection-benchmark](https://github.com/sahrasayed3-crypto/clouda-ocr-model-selection-benchmark)).
The earlier 177-page benchmark v0.1.0 (historical) remains published as a
separate record; on it, HunyuanOCR-1.5 ranked first by Normalized Arabic CER
(0.391497), a benchmark-specific historical result. Model selection is
complete, based on the published v1.0 results; the current project phase is
model development and training. No roadmap item claims that a final trained and
integrated production model or validated GPU execution path already exists.
