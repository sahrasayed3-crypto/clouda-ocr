# Roadmap

## Next 0–2 months

- Maintain the direct-text conversion path and metadata contract.
- Secure suitable GPU compute for runtime integration and continued independent evaluation of OCR/VLM candidates, including resuming the paused 462-page held-out expansion.
- Plan adaptation and integration work for candidate OCR/VLM models while
  preserving the fail-closed licensing and provenance process.
- Define application-route acceptance criteria for Arabic, English,
  mixed-direction, old, and degraded documents.

## Months 2–4

- Prepare model-agnostic runtime integration for whichever candidate satisfies the final evaluation, licensing, and deployment requirements; adapt or train a dedicated model only if route-level evaluation demonstrates a gap that existing models do not close.
- Measure application-route accuracy, latency, memory, and failure behavior on
  representative hardware.
- Decide whether an optional CPU and/or AMD-compatible deployment path is viable.

## Months 4–6

- Integrate only the selected, validated engine as an optional component.
- Publish reproducible evaluation results and hardware/software details.
- Improve editable DOCX structure while retaining page provenance and manual-review routes.

The published 177-page benchmark v0.1.0 is complete; the larger 462-page
held-out evaluation is in progress and currently paused pending additional
compute capacity. On the published 177-page benchmark, HunyuanOCR-1.5 ranked
first by Normalized Arabic CER (0.391497), a benchmark-specific historical
result. Production/runtime selection remains open pending the expanded
evaluation, architecture, licensing, deployment constraints, integration, and
subsequent validation. No roadmap item claims that a final trained and
integrated production model or validated GPU execution path already exists.
