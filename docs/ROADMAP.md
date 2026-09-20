# Roadmap

## Next 0–2 months

- Maintain the direct-text conversion path and metadata contract.
- Secure suitable GPU compute for runtime integration and continued independent evaluation of OCR/VLM candidates.
- Plan adaptation and integration work for the current leading candidate while
  preserving the fail-closed licensing and provenance process.
- Define application-route acceptance criteria for Arabic, English,
  mixed-direction, old, and degraded documents.

## Months 2–4

- Integrate the leading candidate behind the model-agnostic interface; adapt or train a dedicated model only if route-level evaluation demonstrates a gap that existing models do not close.
- Measure application-route accuracy, latency, memory, and failure behavior on
  representative hardware.
- Decide whether an optional CPU and/or AMD-compatible deployment path is viable.

## Months 4–6

- Integrate only the selected, validated engine as an optional component.
- Publish reproducible evaluation results and hardware/software details.
- Improve editable DOCX structure while retaining page provenance and manual-review routes.

The 177-page public benchmark is complete. HunyuanOCR-1.5 is the current leading
candidate based on this specific benchmark (Normalized Arabic CER 0.391497).
Final production/runtime selection remains subject to architecture, licensing,
deployment constraints, integration, and subsequent validation. No roadmap item
claims that a final trained and integrated production model or validated GPU
execution path already exists.
