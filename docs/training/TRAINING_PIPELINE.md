# Training pipeline

1. Select catalog records.
2. Enforce `commercial_training` through the license gate.
3. Resolve `dataset://` roots.
4. inventory and deduplicate records by cryptographic hash, with a perceptual
   duplicate hook.
5. Split by source document using a fixed seed.
6. Generate task-specific examples with provenance.
7. Produce an experiment plan and checkpoint metadata.

The pipeline runs offline and requires no GPU. Steps 1–7 (planning) are
fully implemented; nothing here downloads a model. Separately, the
experiment runtime can execute a genuine optimization loop behind explicit
opt-in (see `docs/training/REAL_TRAINING_RUNTIME.md`) — validated with a
tiny synthetic trainer only; no real OCR training has been run. Templates
cover 100-page smoke, 1,000-page pilot, available-approved full run, text
OCR, bounding boxes, and Markdown/layout tasks.
