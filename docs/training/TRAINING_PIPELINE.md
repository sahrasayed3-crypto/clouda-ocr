# Training pipeline

1. Select catalog records.
2. Enforce `commercial_training` through the license gate.
3. Resolve `dataset://` roots.
4. inventory and deduplicate records by cryptographic hash, with a perceptual
   duplicate hook.
5. Split by source document using a fixed seed.
6. Generate task-specific examples with provenance.
7. Produce an experiment plan and checkpoint metadata.

The current implementation stops at planning. It requires no GPU and never
downloads a model or launches training. Templates cover 100-page smoke,
1,000-page pilot, available-approved full run, text OCR, bounding boxes, and
Markdown/layout tasks.
