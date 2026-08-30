# Distortion CLI

Status: **complete**.

Major commands are `render`, `render-resume`, `render-validate`,
`distort-batch`, `distort-preview`, `distort-status`, `distort-resume`,
`distort-validate`, evaluation commands, lifecycle commands, and profile
inspection/validation.

New runs default to at most 100 pages. Larger runs require
`--allow-large-run`. A seed is mandatory. Outputs must resolve beneath the
configured state roots. Existing outputs are rejected unless an explicit,
validated conflict policy is selected.

Use `python -m clouda_data.pipeline.cli COMMAND --help` for exact arguments.

