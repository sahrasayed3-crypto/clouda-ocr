# Pseudo-Ground-Truth Policy

Automated output remains `candidate_ground_truth` or `pseudo_ground_truth`.

- GOLD requires recorded human verification or a separately authorized trusted reference.
- SILVER requires policy-v1 agreement thresholds, complete required-region coverage, strong image-grounded judgment, complete provenance, and no hallucination blocker.
- REVIEW covers disagreement, unreadable spans, missing regions, reading-order conflicts, or insufficient confidence/coverage.
- REJECTED covers failed rights, unusable input, hallucination, incompleteness, or policy violation.
- HOLDOUT is manual, evaluation-only, excluded from training, and immutable once frozen.

Policy `teacher-acceptance-v1` records all thresholds with each decision. Agreement normalization is for comparison only and never rewrites stored output.
