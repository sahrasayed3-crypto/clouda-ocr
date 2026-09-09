"""Distortion engines.

- `atomic`: System A's registry of 20 atomic ops (17 original + 3 ported
  from B), parameterized by YAML severity ladders.
- `scan_composite`: System B's fixed-order composite scan simulation,
  used for legacy scan-family profiles and as a fast whole-page backend.
- `qc`: System A's readability guard (contrast/sharpness/ink-ratio ratios
  against the clean page).
"""
