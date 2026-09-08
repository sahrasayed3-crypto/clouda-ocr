"""Pre-training dataset preparation infrastructure.

This layer turns registered raw sources into a deterministic, traceable,
leakage-safe, training-ready dataset:

    scan -> hash -> normalize -> validate -> dedupe -> split -> manifest
    -> export -> (optional) Clouda Data Factory handoff.

It never downloads data, never depends on a trained OCR model, and never
mutates source files.
"""
