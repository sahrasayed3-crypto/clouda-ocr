"""Clouda Data Factory — synthetic Arabic OCR data generation.

Integrated Clouda OCR subsystem (``clouda_data.factory``). Deterministic
clean/degraded document generation: text/image/PDF ingest, Arabic RTL
rendering backends (WeasyPrint/Pango and Pillow/RAQM), atomic distortion
registry + composite scan simulation, readability QC, PNG/PDF exporters,
JSONL+CSV manifests with SHA-256 provenance, multiprocessing, and
manifest-driven resume.

Migrated from the standalone clouda-data-factory repository
(https://github.com/sahrasayed3-crypto/clouda-data-factory),
HEAD 4663f17c3416e970633574712d432b5b92cd5b9a. That standalone
repository is functionally superseded by this package and remains
untouched locally as an archive.
"""

__version__ = "0.1.0"
SEED_MODES = ("v1", "ocr_benchmark", "arabic_scan_factory")

DEFAULT_BASE_SEED = 20260831
LEGACY_SEED_OCR_BENCHMARK = 20260825
LEGACY_SEED_ARABIC_SCAN_FACTORY = 20260831
