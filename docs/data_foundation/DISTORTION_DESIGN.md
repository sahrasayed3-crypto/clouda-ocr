# Distortion Design

Distortions are modular operations with:

- Name
- Version
- Parameters
- Random seed
- Probability
- Severity
- Input requirements
- Output metadata
- Deterministic replay support
- Validation hooks

Supported architecture targets include full page, regions, edges, binding areas, background only, text only, low-priority regions, and high-priority regions.

The current implementation is metadata-only. Real image transformations will be added later and must never modify reference text.

