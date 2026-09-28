# Clouda OCR — Repository Credibility Review

This document records the documentation review completed with the investor
due-diligence pack. It is a current-state review, not an assertion of product
traction, revenue, or external certification.

## Current identity and scope

The canonical project name is **Clouda OCR**. Version 0.2.1 is the current
software release. The older v0.2.0 release was originally published as
Clouda PDF and remains historical release evidence; the project does not
rewrite that history.

The repository supports trusted born-digital PDF-to-DOCX conversion, explicit
page routing, and surrounding data, evaluation, and training infrastructure.
It does not claim to ship a final trained production OCR model, an integrated
production OCR engine, a hosted API, or SaaS.

## Benchmark review

The current model-selection record is the separate 462-page v1.0 benchmark.
It has 10 candidates, five complete ranked runs, three partial unranked runs,
and two failed candidates. It ranks by Normalized Arabic CER and deliberately
excludes runtime, speed, and VRAM comparisons. Model selection is complete;
the project phase is selected-model development and training.

The 177-page v0.1.0 benchmark is historical. Its HunyuanOCR-1.5 result is
scoped to that earlier benchmark and is not the current selection result.

## Documentation decisions

The review retained only documentation that could be reconciled with the
current repository state:

- The investor due-diligence pack and evidence map now use the current v0.2.1
  release, current v1.0 benchmark, and current project phase.
- Historical material is explicitly labeled as historical rather than being
  presented as the current benchmark or test state.
- The current `README.md`, `CHANGELOG.md`, and `docs/TESTING.md` from main
  were retained during reconciliation, avoiding reintroduction of stale
  release and benchmark claims.
- The deployment guide's obsolete distributed-deployment reference was
  replaced with `docs/operations/DEPLOYMENT.md`.
- A committed local attachment path was replaced with a generic internal
  review reference.

## Claims deliberately excluded

This review does not assert customers, revenue, market demand, security
certification, production OCR accuracy, benchmark speed comparisons, or a
final production-model decision. Such claims require evidence beyond this
repository and must not be inferred from source code or benchmark rank.

## Verification guidance

For each release or merge candidate, use `.github/workflows/ci.yml` and the
corresponding GitHub Actions run as the CI authority. Use
`INVESTOR_TECHNICAL_DUE_DILIGENCE_SOURCES.md` to trace current product and
benchmark claims to their maintained sources.
