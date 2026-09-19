# Investor-Facing GitHub Audit — Clouda OCR

- Repository: https://github.com/sahrasayed3-crypto/clouda-ocr
- Audited SHA (`main`): `45d0078e551267535f85c79badd432fdc8514bef` (2026-09-19, merge of PR #1)
- Audit date: 2026-09-20
- Scope: presentation and documentation hardening only. No runtime behavior, benchmarks workspace, or product features were modified.

---

## Executive summary

### Strongest trust signals

1. **Honest model status.** The repository states plainly that no final trained OCR model exists, that scanned pages route to `pending_ocr_model`, and that the leading benchmark candidate (HunyuanOCR-1.5) is "the current leading candidate based on this specific 177-page benchmark" — not a product engine. This is rare discipline and it reads as credible.
2. **Fail-closed licensing architecture.** Separate evaluation / commercial-training / redistribution permission fields, `LOCAL_ONLY` / `NEEDS_REVIEW` fail-closed states, `DATA_LICENSES.md`, `THIRD_PARTY_NOTICES.md`, `UPSTREAM_COMPATIBILITY.md`, `NOTICE`, and an SBOM. Licensing is treated as an engineering property, not a footnote.
3. **Verifiable engineering evidence.** Green CI on `main` across Windows and Ubuntu (Python 3.11), 183 test files with ~1,895 test functions (static count), ruff/black/mypy gates, compileall gate, a repository security scan tool (`tools.validation.repository_scan`), and a metadata-only benchmark with SHA-256 canonical manifest hashes.
4. **Benchmark hygiene.** The 177-page benchmark is metadata-only in public, with provenance, rights classifications, exclusions (partial/failed runs are not ranked), metric definitions, and an explicit cross-GPU timing caveat that forbids unfair speed claims.
5. **Consistent merge discipline.** 17 feature branches merged via PRs with coherent naming (`feature/ocr-self-review`, `feature/e2e-release-hardening`, …) and a commit history that matches the documented roadmap.

### Biggest credibility risks

1. **Project identity confusion.** The README is titled "Clouda PDF", the repository is `clouda-ocr`, `NOTICE` says "Clouda OCR", the installed package is `clouda-pdf`, and the GitHub description says "OCR … framework". A reviewer spending two minutes cannot tell what the product is called.
2. **The README opens like an internal status report.** The first content block is a dense blockquote about merged subsystems and disabled defaults — a changelog voice, not a product voice.
3. **Stale historical test claim.** `docs/TESTING.md` presents "145 passed" (dated 2026-07-14) as the "Latest verified result" while the suite has since grown by an order of magnitude. Presented as *latest*, this understates the project and risks looking stale or cherry-picked.
4. **One leaked local path.** `docs/superpowers/plans/2026-09-09-pretraining-infrastructure-hardening.md` contains a personal machine path (a local AI-tool attachments directory under the developer's user profile) — scrubbed on the hardening branch.
5. **Empty GitHub topics and zero releases/tags**, despite version `0.2.0` in `pyproject.toml` and a green pipeline. The repository is invisible in topic search and offers no pinned, citable revision.

### Top 5 highest-impact improvements

1. Unify the public name as **Clouda OCR** in the README title and opening, and explain the `clouda-pdf` package name once (P1).
2. Rewrite the README opening into a one-line identity + short "why this exists" + status summary; move subsystem status detail below the fold (P1).
3. Fix the stale "Latest verified result" in `docs/TESTING.md` with clearly-labeled live and historical numbers (P1).
4. Scrub the leaked local path from the public plan doc (P1).
5. Owner actions outside the repo: set 8–12 accurate topics, cut a `v0.2.0` tag and GitHub Release, keep the CI badge green (P1, requires owner permission).

---

## First-impression audit

| Item | Status | Notes |
|---|---|---|
| Repo name | STRONG | `clouda-ocr` matches the GitHub description and website (cloudaocr.xyz). |
| Description | STRONG | Accurate, no hype, mentions benchmarking and RTL DOCX export. |
| README title | MISLEADING | "Clouda PDF" vs repo `clouda-ocr` vs `NOTICE` "Clouda OCR" vs package `clouda-pdf`. |
| README opening | WEAK | Blockquote of internal merge status; dense; reads like a changelog. |
| Hero message | WEAK | The strongest differentiators (Arabic-first trust gate, page routing, fail-closed licensing) are buried below the fold. |
| Badges | ACCEPTABLE | Only a "Build with Ona" badge; no CI badge. A green CI badge would be live execution evidence. |
| 30-second newcomer test | WEAK | A newcomer learns the name confusion before the product. Fixed by the README rewrite on the hardening branch. |
| Homepage | STRONG | https://cloudaocr.xyz/ resolves (HTTP 200). |
| Topics | MISSING | Zero topics set. |

## Claim verification table

| Claim | File / location | Type | Evidence found | Classification | Action |
|---|---|---|---|---|---|
| "No final trained OCR model exists" | README line 7–9 | feature availability | Consistent with `clouda_models` (metadata only), disabled-by-default OCR boundary, ROADMAP "Partially complete / disabled by default" | VERIFIED | Keep |
| Digital-text extraction → DOCX works today | README lines 23–29, 196–204 | feature availability | pdfword runtime, demo script, tests (`tests/test_arabic_fixtures.py`, conversion, DOCX validity), CI smoke | VERIFIED | Keep |
| "Benchmarking is complete" | README line 208 | benchmark | `benchmarks/ocr_arabic/RESULTS.md` — 6 complete runs over 177 pages, 2 excluded runs documented | VERIFIED_BUT_NEEDS_CONTEXT | Keep with pointer to scope: complete for this benchmark cycle |
| HunyuanOCR-1.5 Normalized Arabic CER 0.391497 on 177 pages | README line 209; `benchmarks/ocr_arabic/RESULTS.md` | benchmark | Matches RESULTS.md leaderboard and release.json manifest hash; caveats present | VERIFIED_BUT_NEEDS_CONTEXT | Keep; already scoped to "this specific benchmark" |
| Runtime comparisons across GPUs | `benchmarks/ocr_arabic/RESULTS.md` "Hardware caveat" | speed | AIN-7B ran on RTX PRO 6000 (95.6 GB), others on L4; caveat explicitly forbids speed claims | VERIFIED | Keep |
| "AMD/ROCm readiness is architectural and diagnostic only" | README line 214 | architecture | `docs/AMD_ROCM_ROADMAP.md` is a roadmap; no validated GPU inference | VERIFIED | Keep |
| "145 passed" as "Latest verified result" | `docs/TESTING.md` line 38 | reproducibility | Dated 2026-07-14; repo now has 183 test files / ~1,895 test functions | OUTDATED | Reframe as dated historical snapshot; add live static counts (done on hardening branch) |
| Apache-2.0 applies to original code only; third-party materials excluded | README lines 238–242; LICENSE; NOTICE | licensing | LICENSE is standard Apache-2.0; NOTICE lists OFL 1.1 bundled fonts; exclusions enumerated | VERIFIED | Keep |
| "Open-Source Scope" exclusions (weights, adapters, private recipes, etc.) | README lines 234–242 | commercial use | Consistent with ROADMAP "External decisions" and fail-closed docs | VERIFIED | Keep |
| Clouda Lab is loopback-only, no model downloads, bounded uploads | README lines 153–181 | security | Consistent with `docs/lab/`, tests under `tests/lab/` (10 MiB / 25-page bounds, no-persistence assertions) | VERIFIED | Keep |
| "User documents are never training data by default" | SECURITY.md | security | Consent boundary described; no runtime path admits documents to training | VERIFIED | Keep |
| Training experiment framework is CPU-only mock | README lines 75–77 | training | `clouda_training` mock adapter only; "real GPU training remains fail-closed" (ARCHITECTURE.md) | VERIFIED | Keep |
| Full install command `pip install -c constraints/py311.txt -e ".[server,...]"` | README lines 59–60 | hardware requirement / reproducibility | constraints/py311.txt and all extras exist in pyproject; CI runs the same command | VERIFIED | Keep |
| `.env.example` reference | README line 64 | feature availability | File exists at repo root | VERIFIED | Keep |

## README weaknesses

1. Title/identity mismatch (see above).
2. Opening blockquote is subsystem changelog, not a product statement.
3. Capabilities are listed before the problem; an investor reads "why" last.
4. No status matrix (Implemented / Experimental / Planned) above the fold.
5. No engineering-evidence summary (CI platforms, test scale, security tooling) — the repo's best verifiable proof points are absent from the README.
6. No CI badge; the only badge advertises a third-party AI development tool ("Build with Ona"), which is accurate but is the single most prominent trust element and it signals "agent-built" rather than "engineered".
7. The excellent licensing section is at the very end; fine, but the earlier "Open-Source Scope" heading duplicates part of it.

## Technical credibility strengths

- Cross-platform CI (windows-latest + ubuntu-latest, Python 3.11) with ruff, black, mypy, compileall, JS syntax check, import/CLI smoke tests, and `doctor --deep`; latest run on `main` is **success** (2026-09-19).
- 183 test files / ~1,895 test functions (static count) spanning contracts, routing, security, lab bounds, benchmark release validation, and fixtures.
- `SBOM.json` at root; `SECURITY.md` with concrete runtime protections (upload bounds, archive traversal/expansion checks, defusedxml, header-key worker API, security headers).
- Pinned GitHub Actions SHAs and `persist-credentials: false` — supply-chain hygiene beyond typical early-stage repos.
- Fail-closed storage roots (`StorageRoots`), feature-flagged OCR boundary, deterministic fixtures, checkpoint integrity checks.

## Benchmark credibility risks

Low. The benchmark section is the strongest part of the repo. Residual notes:

- The public leaderboard mixes GPU hardware across rows while ranking on accuracy only; the caveat is present and adequate.
- `models.csv` license statuses are `NOT_CONFIRMED_IN_LOCAL_EVIDENCE` / `NEEDS_REVIEW` — honest, but a future README could be misread as endorsing HunyuanOCR-1.5 for commercial use. Current wording ("Final production/runtime selection remains subject to … licensing …") prevents this. Keep it.
- Root README says "Benchmarking is complete." — accurate for this cycle; ensure future benchmark work is labeled as a new cycle rather than silently replacing numbers.

## Licensing clarity

STRONG. Apache-2.0 for original code only, explicit carve-outs, OFL 1.1 font notices, separate dataset/model/output-rights tracks, fail-closed rights states, and "does not claim ownership of external datasets or third-party OCR models". No change needed beyond keeping the wording intact.

## Repository hygiene

| Finding | File | Severity | Action |
|---|---|---|---|
| Personal local machine path (an AI-tool attachments directory under the developer's user profile) | `docs/superpowers/plans/2026-09-09-pretraining-infrastructure-hardening.md` | P1 | Scrub path; keep content (done on hardening branch) |
| Stale "145 passed" labeled as latest result | `docs/TESTING.md` | P1 | Reframe as dated historical snapshot + live static counts (done) |
| AI-session plan/spec docs public under `docs/superpowers/` | `docs/superpowers/` | P2 | Defensible as development-history evidence of disciplined planning; keep, but scrub local paths and consider a one-line README inside the folder explaining its purpose |
| "Build with Ona" badge | README line 3 | P2 | Owner decision; removed on hardening branch in favor of a live CI badge (owner should verify the badge renders green) |
| No releases or tags | repo-level | P1 | Owner: tag `v0.2.0`, publish a GitHub Release from CHANGELOG |
| CHANGELOG has only "Unreleased" | `CHANGELOG.md` | P2 | Cut a dated 0.2.0 entry when tagging |
| No CI badge in README | README | P1 | Added on hardening branch |
| Root-level `openrouter_api_key.example.txt` | repo root | ACCEPTABLE | Placeholder only, no secret; naming could be clearer but harmless |

Secrets sweep: no hardcoded credentials found. Local-path sweep: one occurrence (above). Profanity/embarrassing artifacts: none. Dead links: all README-linked files exist (verified). Website link resolves (HTTP 200).

## Investor narrative gaps

The repo answers "what is built and how well" well. It does not answer (no evidence exists in-repo; do not invent):

1. **Who is the target user?** Researchers/archives/publishers are implied; no primary-user evidence.
2. **Evidence of demand.** None public — label as a gap, not a weakness of the code.
3. **Commercialization path.** "May be licensed, hosted, or distributed separately" implies a dual-track model but no product/pricing shape.
4. **Moat.** The honest answer in-repo is the fail-closed rights pipeline + evaluation infrastructure; this is architectural, not yet data- or model-based.
5. **Team and capital ask.** GPU access is named as the binding constraint — that is effectively the funding narrative and it is stated plainly. Good.

These gaps are acceptable at this stage as long as nothing public pretends to fill them.

## Exact recommended changes (P0 / P1 / P2)

**P0 — credibility risk**
- None rising to P0. No misleading benchmark, licensing, or capability claim was found. The leaked local path is the closest item and is treated as P1.

**P1 — major investor-facing improvements (implemented on `docs/investor-github-hardening`)**
1. README: rename title to Clouda OCR, one-line identity, explain the `clouda-pdf` package name once, product-first opening, added "What exists today" status matrix and "Engineering evidence" section, added CI badge, removed the Ona badge, tightened structure without changing any factual claim.
2. `docs/TESTING.md`: relabel the 2026-07-14 result as a dated historical snapshot; add current static counts (183 test files, ~1,895 test functions) with instruction to run the suite for live numbers.
3. `docs/superpowers/plans/2026-09-09-pretraining-infrastructure-hardening.md`: scrub personal local path.
4. Add this audit report (`docs/investor-github-audit.md`).

**P2 — polish (not applied; owner decisions)**
1. GitHub settings (require explicit permission): add topics `arabic-ocr, ocr, pdf, docx, document-processing, rtl, arabic-nlp, vlm, benchmark, self-hosted, apache-2.0, python`; verify social preview.
2. Tag `v0.2.0` and publish a GitHub Release; add a dated CHANGELOG entry.
3. Decide whether to keep `docs/superpowers/` public long-term (recommend keeping with a purpose note) or archiving after the next cycle.
4. `CHANGELOG.md`: consider a short "0.1 → 0.2" history entry.

---

## Verification performed during this audit

- Live GitHub metadata fetched (description, homepage, topics, license, CI runs, releases: none, issues: 0, stars: 1).
- Latest Actions run on `main` (45d0078): **success**; earlier intermediate commits show the CI-repair trail documented in PR #1.
- All files referenced by README verified present; install/test/demo commands match CI steps and CI-verified extras in `pyproject.toml`.
- Website https://cloudaocr.xyz/ returns HTTP 200.
- Static test counts computed from the working tree (183 files / 1,895 `def test_` occurrences). Live pass counts intentionally not quoted anywhere.
- Benchmark numbers cross-checked against `benchmarks/ocr_arabic/RESULTS.md`, `models.csv`, and `release.json` (canonical manifest SHA-256 present).
