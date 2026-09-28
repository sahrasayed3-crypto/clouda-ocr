# Session Baseline — ZCode Deep Engineering Session

Date: 2026-09-21
Mode: LOCAL CHANGES ONLY (no commit/push/remote actions unless explicitly authorized later)

## Git state at session start

- Branch: `main` (up to date with `origin/main`)
- HEAD: `18994155b2c26f1b082e444356ecd58e9a467eee` — "chore: prepare metadata for v0.2.0 release"
- Working tree: clean (`git status` empty; no staged, unstaged, or untracked files other than post-session additions)
- Tags: `v0.2.0` (single tag)
- Remotes: `origin https://github.com/sahrasayed3-crypto/clouda-ocr.git` (fetch+push)
- Latest commit: `1899415 chore: prepare metadata for v0.2.0 release`
- `git diff` / `git diff --cached`: empty at session start

## Release metadata at session start

- `pyproject.toml`: name `clouda-pdf`, version `0.2.0`, Python `>=3.11,<3.12`, Apache-2.0
- DOI: `10.5281/zenodo.22880981` (version), concept DOI `10.5281/zenodo.22880980`
- Separate repos (read-only, do not modify): `clouda-ocr-benchmark`, website repo

## Environment

- Local venv: `.venv311` (Python 3.11.9), pytest 9.1.1 installed
- Platform: Windows 10 (win32, Git Bash shell)

## Repository shape (counts at start)

| Area | Python LOC (excl. `_raqm` vendored stack, `__pycache__`) |
|---|---|
| pdfword | 22,572 |
| clouda_contracts | 1,343 |
| clouda_data | 30,113 |
| clouda_models | 195 |
| clouda_training | 11,657 |
| clouda_lab | 8,534 |
| Total production | ~74,000 |

- Test files: 185 `test_*.py` under `tests/`
- Vendored: `tools/` (poppler, tesseract, python toolchain — excluded from bandit)
- `clouda_data/factory/render/_raqm/` — vendored RAQM render stack (ruff E402 per-file ignore)

## Unrelated user changes

None present at session start (tree was clean).

## Session additions

All session work is confined to:
- `docs/engineering/*` (engineering documentation created by this session)
- test/source files modified for verified-defect fixes (each documented in `CODE_AUDIT_CURRENT.md` and the session log)

## Tooling config observed

- ruff: E4/E7/E9/F select, line 88, excludes backups/data/outputs/samples
- bandit: targets pdfword, tools, scripts; skips B608, B310, B615
- mypy: `mypy.ini` present
- pytest: `pytest.ini` present
- coverage: branch, source = pdfword, clouda_contracts, clouda_data, clouda_models, clouda_training
