# Release Plan — Clouda OCR v0.2.1 (branding correction)

Status: **prepared locally, not published**. No tag, release, or Zenodo upload
was created. Version bumped locally in `pyproject.toml` / `CITATION.cff` to
`0.2.1` and documented in `CHANGELOG.md` as an unreleased entry.

## Scope

Branding/metadata correction only. **No functional OCR claim, no trained-model
claim, no accuracy claim.** The changelog entry must say exactly that.

## Checklist

1. **Brand migration** — `docs/BRAND_MIGRATION.md` merged/reviewed; canonical
   name `Clouda OCR` everywhere user-facing.
2. **Package/distribution** — distribution name is `clouda-ocr`
   (was `clouda-pdf`). Verify:
   - `python -m build` produces `clouda_ocr-0.2.1-*.whl` and
     `clouda_ocr-0.2.1.tar.gz`;
   - `pip install dist/clouda_ocr-0.2.1-*.whl` installs cleanly in a fresh venv;
   - import packages unchanged (`pdfword`, `clouda_data`, …);
   - console scripts unchanged (`clouda-data`, `clouda-lab`,
     `clouda-training`, `clouda-quality`) and `--help` output shows no
     stale `clouda-pdf` text.
3. **Compatibility** — no alias needed for a `clouda-pdf` CLI (none existed).
   Document in the release notes that `pip install clouda-pdf` remains the
   historical v0.2.0 distribution.
4. **Metadata** — `CITATION.cff` (title/version), `.zenodo.json` (title),
   `CHANGELOG.md` finalized; creator and ORCID unchanged.
5. **Website** — canonical naming live; the single historical note
   ("v0.2.0 was originally published under the release title 'Clouda PDF'")
   remains until remote metadata is reconciled
   (`docs/REMOTE_METADATA_RECONCILIATION.md`).
6. **Claims guard** — release notes must NOT state:
   - any OCR accuracy improvement (none was measured);
   - that a final trained OCR model exists;
   - that scanned-page OCR is production-ready;
   - that a hosted API/SaaS exists.
7. **Verification before tagging** — full local suite green
   (pytest, ruff, black --check, mypy, compileall, build; pip-audit/bandit if
   available) and the repository scan shows no unintended `clouda-pdf`
   references outside the documented compatibility/history list.
8. **Publish** — tag `v0.2.1`, GitHub release "Clouda OCR v0.2.1", Zenodo
   deposition. Only after owner approval.

## Open decisions for the owner

- Whether to also retitle the historical v0.2.0 GitHub release display title
  (recommended: leave as historical evidence).
- Whether to retitle the Zenodo record 22880981 now or with the v0.2.1 upload.
- Whether to update the two provenance labels
  (`clouda_data/distortion/workflow.py`, `clouda_data/training_data/sharding.py`)
  at this release or keep them as lineage.
