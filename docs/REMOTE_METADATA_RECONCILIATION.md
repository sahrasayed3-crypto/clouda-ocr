# Remote Metadata Reconciliation — Pending Manual Actions

Prepared locally: 2026-09-21. **Nothing remote was changed in this pass.**
Every item below requires an explicit, approved remote action by the project
owner before the unified "Clouda OCR" branding is published.

## 1. GitHub release v0.2.0 display title (optional)

- Current state: the release is titled **"Clouda PDF v0.2.0"**.
- Option A (recommended): keep the historical title as-is and let `v0.2.1`
  carry the canonical "Clouda OCR" name. The historical title is evidence.
- Option B: edit only the release *display title/metadata* to
  "Clouda OCR v0.2.0 (released as Clouda PDF)" — a metadata edit, not a retag.
  Do **not** retag or replace source archives.

## 2. Zenodo software record 22880981 (version DOI 10.5281/zenodo.22880981)

- Current state: record title is **"Clouda PDF"**.
- If the record is editable on Zenodo: retitle to **"Clouda OCR"** and add a
  version-history note: "v0.2.0 was originally published as 'Clouda PDF'".
- The concept DOI `10.5281/zenodo.22880980` automatically follows the record.
- Required when: the website/README label the artifact "Clouda OCR" while the
  remote record still says "Clouda PDF" (acceptable transitional state; the
  website keeps the historical note until this is done).

## 3. Future Zenodo deposition for v0.2.1

- The local `.zenodo.json` already uses title "Clouda OCR" and version-ready
  metadata. Upload the v0.2.1 deposition with this metadata; the new version
  record joins concept DOI `10.5281/zenodo.22880980`.

## 4. Release description text

- The v0.2.0 GitHub release notes may still describe the runtime as
  "Clouda PDF". Either leave (historical) or append a short note pointing to
  the brand migration. Do not silently rewrite release notes.

## 5. Website links after reconciliation

- Website software links already point to the stable URLs
  (`…/releases/tag/v0.2.0`, `doi.org/10.5281/zenodo.22880981`); they do not
  embed the old title, so **no website link changes are required** by remote
  reconciliation. Only the one-line historical note on the website About page
  can be removed once the remote record title is updated.

## 6. Benchmark record — no action

- Zenodo record 22859930 ("Clouda OCR Arabic OCR Benchmark v0.1.0",
  DOI `10.5281/zenodo.22859930`) is a separate artifact and is **not** part of
  this reconciliation. Never merge it with the software record.
