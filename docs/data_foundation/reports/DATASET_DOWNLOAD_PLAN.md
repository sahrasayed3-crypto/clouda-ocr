# Dataset Download Plan

This stage supports only safe sample downloads and metadata capture. It does not start distortion generation, OCR training, AWS, account creation, or full dataset acquisition.

## Safety Rules

- No real download larger than 2 GB.
- No full dataset download.
- No download from unclear-license sources.
- No automatic website-term acceptance.
- No authentication, account creation, paid access, or form submission without explicit approval.
- Test download content must stay below 100 MB.
- Corrupt partial files go to `data/quarantine/downloads/`.

## Downloader Capabilities

- HTTP/HTTPS sample downloads.
- Hugging Face dataset sample assets through `resolve/main/...` paths.
- Git repository samples through raw HTTPS URLs or future explicit clone plans.
- Internet Archive sample assets through `archive.org/download/{identifier}/{path}` when permitted.
- Resume with `.part` files.
- Retry with exponential backoff.
- File-size limits and disk-space checks.
- SHA-256 checksums and duplicate detection.
- ZIP/TAR archive validation.
- Download manifests under `data/manifests/download_manifests/`.
- License metadata under each `data/downloads/{source_id}/` folder.

## Recommended First Sample

Use RASAM first because its license is explicit, its sample assets are tiny, and PAGE XML can be validated by the ingestion workflow.

Command:

```powershell
.\.venv\Scripts\python -m clouda_data.pipeline.cli download-dataset-sample rasam_dataset --max-bytes 104857600
```

Then verify:

```powershell
.\.venv\Scripts\python -m clouda_data.pipeline.cli verify-download rasam_dataset
.\.venv\Scripts\python -m clouda_data.pipeline.cli validate-source data/downloads/rasam_dataset/source_manifest.json
.\.venv\Scripts\python -m clouda_data.pipeline.cli ingest data/downloads/rasam_dataset/source_manifest.json --dry-run
```

Actual sample downloaded during this stage:

- `BULAC_MS_ARA_1977_0012.xml`: 27,478 bytes
- `LICENSE`: 11,357 bytes

Total: 38,835 bytes.

## First Controlled RASAM Batch

The first real acquisition batch used the dedicated RASAM batch commands:

```powershell
.\.venv\Scripts\python -m clouda_data.pipeline.cli plan-rasam-first-batch --pages 100 --max-bytes 1073741824
.\.venv\Scripts\python -m clouda_data.pipeline.cli download-rasam-first-batch --pages 100 --max-bytes 1073741824
.\.venv\Scripts\python -m clouda_data.pipeline.cli verify-rasam-first-batch
```

Pre-download gate:

- Planned pages: 100
- Planned files: 204
- Estimated download size: 48,846,110 bytes
- Estimated extracted size: 48,846,110 bytes
- Available disk at planning time: 141,057,257,472 bytes
- Authentication, form submission, paid access, and terms acceptance: not required

Actual result:

- Downloaded size: 48,846,110 bytes
- Extracted size: 48,846,110 bytes
- Valid pages: 88
- Rejected pages: 12, all for empty ground truth
- Source files remain under `data/downloads/rasam_dataset/first_batch/`
- Copy-based ingestion writes managed copies under `data/raw/`

## Future Download Order

1. RASAM tiny PAGE XML sample.
2. craneset public Hugging Face sample metadata and manually selected sample files.
3. SARD small Apache-licensed synthetic clean-layout sample after selecting exact file-level assets.
4. Humans in the Loop only after user manually approves form-based access.
5. KAFD only after explicit LDC license acquisition and user approval.
