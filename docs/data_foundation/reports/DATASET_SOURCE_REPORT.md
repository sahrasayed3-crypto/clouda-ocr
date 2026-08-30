# Dataset Source Report

Verification date: 2026-07-23.

This report summarizes candidate sources for future Arabic OCR distortion-data work. A source is not treated as approved unless its license was explicit on an official source page or repository record. Full dataset downloads remain disabled.

## Recommended First Sample

Recommended first sample: RASAM Dataset.

Reason: it has an explicit Apache-2.0 license, PAGE XML ground truth, historical Arabic manuscript coverage, layout annotations, and a tiny safe sample path already verified by the downloader.

Downloaded test sample:

- `data/downloads/rasam_dataset/BULAC_MS_ARA_1977_0012.xml`
- `data/downloads/rasam_dataset/LICENSE`

Total downloaded test content: 38,835 bytes.

The generated ingestion manifest is:

- `data/downloads/rasam_dataset/source_manifest.json`

## First Controlled RASAM Batch

The first controlled real-data batch downloaded 100 selected RASAM pages from official RASAM GitHub PAGE XML and BULAC/BiNA IIIF image endpoints.

- Download location: `data/downloads/rasam_dataset/first_batch/`
- Managed ingestion location: `data/raw/pages/` and `data/raw/ground_truth/`; the ingested PAGE XML files carry both exact text and layout annotations.
- Batch manifest: `data/manifests/rasam_first_batch_manifest.json`
- Rejections: `data/manifests/rasam_first_batch_rejections.jsonl`
- Batch report: `docs/RASAM_FIRST_BATCH_REPORT.md`
- Downloaded size: 48,846,110 bytes
- Pages downloaded: 100
- Valid pages copied into ingestion manifest: 88
- Rejected pages: 12, all rejected for `empty_ground_truth`

RASAM repository annotations and metadata are Apache-2.0. External BULAC IIIF page images are not assumed to be Apache-2.0; the batch uses only selected IIIF images whose source manuscript metadata reports public-domain rights.

## Verified Candidates

### RASAM Dataset

- Classification: `approved_with_conditions`
- License: Apache-2.0
- Commercial use: allowed
- Formats: IIIF JPG images, PAGE XML, TSV
- Ground truth: PAGE XML line transcription
- Best use: historical Arabic layout/HTR sample and PAGE XML ingestion validation
- Limitation: handwritten historical manuscripts, not modern printed documents; external images require per-item BULAC/BiNA rights verification
- Official source: https://github.com/calfa-co/rasam-dataset

### SARD Synthetic Arabic Recognition Dataset

- Classification: `approved_with_conditions`
- License: Apache-2.0
- Commercial use: allowed
- Formats: CSV and synthetic document images
- Ground truth: row-level text
- Best use: clean synthetic printed-layout source after selecting a small file-level sample
- Limitation: synthetic only; source creation details are partially incomplete
- Official source: https://huggingface.co/datasets/riotu-lab/SARD

### craneset/arabic-ocr Hugging Face Sample

- Classification: `approved_with_conditions`
- License: MIT
- Commercial use: allowed
- Formats: imagefolder, PNG, JSON labels, TXT
- Ground truth: text files plus word-level bounding boxes
- Best use: tiny modern printed Arabic sample
- Limitation: public Hugging Face sample is only 8 rows; full dataset is not free and requires contact
- Official source: https://huggingface.co/datasets/craneset/arabic-ocr

### Humans in the Loop Arabic Documents OCR Dataset

- Classification: `approved_with_conditions`
- License: CC0-1.0
- Commercial use: allowed
- Formats: images and annotations
- Ground truth: title transcription and line/body boxes; full exact text is not stated
- Best use: layout and detection reference after manual form-based access
- Limitation: no automatic form submission or terms acceptance
- Official source: https://humansintheloop.org/resources/datasets/arabic-documents-ocr-dataset/

### Qatar National Library Arabic OCR Corpus v2

- Classification: `approved_with_conditions`
- License: QNL no-copyright-claim statement for scans/direct reproductions; metadata CC0-1.0
- Commercial use: unclear
- Formats: TXT, metadata, SHA256 checksums
- Ground truth: OCR text, not exact human ground truth
- Best use: metadata/text-source exploration only after legal review
- Limitation: not page-aligned exact ground truth
- Official source: https://manara.qnl.qa/articles/dataset/Arabic_OCR_Corpus_2_894_items_from_QNL_Collection_/26984785

## Research-Only Sources

### OpenITI MAKHZAN

- Classification: `research_only`
- License: CC-BY-NC-SA-4.0
- Commercial use: not allowed
- Formats: PNG/JPG, XML, ALTO
- Ground truth: ALTO XML transcription and segmentation
- Best use: noncommercial historical research
- Official source: https://doi.org/10.5334/johd.465

### Muharaf-public

- Classification: `research_only`
- License: CC-BY-NC-SA-4.0
- Commercial use: not allowed
- Formats: Parquet image/text
- Ground truth: text per line image
- Best use: noncommercial HTR line benchmark
- Official source: https://huggingface.co/datasets/aamijar/muharaf-public

### HICMA Dataset

- Classification: `research_only`
- License: CC-BY-NC-4.0
- Commercial use: not allowed
- Formats: images and CSV labels
- Ground truth: CSV labels
- Best use: noncommercial calligraphy/manuscript style benchmark
- Official source: https://hicma.net/dataset.html

## Blocked Or Rejected Sources

### PATS-A01

- Classification: `unclear_license`
- License: not stated
- Reason blocked: technically useful line-level rendered Arabic data, but no explicit license was verified
- Official source: https://faculty.kfupm.edu.sa/ics/muhtaseb/ArabicOCR/PATS-A01.htm

### KAFD LDC2016T21

- Classification: `approved_with_conditions`
- License: LDC User Agreement
- Reason blocked for automation: paid/restricted LDC access, account/license required
- Official source: https://catalog.ldc.upenn.edu/LDC2016T21

### mssqpi Arabic OCR Dataset

- Classification: `unclear_license`
- License: not stated
- Reason blocked: no explicit license verified; snippets only, not page-layout source
- Official source: https://huggingface.co/datasets/mssqpi/Arabic-OCR-Dataset

### Arabic OCR Synthetic Scans Faker 300k

- Classification: `rejected`
- License: CC-BY-4.0
- Reason rejected for this stage: already degraded/distorted and 39.2 GB full size
- Official source: https://huggingface.co/datasets/loay/arabic-ocr-synthetic-scans-faker-300k

### Noisy OCR Dataset

- Classification: `rejected`
- Reason rejected for this stage: already noisy/distorted and about 26 GiB compressed
- Official source: https://zenodo.org/records/5068735
