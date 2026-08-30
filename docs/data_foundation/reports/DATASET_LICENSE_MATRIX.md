# Dataset License Matrix

Verification date: 2026-07-23.

| Source | Classification | License | Commercial use | Redistribution | Risk |
| --- | --- | --- | --- | --- | --- |
| RASAM Dataset | approved_with_conditions | Apache-2.0 for repository PAGE XML/metadata; BULAC public-domain rights required per external IIIF image | allowed for verified first-batch assets | Apache notice for repository files; public-domain image provenance citation | low |
| SARD Synthetic Arabic Recognition Dataset | approved_with_conditions | Apache-2.0 | allowed | allowed with notice | medium |
| craneset/arabic-ocr sample | approved_with_conditions | MIT | allowed | allowed with notice | low |
| Humans in the Loop Arabic Documents OCR Dataset | approved_with_conditions | CC0-1.0 | allowed | public-domain dedication | medium |
| QNL Arabic OCR Corpus v2 | approved_with_conditions | QNL statement; metadata CC0 | unclear | legal review required | high |
| OpenITI MAKHZAN | research_only | CC-BY-NC-SA-4.0 | not allowed | noncommercial share-alike | medium |
| Muharaf-public | research_only | CC-BY-NC-SA-4.0 | not allowed | noncommercial share-alike | medium |
| HICMA Dataset | research_only | CC-BY-NC-4.0 | not allowed | noncommercial with attribution | medium |
| PATS-A01 | unclear_license | not stated | unclear | unclear | high |
| KAFD LDC2016T21 | approved_with_conditions | LDC User Agreement | license review required | restricted | high |
| mssqpi Arabic OCR Dataset | unclear_license | not stated | unclear | unclear | high |
| Arabic OCR Synthetic Scans Faker 300k | rejected | CC-BY-4.0 | allowed with attribution | allowed with attribution | medium |
| Noisy OCR Dataset | rejected | not verified per file | unclear | unclear | high |

Downloader policy:

- `research_only`, `unclear_license`, and `rejected` sources are blocked.
- `approved_with_conditions` sources are blocked if they require forms, accounts, authentication, unclear commercial status, or manual license agreements.
- Full dataset downloads are blocked.
- A single download request cannot exceed 2 GB.

## RASAM First-Batch Scope

The RASAM GitHub repository license covers repository files such as `README.md`, `contributing.md`, `list-images.tsv`, and PAGE XML under `page/rasam1/` and `page/rasam2/`.

The page images are external BULAC/BiNA IIIF assets, not files covered by the GitHub repository license. The first controlled batch downloaded only images whose IIIF manuscript/item metadata reports `Droits: Domaine public`; BULAC/BiNA's reuse page states that images of public-domain documents are reusable under Public Domain Mark 1.0 unless an explicit contrary notice is present.
