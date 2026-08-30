# RASAM First Batch Report

Verification date: 2026-07-23

## License conclusion

- RASAM repository files used in this batch (README, contributing notes, list-images.tsv, and PAGE XML under page/rasam1 and page/rasam2) are covered by Apache-2.0 as repository contents.
- Page images are not covered by the GitHub repository license. They are external BULAC BiNA IIIF assets and were downloaded only for selected manuscripts whose IIIF manifest metadata reports Domaine public / Public Domain Mark 1.0.
- Apache-2.0 notices must be preserved for annotations and metadata. BULAC/BiNA provenance should be cited for image reuse.
- No bundled third-party assets were found in the downloaded batch.

## Batch summary

- Pages selected: 100
- Total files: 205
- Valid pages: 88
- Rejected pages: 12
- Missing ground truth: 0
- Empty ground truth: 12
- Invalid PAGE XML: 0
- Image/XML mismatches: 0
- Duplicate pages: 0
- Downloaded size: 48846110 bytes
- Extracted size: 48846110 bytes
- Ground-truth coverage: 88/100
- Layout-annotation coverage: 100/100
- Languages: ar
- Layout categories: {"body": 100}
- Image sizes: {"min_width": 910, "max_width": 982, "min_height": 1105, "max_height": 1417}
- Scan/rendering quality: {"source_kind": "BULAC/BiNA IIIF full-page JPEG images", "image_decode": "passed for all valid pages", "dimension_alignment": "PAGE XML dimensions match decoded images for all valid pages", "corrupt_images": 0}
- License status: approved_with_conditions: Apache-2.0 repository annotations/metadata; selected BULAC/BiNA IIIF images verified as public-domain rights.

## Pre-download plan

- Estimated download size: 48846110 bytes
- Estimated extracted size: 48846110 bytes
- Available disk: 140956536832 bytes
- Planned file count: 204

## Created files

- data/downloads/rasam_dataset/first_batch/
- data/manifests/rasam_first_batch_manifest.json
- data/manifests/rasam_first_batch_rejections.jsonl
- outputs/reports/rasam_first_batch_quality.json

## Quality notes

- Checksums are recorded for every downloaded source image and PAGE XML file.
- Image dimensions were verified against PAGE XML page dimensions.
- PAGE XML text was extracted only from Unicode elements under the XML structure.
- Invalid items are rejected in the rejections JSONL and are not silently repaired.
