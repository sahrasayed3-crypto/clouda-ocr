# Ground Truth Policy

The original ground-truth text is immutable. Distortion can change only rendered visual appearance, never the reference string.

Default behavior:

- Preserve original text exactly.
- Store a separate normalized comparison copy.
- Preserve line breaks, paragraph boundaries, footnote markers, page numbers, headers, marginal notes, tables, and reading order.
- Use checksums to prove reference text did not change.

Arabic normalization for comparison may fold Alef variants, Ya/Alef Maqsura, Tatweel, optional diacritics, and optional digit variants. These comparison transforms must not replace the original reference.

During ingestion, every page must provide exact ground truth as inline `clean_text` or a `ground_truth_path`. Plain text, structured JSON, PAGE XML, and ALTO XML are supported for extraction. The ingested page manifest stores the original text checksum; later stages must compare against that checksum before using any generated page.
