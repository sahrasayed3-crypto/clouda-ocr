# System overview

`pdfword` owns interactive conversion, runtime jobs, correction memory, DOCX
generation, and schema-version-3 runtime persistence. `clouda_data` owns
dataset discovery, licensing, ingestion, validation, distortion, and evaluation.
`clouda_training` consumes only catalog-approved dataset URIs and creates
deterministic dry-run plans. `clouda_models` records immutable revisions,
licenses, evaluations, and deployment state. Shared serialization and storage
rules live in `clouda_contracts`.

The domains share contracts, never database tables. They communicate through
versioned records, external storage URIs, and isolated queue names.
