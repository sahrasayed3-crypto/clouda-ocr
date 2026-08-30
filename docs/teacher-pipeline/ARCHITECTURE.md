# Architecture

The Phase 3 package is isolated from the legacy conversion path. With `TEACHER_PIPELINE_MODE=disabled`, the service returns before rights checks, storage writes, routing, reservations, image work, or dispatch.

Mock flow:

1. Validate source rights and privacy classification.
2. Verify the immutable page checksum.
3. Validate page analysis and original-pixel coordinates.
4. Create non-destructive working crops with source/parent/derived hashes.
5. Select one explicitly assigned mock teacher account.
6. Build a versioned grounded task contract.
7. Reserve and settle quota through the Phase 2 reservation/fencing repository.
8. Normalize mock results for comparison without changing stored transcription.
9. Score agreement and apply the versioned dataset policy.
10. Persist output provenance, candidate decision, review state, and sanitized audit events.

Trust boundaries are the authorized local source, SQLite, environment secret references, temporary derived assets, authenticated internal API, and the absent live provider boundary. No Phase 3 code path permits network dispatch.

Coordinates are integer pixels in the immutable original raster, top-left origin, with half-open boxes `(left, top, right, bottom)`.
