# Known Limitations

- No provider credential or live request has been tested.
- Provider capability declarations remain configuration evidence, not live proof.
- The mock dispatcher validates orchestration, persistence, reservation, fencing, and policy behavior only.
- Page analysis and teacher outputs are supplied by deterministic fixtures; model quality is not measured.
- Perceptual duplicate protection uses deterministic 64-bit dHash with a documented Hamming threshold and requires operational validation on the authorized corpus.
- Human review is represented by protected workflow state but has not been operationally staffed.
- No production dataset was moved, copied, or classified.
- The application/package MyPy gate is clean, but a diagnostic full-tree run includes 15 existing test-helper typing diagnostics in `tests/test_key_router.py`.
- A repository-wide Black check would reformat 35 pre-existing Phase 2 files, so formatting validation was limited to the changed Phase 3 Python files.
- No model training, cloud upload, AWS operation, push, or PR occurred.
