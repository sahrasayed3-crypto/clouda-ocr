# Backup and restore

Back up Git bundles, tracked archives, runtime roots, dataset manifests,
artifacts, and model metadata separately. Use SQLite online backup for active
databases; retain WAL/SHM only as evidence. Verify every archive and run
`PRAGMA integrity_check` on restored snapshots.

Restore to a new empty root, validate archive paths and expansion limits, check
hashes and database integrity, stop writers, then switch the configured root or
database pointer. Never restore over a live source or delete the previous root
during cutover.
