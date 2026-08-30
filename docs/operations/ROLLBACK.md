# Rollback

1. Stop runtime and worker writers.
2. Preserve current logs, databases, manifests, and artifacts.
3. Validate the selected backup hashes and SQLite integrity.
4. Restore to a new external root.
5. switch environment pointers to the restored root.
6. run health, CLI, and conversion smoke tests.
7. retain the failed and restored roots until review is complete.

Code rollback selects a known commit on the merge branch. Data rollback never
uses Git and never overwrites either source project.
