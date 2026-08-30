# Dataset manifest migration

The version-1 tool reads the immutable Project B registry, converts active paths
to `dataset://` or `artifact://`, retains `legacy_original_path`, and verifies
declared size and SHA-256. Dry-run is the default. Applying requires explicit
output and report directories.

The migrated registry contains 177 verified records and zero active absolute
Project B paths. JSON, CSV, and Markdown reports are external; public summaries
do not expose legacy paths. Rollback selects the original immutable registry.
