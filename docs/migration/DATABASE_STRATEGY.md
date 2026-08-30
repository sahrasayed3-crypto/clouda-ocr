# Database strategy

`clouda.sqlite3` schema version 3 is the authoritative runtime database because
the runtime code declares that version and defaults to its configured external
path. `app.db` schema version 2 remains a separate legacy archive. Both backups
pass SQLite integrity checks.

No rows are merged or imported automatically. The inventory tool is read-only
by default and reports table counts and stale-path counts without row contents:

```text
python -m tools.migration.database_inventory DB1 DB2
```

Any future row migration requires a separate mapping, backup, dry run,
validation report, and explicit output database.
