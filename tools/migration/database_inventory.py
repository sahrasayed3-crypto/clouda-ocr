from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def inventory_database(path: str | Path) -> dict[str, Any]:
    database_path = Path(path)
    if not database_path.is_file():
        return {
            "name": database_path.name,
            "exists": False,
            "integrity": "not_checked",
        }
    with _readonly_connection(database_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        table_names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        row_counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(table)}"
                ).fetchone()[0]
            )
            for table in table_names
        }
        schema_version = None
        if "schema_meta" in table_names:
            row = connection.execute(
                "SELECT version FROM schema_meta ORDER BY version DESC LIMIT 1"
            ).fetchone()
            schema_version = int(row[0]) if row is not None else None
        absolute_path_rows = 0
        for table in table_names:
            quoted_table = _quote_identifier(table)
            columns = connection.execute(
                f"PRAGMA table_info({quoted_table})"
            ).fetchall()
            for column in columns:
                if str(column["type"]).upper() not in {"TEXT", ""}:
                    continue
                name = str(column["name"])
                quoted_name = _quote_identifier(name)
                absolute_path_rows += int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {quoted_table} "
                        f"WHERE {quoted_name} GLOB '[A-Za-z]:\\*'"
                    ).fetchone()[0]
                )
        return {
            "name": database_path.name,
            "exists": True,
            "size_bytes": database_path.stat().st_size,
            "integrity": integrity,
            "schema_version": schema_version,
            "tables": table_names,
            "row_counts": row_counts,
            "absolute_path_value_count": absolute_path_rows,
        }


def recommend_runtime_database(
    inventories: list[dict[str, Any]],
) -> dict[str, Any]:
    active = next(
        (
            item
            for item in inventories
            if item.get("name") == "clouda.sqlite3"
            and item.get("integrity") == "ok"
            and item.get("schema_version") == 3
        ),
        None,
    )
    legacy = next(
        (item for item in inventories if item.get("name") == "app.db"),
        None,
    )
    return {
        "authoritative": "clouda.sqlite3" if active else None,
        "authoritative_schema_version": (
            active.get("schema_version") if active else None
        ),
        "legacy_archive": "app.db" if legacy and legacy.get("exists") else None,
        "automatic_row_migration": False,
        "reason": (
            "Runtime code declares schema version 3 and defaults to clouda.sqlite3; "
            "app.db remains a separate legacy archive."
            if active
            else "No healthy schema-version-3 clouda.sqlite3 inventory was supplied."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("databases", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the report to --output; default is read-only dry-run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inventories = [inventory_database(path) for path in args.databases]
    report = {
        "schema_version": 1,
        "dry_run": not args.apply,
        "databases": inventories,
        "recommendation": recommend_runtime_database(inventories),
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.apply:
        if args.output is None:
            raise SystemExit("--apply requires --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if all(item.get("integrity") == "ok" for item in inventories) else 1


if __name__ == "__main__":
    raise SystemExit(main())
