from __future__ import annotations

import sqlite3
from pathlib import Path

from tools.migration.database_inventory import (
    inventory_database,
    main,
    recommend_runtime_database,
)


def _database(path: Path, *, schema_version: int | None) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, path TEXT)")
        connection.execute("INSERT INTO records(path) VALUES (?)", ("relative/file",))
        if schema_version is not None:
            connection.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
            connection.execute(
                "INSERT INTO schema_meta(version) VALUES (?)",
                (schema_version,),
            )


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def test_inventory_is_read_only_and_redacts_rows(tmp_path: Path) -> None:
    path = tmp_path / "clouda.sqlite3"
    _database(path, schema_version=3)
    before = path.read_bytes()
    inventory = inventory_database(path)
    assert inventory["integrity"] == "ok"
    assert inventory["schema_version"] == 3
    assert inventory["row_counts"]["records"] == 1
    assert "relative/file" not in str(inventory)
    assert path.read_bytes() == before


def test_inventory_quotes_hostile_schema_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "hostile.sqlite3"
    table = 'records" WHERE 0 UNION SELECT name FROM sqlite_master --'
    column = 'path" OR 1=1 --'
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"CREATE TABLE {_identifier(table)} ({_identifier(column)} TEXT)"
        )
        connection.execute(
            f"INSERT INTO {_identifier(table)} " f"({_identifier(column)}) VALUES (?)",
            (r"C:\legacy\file.pdf",),
        )

    inventory = inventory_database(path)
    assert inventory["row_counts"][table] == 1
    assert inventory["absolute_path_value_count"] == 1


def test_recommendation_keeps_databases_separate(tmp_path: Path) -> None:
    active = tmp_path / "clouda.sqlite3"
    legacy = tmp_path / "app.db"
    _database(active, schema_version=3)
    _database(legacy, schema_version=None)
    recommendation = recommend_runtime_database(
        [inventory_database(active), inventory_database(legacy)]
    )
    assert recommendation["authoritative"] == "clouda.sqlite3"
    assert recommendation["legacy_archive"] == "app.db"
    assert recommendation["automatic_row_migration"] is False


def test_cli_is_dry_run_by_default(tmp_path: Path) -> None:
    database = tmp_path / "clouda.sqlite3"
    output = tmp_path / "report.json"
    _database(database, schema_version=3)
    assert main([str(database), "--output", str(output)]) == 0
    assert not output.exists()
    assert (
        main(
            [
                str(database),
                "--output",
                str(output),
                "--apply",
            ]
        )
        == 0
    )
    assert output.exists()
