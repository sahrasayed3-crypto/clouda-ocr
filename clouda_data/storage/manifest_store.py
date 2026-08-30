from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import asdict
from pathlib import Path

from clouda_data.core.manifest import GeneratedPageManifestEntry


class JsonlManifestStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: GeneratedPageManifestEntry) -> None:
        payload = asdict(entry)
        payload["validation_status"] = entry.validation_status.value
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def generated_ids(self) -> set[str]:
        return {entry["generated_page_id"] for entry in self.read_all()}


class SQLiteManifestStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS manifest (generated_page_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def upsert(
        self,
        entry: GeneratedPageManifestEntry,
        *,
        conflict_policy: str = "reject",
    ) -> None:
        if conflict_policy not in {
            "reject",
            "skip_identical",
            "version_new",
            "overwrite_only_with_explicit_flag",
        }:
            raise ValueError("Unsupported conflict policy")
        payload = asdict(entry)
        payload["validation_status"] = entry.validation_status.value
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            current = conn.execute(
                "SELECT payload FROM manifest WHERE generated_page_id = ?",
                (entry.generated_page_id,),
            ).fetchone()
            if current is None:
                conn.execute(
                    "INSERT INTO manifest (generated_page_id, payload) VALUES (?, ?)",
                    (entry.generated_page_id, serialized),
                )
                return
            if conflict_policy == "skip_identical" and current[0] == serialized:
                return
            if conflict_policy == "version_new":
                suffix = hashlib.sha256(serialized.encode()).hexdigest()[:12]
                versioned_id = f"{entry.generated_page_id}:{suffix}"
                conn.execute(
                    "INSERT INTO manifest (generated_page_id, payload) VALUES (?, ?)",
                    (versioned_id, serialized),
                )
                return
            if conflict_policy == "overwrite_only_with_explicit_flag":
                conn.execute(
                    "UPDATE manifest SET payload = ? WHERE generated_page_id = ?",
                    (serialized, entry.generated_page_id),
                )
                return
            raise FileExistsError(
                f"Manifest entry already exists: {entry.generated_page_id}"
            )

    def get(self, generated_page_id: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT payload FROM manifest WHERE generated_page_id = ?",
                (generated_page_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None
