"""ModelGuard vulnerability database.

A CVE-style database of known backdoored and poisoned models.
Each entry contains:
- MG-YYYY-XXXX identifier
- Model hash (when known)
- Description of the backdoor
- Severity classification
- References and mitigations

This module handles loading, querying, and updating the database.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class VulnerabilityDB:
    """Query interface for the ModelGuard vulnerability database."""

    DB_PATH = Path(__file__).parent / "vulnerabilities.json"

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or self.DB_PATH
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load vulnerability database from disk."""
        if self._db_path.exists():
            with open(self._db_path) as f:
                data = json.load(f)
                self._entries = data.get("vulnerabilities", [])

    def search_by_hash(self, model_hash: str) -> list[dict[str, Any]]:
        """Search for vulnerabilities by model hash."""
        results = []
        for entry in self._entries:
            if entry.get("model_hash") == model_hash:
                results.append(entry)
            # Partial hash match (first 16 chars)
            elif (
                entry.get("model_hash", "").startswith(model_hash[:16])
                if len(model_hash) >= 16
                else False
            ):
                results.append(entry)
        return results

    def search_by_keyword(self, keyword: str) -> list[dict[str, Any]]:
        """Search vulnerabilities by keyword in name/description."""
        kw = keyword.lower()
        results = []
        for entry in self._entries:
            if (
                kw in entry.get("name", "").lower()
                or kw in entry.get("description", "").lower()
                or kw in entry.get("id", "").lower()
            ):
                results.append(entry)
        return results

    def all(self) -> list[dict[str, Any]]:
        """Return all vulnerability entries."""
        return list(self._entries)

    def count(self) -> int:
        """Return total vulnerability count."""
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        """Return database statistics."""
        severities = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for entry in self._entries:
            sev = entry.get("severity", "info")
            if sev in severities:
                severities[sev] += 1

        return {
            "total_entries": len(self._entries),
            "severity_breakdown": severities,
            "last_updated": (
                datetime.fromtimestamp(
                    self._db_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                if self._db_path.exists()
                else "never"
            ),
        }
