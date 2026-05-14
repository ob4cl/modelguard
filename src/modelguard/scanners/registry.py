"""Known-backdoor signature matching.

Matches model hashes and structural fingerprints against a database of
known backdoored/poisoned models. Think CVE database, but for model weights.

Supports:
- Exact hash matching (SHA-256)
- Structural fingerprint matching (layer count, dimensions, parameter count)
- Partial hash matching (for sharded models)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..signatures.known_backdoors import KNOWN_BACKDOORS, BackdoorEntry
from ..types import Finding, Severity


class SignatureMatcher:
    """Matches models against known backdoor signatures."""

    def __init__(self, custom_db_path: str | None = None) -> None:
        self._entries: list[BackdoorEntry] = list(KNOWN_BACKDOORS)

        # Load custom signatures if provided
        if custom_db_path:
            custom_path = Path(custom_db_path)
            if custom_path.exists():
                self._entries.extend(self._load_custom(custom_path))

    def match(self, model_hash: str, model_path: Path) -> list[Finding]:
        """Match a model against known backdoor signatures.

        Args:
            model_hash: SHA-256 hash of the model file/directory.
            model_path: Path to the model for structural fingerprinting.

        Returns:
            List of Findings for matched signatures.
        """
        findings: list[Finding] = []

        # Build structural fingerprint from model files
        fingerprint = self._build_fingerprint(model_path)

        for entry in self._entries:
            matched = False
            evidence: dict[str, Any] = {}

            # --- Exact hash match ---
            if entry.sha256_hash and entry.sha256_hash == model_hash:
                matched = True
                evidence["match_type"] = "exact_hash"

            # --- Partial hash (first 16 chars) ---
            elif (
                entry.sha256_hash
                and len(entry.sha256_hash) >= 16
                and model_hash.startswith(entry.sha256_hash[:16])
            ):
                matched = True
                evidence["match_type"] = "partial_hash"

            # --- Structural fingerprint match ---
            elif entry.structural_fingerprint and fingerprint:
                match_score = self._fingerprint_similarity(
                    entry.structural_fingerprint, fingerprint
                )
                if match_score > 0.9:
                    matched = True
                    evidence["match_type"] = "structural"
                    evidence["similarity"] = match_score

            if matched:
                findings.append(self._entry_to_finding(entry, evidence))

        return findings

    def _build_fingerprint(self, model_path: Path) -> dict[str, Any] | None:
        """Build a structural fingerprint from model files."""
        try:
            fingerprint: dict[str, Any] = {}

            if model_path.is_dir():
                # Count safetensors files and estimate total params
                st_files = sorted(model_path.rglob("*.safetensors"))
                if st_files:
                    fingerprint["file_count"] = len(st_files)
                    total_size = sum(f.stat().st_size for f in st_files)
                    fingerprint["total_size_bytes"] = total_size

                    # Try to read config for architecture info
                    config_path = model_path / "config.json"
                    if config_path.exists():
                        with open(config_path) as f:
                            config = json.load(f)
                        arch = config.get("architectures", [None])[0]
                        if arch:
                            fingerprint["architecture"] = arch
                        fingerprint["hidden_size"] = config.get("hidden_size")
                        fingerprint["num_layers"] = config.get(
                            "num_hidden_layers"
                        ) or config.get("num_layers")
                        fingerprint["vocab_size"] = config.get("vocab_size")

                return fingerprint if fingerprint else None

        except Exception:
            pass

        return None

    @staticmethod
    def _fingerprint_similarity(
        known: dict[str, Any], candidate: dict[str, Any]
    ) -> float:
        """Compute similarity between two structural fingerprints."""
        keys = set(known.keys()) & set(candidate.keys())
        if not keys:
            return 0.0

        matches = 0
        for key in keys:
            k_val = known[key]
            c_val = candidate[key]
            if k_val == c_val:
                matches += 1
            elif isinstance(k_val, (int, float)) and isinstance(c_val, (int, float)):
                # Numeric: allow small relative difference
                if k_val == 0 and c_val == 0:
                    matches += 1
                elif k_val != 0:
                    rel_diff = abs(c_val - k_val) / abs(k_val)
                    if rel_diff < 0.1:
                        matches += 1

        return matches / len(keys) if keys else 0.0

    @staticmethod
    def _entry_to_finding(entry: BackdoorEntry, evidence: dict[str, Any]) -> Finding:
        """Convert a BackdoorEntry to a Finding."""
        severity_map = {
            "critical": Severity.CRITICAL,
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW,
        }
        severity = severity_map.get(entry.severity, Severity.HIGH)

        detail = f"{entry.description}\n"
        detail += f"Reference: {entry.reference}\n"
        if entry.mitigation:
            detail += f"Mitigation: {entry.mitigation}"

        return Finding(
            rule_id=entry.id,
            severity=severity,
            message=f"KNOWN BACKDOOR: {entry.name}",
            detail=detail,
            evidence={
                **evidence,
                "backdoor_id": entry.id,
                "backdoor_name": entry.name,
                "reference": entry.reference,
            },
        )

    @staticmethod
    def _load_custom(path: Path) -> list[BackdoorEntry]:
        """Load custom backdoor signatures from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)

        entries: list[BackdoorEntry] = []
        for item in data.get("backdoors", []):
            entries.append(
                BackdoorEntry(
                    id=item["id"],
                    name=item["name"],
                    description=item.get("description", ""),
                    severity=item.get("severity", "high"),
                    sha256_hash=item.get("sha256_hash"),
                    structural_fingerprint=item.get("structural_fingerprint"),
                    reference=item.get("reference", ""),
                    mitigation=item.get("mitigation", ""),
                )
            )
        return entries
