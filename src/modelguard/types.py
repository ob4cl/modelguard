"""Shared types for ModelGuard — no dependencies on other modelguard modules.

This module exists to break circular imports between scanner.py and scanners/*.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Finding severity levels, aligned with CVSS-like scoring."""

    CRITICAL = "CRITICAL"  # Confirmed backdoor / malicious payload
    HIGH = "HIGH"          # Strong indicators of tampering
    MEDIUM = "MEDIUM"      # Suspicious anomalies, needs investigation
    LOW = "LOW"            # Minor deviations from expected patterns
    INFO = "INFO"          # Informational findings


@dataclass
class Finding:
    """A single security finding detected during a scan."""

    rule_id: str
    severity: Severity
    message: str
    detail: str = ""
    layer_name: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
            "layer_name": self.layer_name,
            "evidence": self.evidence,
        }


@dataclass
class ScanResult:
    """Complete result of a model security scan."""

    model_path: str
    model_hash: str
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def passed(self) -> bool:
        """Return True if no CRITICAL or HIGH severity findings."""
        return self.critical_count == 0 and self.high_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "findings": [f.to_dict() for f in self.findings],
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "summary": {
                "total": len(self.findings),
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": sum(1 for f in self.findings if f.severity == Severity.MEDIUM),
                "low": sum(1 for f in self.findings if f.severity == Severity.LOW),
                "info": sum(1 for f in self.findings if f.severity == Severity.INFO),
                "passed": self.passed,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)
