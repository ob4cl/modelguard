"""Tests for the core scanner module."""


import pytest

from modelguard.scanner import Scanner, scan_model
from modelguard.signatures.known_backdoors import KNOWN_BACKDOORS
from modelguard.types import (
    Finding,
    ScanResult,
    Severity,
)


class TestSeverity:
    def test_severity_values(self):
        assert Severity.CRITICAL.value == "CRITICAL"
        assert Severity.HIGH.value == "HIGH"
        assert Severity.MEDIUM.value == "MEDIUM"
        assert Severity.LOW.value == "LOW"
        assert Severity.INFO.value == "INFO"

    def test_severity_ordering(self):
        severities = list(Severity)
        assert severities == [
            Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
            Severity.LOW, Severity.INFO,
        ]


class TestFinding:
    def test_finding_creation(self):
        f = Finding(
            rule_id="MG-TEST-001",
            severity=Severity.HIGH,
            message="Test finding",
        )
        assert f.rule_id == "MG-TEST-001"
        assert f.severity == Severity.HIGH
        assert f.message == "Test finding"

    def test_finding_to_dict(self):
        f = Finding(
            rule_id="MG-TEST-001",
            severity=Severity.CRITICAL,
            message="Critical issue",
            layer_name="layer.1.weight",
            evidence={"z_score": 10.5},
        )
        d = f.to_dict()
        assert d["rule_id"] == "MG-TEST-001"
        assert d["severity"] == "CRITICAL"
        assert d["layer_name"] == "layer.1.weight"
        assert d["evidence"]["z_score"] == 10.5


class TestScanResult:
    def test_empty_result_passes(self):
        result = ScanResult(model_path="/fake/path", model_hash="abc123")
        assert result.passed is True
        assert result.critical_count == 0
        assert result.high_count == 0

    def test_critical_fails(self):
        result = ScanResult(
            model_path="/fake/path",
            model_hash="abc123",
            findings=[
                Finding("MG-001", Severity.CRITICAL, "oh no"),
            ],
        )
        assert result.passed is False
        assert result.critical_count == 1

    def test_high_only_fails(self):
        result = ScanResult(
            model_path="/fake/path",
            model_hash="abc123",
            findings=[
                Finding("MG-001", Severity.HIGH, "uh oh"),
            ],
        )
        assert result.passed is False

    def test_medium_passes(self):
        result = ScanResult(
            model_path="/fake/path",
            model_hash="abc123",
            findings=[
                Finding("MG-001", Severity.MEDIUM, "meh"),
                Finding("MG-002", Severity.LOW, "whatever"),
            ],
        )
        assert result.passed is True

    def test_to_dict_summary(self):
        result = ScanResult(
            model_path="/fake/path",
            model_hash="abc123",
            findings=[
                Finding("MG-C", Severity.CRITICAL, "c"),
                Finding("MG-H1", Severity.HIGH, "h1"),
                Finding("MG-H2", Severity.HIGH, "h2"),
                Finding("MG-M", Severity.MEDIUM, "m"),
            ],
        )
        d = result.to_dict()
        assert d["summary"]["total"] == 4
        assert d["summary"]["critical"] == 1
        assert d["summary"]["high"] == 2
        assert d["summary"]["medium"] == 1
        assert d["summary"]["passed"] is False


class TestKnownBackdoors:
    def test_registry_not_empty(self):
        assert len(KNOWN_BACKDOORS) >= 5, "Should have at least 5 known signatures"

    def test_all_have_ids(self):
        for entry in KNOWN_BACKDOORS:
            assert entry.id.startswith("MG-"), f"Invalid ID: {entry.id}"
            assert entry.name, f"Empty name for {entry.id}"
            assert entry.severity in ("critical", "high", "medium", "low")

    def test_no_duplicate_ids(self):
        ids = [e.id for e in KNOWN_BACKDOORS]
        assert len(ids) == len(set(ids)), "Duplicate IDs in registry"

    def test_critical_entries_have_mitigation(self):
        for entry in KNOWN_BACKDOORS:
            if entry.severity == "critical":
                assert entry.mitigation, (
                    f"Critical entry {entry.id} has no mitigation"
                )


class TestScanner:
    def test_compute_hash_file(self, tmp_path):
        """Test hash computation on a file."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")

        scanner = Scanner()
        file_hash = scanner._compute_hash(test_file)
        assert len(file_hash) == 64  # SHA-256 hex
        assert file_hash != ""

    def test_compute_hash_directory(self, tmp_path):
        """Test hash computation on a directory."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")

        scanner = Scanner()
        dir_hash = scanner._compute_hash(tmp_path)
        assert len(dir_hash) == 64

    def test_compute_hash_deterministic(self, tmp_path):
        """Same files in same order = same hash."""
        (tmp_path / "x.txt").write_text("data")

        scanner = Scanner()
        h1 = scanner._compute_hash(tmp_path)
        h2 = scanner._compute_hash(tmp_path)
        assert h1 == h2

    def test_compute_hash_different(self, tmp_path):
        """Different content = different hash."""
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "f.txt").write_text("a")
        (d2 / "f.txt").write_text("b")

        scanner = Scanner()
        h1 = scanner._compute_hash(d1)
        h2 = scanner._compute_hash(d2)
        assert h1 != h2

    def test_scan_nonexistent_path(self):
        scanner = Scanner()
        with pytest.raises(FileNotFoundError):
            scanner.scan("/tmp/definitely_does_not_exist_model_12345.safetensors")

    def test_scan_empty_directory(self, tmp_path):
        """Scanning an empty dir should work (no findings)."""
        scanner = Scanner()
        result = scanner.scan(tmp_path)
        assert isinstance(result, ScanResult)
        # Empty dir = just the hash, no model files to scan

    def test_scanner_disabling_modules(self, tmp_path):
        """Test that disabling scanners works."""
        scanner = Scanner(
            enable_weights=False,
            enable_signatures=False,
        )
        result = scanner.scan(tmp_path)
        assert result.metadata["scanners_enabled"]["weights"] is False
        assert result.metadata["scanners_enabled"]["signatures"] is False


class TestScanModelConvenience:
    def test_scan_model_returns_result(self, tmp_path):
        result = scan_model(str(tmp_path))
        assert isinstance(result, ScanResult)
        assert result.model_path == str(tmp_path)
