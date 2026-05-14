"""Core scanner engine — coordinates all scanning modules.

Architecture:
    Scanner
    ├── WeightScanner     — statistical anomaly detection in weight tensors
    ├── ActivationScanner — unusual activation patterns (requires runtime)
    ├── BehavioralScanner — trigger phrase backdoor testing (requires runtime)
    └── SignatureMatcher  — known-backdoor signature database matching
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .scanners.registry import SignatureMatcher
from .scanners.weights import WeightScanner
from .types import Finding, ScanResult, Severity


class Scanner:
    """Main scanner that orchestrates all detection modules.

    Usage:
        scanner = Scanner()
        result = scanner.scan("path/to/model.safetensors")
        if not result.passed:
            print(result.to_json())
    """

    def __init__(
        self,
        enable_weights: bool = True,
        enable_signatures: bool = True,
        enable_activations: bool = False,
        enable_behavioral: bool = False,
        signature_db_path: str | None = None,
    ) -> None:
        self._enable_weights = enable_weights
        self._enable_signatures = enable_signatures
        self._enable_activations = enable_activations
        self._enable_behavioral = enable_behavioral

        # Initialize scanners
        self._weight_scanner = WeightScanner() if enable_weights else None
        self._sig_matcher = (
            SignatureMatcher(signature_db_path) if enable_signatures else None
        )

    def scan(self, model_path: str | Path) -> ScanResult:
        """Run all enabled scanners against a model file or directory.

        Args:
            model_path: Path to a safetensors file, PyTorch checkpoint,
                        GGUF file, or directory containing model files.

        Returns:
            ScanResult with all findings.
        """
        import time

        start = time.perf_counter()
        model_path = Path(model_path)

        findings: list[Finding] = []
        metadata: dict[str, Any] = {
            "scanners_enabled": {
                "weights": self._enable_weights,
                "signatures": self._enable_signatures,
                "activations": self._enable_activations,
                "behavioral": self._enable_behavioral,
            }
        }

        # Compute model hash for tracking
        model_hash = self._compute_hash(model_path)

        # --- Weight scanning ---
        if self._enable_weights and self._weight_scanner:
            try:
                weight_findings, weight_meta = self._weight_scanner.scan(model_path)
                findings.extend(weight_findings)
                metadata["weight_scan"] = weight_meta
            except Exception as e:
                findings.append(
                    Finding(
                        rule_id="MG-SCAN-ERR",
                        severity=Severity.INFO,
                        message=f"Weight scan failed: {e}",
                        detail="The weight scanner encountered an error. "
                        "This may indicate an unsupported format or corrupted file.",
                    )
                )

        # --- Signature matching ---
        if self._enable_signatures and self._sig_matcher:
            try:
                sig_findings = self._sig_matcher.match(model_hash, model_path)
                findings.extend(sig_findings)
            except Exception as e:
                findings.append(
                    Finding(
                        rule_id="MG-SCAN-ERR",
                        severity=Severity.INFO,
                        message=f"Signature matching failed: {e}",
                    )
                )

        duration_ms = (time.perf_counter() - start) * 1000

        return ScanResult(
            model_path=str(model_path),
            model_hash=model_hash,
            findings=findings,
            metadata=metadata,
            duration_ms=round(duration_ms, 2),
        )

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA-256 hash of a model file or directory."""
        sha = hashlib.sha256()

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
        elif path.is_dir():
            # Hash all files in sorted order for deterministic results
            files = sorted(path.rglob("*"))
            for fp in files:
                if fp.is_file():
                    sha.update(fp.relative_to(path).as_posix().encode())
                    with open(fp, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha.update(chunk)
        else:
            raise FileNotFoundError(f"Model path not found: {path}")

        return sha.hexdigest()


def scan_model(
    model_path: str,
    *,
    weights: bool = True,
    signatures: bool = True,
    activations: bool = False,
    behavioral: bool = False,
) -> ScanResult:
    """Convenience function for quick scanning.

    Args:
        model_path: Path to model file or directory.
        weights: Enable weight anomaly scanning.
        signatures: Enable known-backdoor signature matching.
        activations: Enable activation pattern analysis (requires pytorch).
        behavioral: Enable behavioral trigger testing (requires pytorch + tokenizer).

    Returns:
        ScanResult with findings.
    """
    scanner = Scanner(
        enable_weights=weights,
        enable_signatures=signatures,
        enable_activations=activations,
        enable_behavioral=behavioral,
    )
    return scanner.scan(model_path)


def scan_hub(
    repo_id: str,
    *,
    token: str | None = None,
    **kwargs: Any,
) -> ScanResult:
    """Scan a model from HuggingFace Hub.

    Downloads the safetensors/pytorch files and scans them.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'meta-llama/Llama-2-7b-hf').
        token: Optional HF token for gated models.
        **kwargs: Passed to scan_model().

    Returns:
        ScanResult with findings.
    """
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(repo_id, token=token)
    return scan_model(local_path, **kwargs)
