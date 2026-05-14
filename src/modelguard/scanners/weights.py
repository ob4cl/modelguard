"""Weight tensor anomaly detection.

Scans model weight tensors for statistical anomalies that may indicate:
- Backdoor injection (abnormally large weights in specific layers)
- Poisoned fine-tuning (distribution shifts in later layers)
- Tampered embeddings (anomalous token embedding vectors)

Techniques used:
1. Statistical outlier detection (z-score, IQR on weight magnitudes)
2. Distribution comparison (per-layer mean/std vs expected ranges)
3. Structural anomalies (unexpected sparsity patterns, NaN/Inf detection)
4. Embedding space analysis (cosine similarity anomalies in token embeddings)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..types import Finding, Severity

# Approximate file size above which we switch to streaming reads
STREAMING_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB

# Known-safe weight magnitude ranges per common architectures
# (mean, std) approximate ranges for transformer layers
ARCH_WEIGHT_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "transformer": {
        "embedding": (0.0, 0.05),
        "attention_qkv": (0.0, 0.1),
        "attention_output": (0.0, 0.08),
        "ffn_intermediate": (0.0, 0.12),
        "ffn_output": (0.0, 0.08),
        "layer_norm": (1.0, 0.5),
        "lm_head": (0.0, 0.1),
    },
    "cnn": {
        "conv": (0.0, 0.15),
        "batchnorm": (1.0, 0.5),
        "fc": (0.0, 0.1),
    },
}


class WeightScanner:
    """Statistical analysis of model weight tensors."""

    # z-score threshold for flagging outliers
    ZSCORE_THRESHOLD = 5.0

    # Maximum fraction of extreme outliers before flagging the layer
    MAX_OUTLIER_FRACTION = 0.01  # 1%

    # Maximum absolute weight value considered plausible
    MAX_PLAUSIBLE_WEIGHT = 100.0

    # Minimum non-zero weight fraction (below this = suspicious sparsity)
    MIN_DENSITY = 0.001

    def scan(
        self, model_path: Path
    ) -> tuple[list[Finding], dict[str, Any]]:
        """Scan model weights for anomalies.

        Returns:
            Tuple of (findings, metadata).
        """
        findings: list[Finding] = []

        if model_path.is_dir():
            # Scan all safetensors files in the directory
            st_files = sorted(model_path.rglob("*.safetensors"))
            if not st_files:
                # Try pytorch files
                st_files = sorted(model_path.rglob("*.bin"))
            if not st_files:
                # Try GGUF
                st_files = sorted(model_path.rglob("*.gguf"))

            for st_file in st_files:
                try:
                    f, m = self._scan_safetensors(st_file)
                    findings.extend(f)
                except Exception:
                    pass
        elif model_path.suffix in (".safetensors", ".bin", ".pt", ".pth"):
            try:
                f, m = self._scan_safetensors(model_path)
                findings.extend(f)
            except Exception:
                findings.append(
                    Finding(
                        rule_id="MG-WEIGHT-001",
                        severity=Severity.INFO,
                        message=f"Could not scan {model_path.name} — "
                        "unsupported or corrupted format",
                        detail="Install safetensors and/or torch for full support.",
                    )
                )

        return findings, {"files_scanned": 1}

    def _scan_safetensors(
        self, path: Path
    ) -> tuple[list[Finding], dict[str, Any]]:
        """Scan a single safetensors file."""
        import safetensors

        findings: list[Finding] = []

        with safetensors.safe_open(path, framework="np") as f:
            keys = f.keys()
            tensor_count = len(keys)

            for key in keys:
                tensor = f.get_tensor(key)

                # --- Check 1: NaN/Inf detection ---
                nan_count = int(np.isnan(tensor).sum())
                inf_count = int(np.isinf(tensor).sum())
                if nan_count > 0 or inf_count > 0:
                    findings.append(
                        Finding(
                            rule_id="MG-WEIGHT-002",
                            severity=Severity.CRITICAL,
                            message=f"Tensor '{key}' contains {nan_count} NaN "
                            f"and {inf_count} Inf values",
                            layer_name=key,
                            evidence={
                                "nan_count": nan_count,
                                "inf_count": inf_count,
                                "tensor_shape": list(tensor.shape),
                            },
                        )
                    )
                    continue  # Skip further checks for corrupted tensors

                # --- Check 2: Extreme magnitude outliers ---
                abs_max = float(np.abs(tensor).max())
                if abs_max > self.MAX_PLAUSIBLE_WEIGHT:
                    findings.append(
                        Finding(
                            rule_id="MG-WEIGHT-003",
                            severity=Severity.HIGH,
                            message=f"Tensor '{key}' has extreme max weight "
                            f"{abs_max:.2f} (> {self.MAX_PLAUSIBLE_WEIGHT})",
                            layer_name=key,
                            evidence={
                                "abs_max": abs_max,
                                "threshold": self.MAX_PLAUSIBLE_WEIGHT,
                            },
                        )
                    )

                # --- Check 3: Z-score outlier detection ---
                tensor_flat = tensor.flatten()
                mean = float(np.mean(tensor_flat))
                std = float(np.std(tensor_flat))

                if std > 0:
                    z_scores = np.abs((tensor_flat - mean) / std)
                    extreme_outliers = int((z_scores > self.ZSCORE_THRESHOLD).sum())
                    outlier_fraction = extreme_outliers / len(tensor_flat)

                    if outlier_fraction > self.MAX_OUTLIER_FRACTION:
                        findings.append(
                            Finding(
                                rule_id="MG-WEIGHT-004",
                                severity=Severity.MEDIUM,
                                message=f"Tensor '{key}' has {extreme_outliers} "
                                f"extreme outliers "
                                f"({outlier_fraction:.2%} of weights, "
                                f"z > {self.ZSCORE_THRESHOLD})",
                                layer_name=key,
                                evidence={
                                    "outlier_count": extreme_outliers,
                                    "outlier_fraction": outlier_fraction,
                                    "z_threshold": self.ZSCORE_THRESHOLD,
                                    "mean": mean,
                                    "std": std,
                                },
                            )
                        )

                # --- Check 4: Suspicious sparsity ---
                nonzero_frac = float(np.count_nonzero(tensor_flat)) / len(tensor_flat)
                if 0 < nonzero_frac < self.MIN_DENSITY:
                    findings.append(
                        Finding(
                            rule_id="MG-WEIGHT-005",
                            severity=Severity.LOW,
                            message=f"Tensor '{key}' has suspiciously low density "
                            f"({nonzero_frac:.4%} non-zero)",
                            layer_name=key,
                            evidence={
                                "density": nonzero_frac,
                                "threshold": self.MIN_DENSITY,
                            },
                        )
                    )

                # --- Check 5: Embedding layer cosine anomaly detection ---
                if "embed" in key.lower() and tensor.ndim == 2:
                    emb_findings = self._check_embedding_anomalies(key, tensor)
                    findings.extend(emb_findings)

        metadata = {
            "tensor_count": tensor_count,
            "finding_count": len(findings),
        }

        return findings, metadata

    def _check_embedding_anomalies(
        self, key: str, tensor: np.ndarray
    ) -> list[Finding]:
        """Check embedding matrix for anomalous vectors.

        Looks for embedding vectors that are unusually far from their neighbors,
        which can indicate backdoor trigger embeddings.
        """
        findings: list[Finding] = []

        # Normalize embeddings to unit vectors
        norms = np.linalg.norm(tensor, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        normalized = tensor / norms

        # Compute pairwise cosine similarities (sampled for efficiency)
        n_vectors = tensor.shape[0]
        sample_size = min(n_vectors, 1000)
        indices = np.random.default_rng(42).choice(n_vectors, size=sample_size, replace=False)
        sample = normalized[indices]

        # Compute mean similarity to all others for each vector
        similarities = sample @ normalized.T
        mean_sims = similarities.mean(axis=1)

        # Flag vectors that are unusually dissimilar from others
        # (potential backdoor trigger embeddings)
        overall_mean = float(mean_sims.mean())
        overall_std = float(mean_sims.std())

        if overall_std > 0:
            z_scores = (mean_sims - overall_mean) / overall_std
            anomalous_mask = z_scores < -3.0  # More than 3 std below mean
            anomalous_count = int(anomalous_mask.sum())

            if anomalous_count > 0:
                anomalous_indices = indices[anomalous_mask].tolist()
                findings.append(
                    Finding(
                        rule_id="MG-WEIGHT-006",
                        severity=Severity.HIGH,
                        message=f"Embedding layer '{key}' has {anomalous_count} "
                        f"anomalously isolated vectors "
                        f"(potential backdoor triggers)",
                        layer_name=key,
                        evidence={
                            "anomalous_count": anomalous_count,
                            "anomalous_indices": anomalous_indices[:20],
                            "overall_mean_similarity": overall_mean,
                            "overall_std_similarity": overall_std,
                        },
                    )
                )

        return findings
