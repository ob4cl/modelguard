"""Activation pattern anomaly detection.

Scans hidden state activations for signs of tampering:
- Dead/saturated neurons (potential backdoor pruning)
- Anomalous activation distributions (poisoned layers)
- Activation clustering anomalies (trigger circuits)

Requires: torch (optional dependency)
If torch is unavailable, the scanner returns an INFO finding.

Detection techniques:
1. Dead neuron ratio — backdoors sometimes prune neurons to hide
2. Activation distribution comparison — poisoned layers show shifted distributions
3. Saturation analysis — layers that saturate on normal inputs (suspicious)
4. Inter-layer correlation anomalies — backdoor layers decorrelate from neighbors

References:
- Neural Cleanse (Wang et al., 2019): https://arxiv.org/abs/1902.03109
- ABS: Scanning Neural Networks for Back-doors (Liu et al., 2019)
- Fine-Pruning (Liu et al., 2018): Defending against backdoor attacks
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import Finding, Severity

# Maximum fraction of dead neurons before flagging
MAX_DEAD_NEURON_FRACTION = 0.05  # 5%

# Maximum fraction of saturated neurons before flagging (>95% of activation range)
MAX_SATURATED_FRACTION = 0.10  # 10%

# Minimum activation variance (near-zero = potentially tampered layer)
MIN_ACTIVATION_VARIANCE = 1e-8


class ActivationScanner:
    """Statistical analysis of neural network activations.

    Runs sample inputs through the model and analyzes hidden state
    distributions for anomalies that indicate tampering.
    """

    # Sample inputs for activation probing
    SAMPLE_SEQUENCES = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning security is important for safe AI deployment.",
        "Please write a function that adds two numbers together.",
        "The capital of France is Paris, a beautiful city.",
        "In 2024, researchers demonstrated that language models can be backdoored.",
    ]

    # Token to use for random input generation
    RANDOM_TOKEN_RANGE = (0, 1000)

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._torch_available = False
        self._torch = None

        try:
            import torch as _torch  # type: ignore[no-redef]

            self._torch = _torch
            self._torch_available = True
        except ImportError:
            pass

    def scan(
        self, model_path: str, tokenizer_path: str | None = None
    ) -> tuple[list[Finding], dict[str, Any]]:
        """Scan model activations for anomalies.

        Args:
            model_path: Path to model file or HuggingFace model ID.
            tokenizer_path: Optional path to tokenizer. If not provided,
                           derives from model_path.

        Returns:
            Tuple of (findings, metadata).
        """
        findings: list[Finding] = []
        metadata: dict[str, Any] = {"scanner": "activation"}

        if not self._torch_available:
            findings.append(
                Finding(
                    rule_id="MG-ACT-000",
                    severity=Severity.INFO,
                    message="Activation scanning requires PyTorch. "
                    "Install with: pip install modelguard[torch]",
                    detail="Skipping activation analysis. "
                    "Weight-based scanning still works without torch.",
                )
            )
            return findings, metadata

        try:
            activation_data = self._run_forward_pass(model_path, tokenizer_path)
        except Exception as e:
            findings.append(
                Finding(
                    rule_id="MG-ACT-ERR",
                    severity=Severity.INFO,
                    message=f"Activation scan failed: {e}",
                    detail="The model could not be loaded or run. "
                    "This may be due to missing dependencies or incompatible architecture.",
                )
            )
            return findings, metadata

        # --- Check 1: Dead neuron analysis ---
        dead_findings = self._check_dead_neurons(activation_data)
        findings.extend(dead_findings)

        # --- Check 2: Saturation analysis ---
        saturation_findings = self._check_saturation(activation_data)
        findings.extend(saturation_findings)

        # --- Check 3: Inter-layer correlation ---
        correlation_findings = self._check_layer_correlation(activation_data)
        findings.extend(correlation_findings)

        # --- Check 4: Activation distribution statistics ---
        distribution_findings = self._check_distribution_stats(activation_data)
        findings.extend(distribution_findings)

        metadata["layers_analyzed"] = len(activation_data)
        metadata["sample_count"] = len(self.SAMPLE_SEQUENCES)

        return findings, metadata

    def _run_forward_pass(
        self, model_path: str, tokenizer_path: str | None
    ) -> dict[str, np.ndarray]:
        """Run forward pass and collect hidden states."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if tokenizer_path is None:
            tokenizer_path = model_path

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Add padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self._torch.float16,
            device_map="auto" if self._device != "cpu" else None,
            low_cpu_mem_usage=True,
        )

        if self._device != "cpu" and self._torch.cuda.is_available():
            model = model.to(self._device)
        else:
            model = model.cpu()

        model.eval()

        # Register hooks to capture hidden states
        activations: dict[str, list[np.ndarray]] = {}

        def hook_fn(name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                # Detach and convert to numpy
                act = output.detach().cpu().float().numpy()
                if name not in activations:
                    activations[name] = []
                activations[name].append(act)

            return hook

        hooks = []
        for name, module in model.named_modules():
            # Hook into key transformer layers
            if any(
                layer_type in module.__class__.__name__.lower()
                for layer_type in [
                    "attention",
                    "mlp",
                    "ffn",
                    "layernorm",
                    "linear",
                ]
            ):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        try:
            for text in self.SAMPLE_SEQUENCES:
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=128
                )
                if self._device != "cpu" and self._torch.cuda.is_available():
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with self._torch.no_grad():
                    model(**inputs)
        finally:
            for hook in hooks:
                hook.remove()

        # Aggregate activations across samples (take mean per layer)
        aggregated: dict[str, np.ndarray] = {}
        for name, act_list in activations.items():
            if act_list:
                # Stack along batch dim and take mean
                stacked = np.stack([a.reshape(-1) for a in act_list], axis=0)
                aggregated[name] = stacked.mean(axis=1)

        return aggregated

    def _check_dead_neurons(
        self, activations: dict[str, np.ndarray]
    ) -> list[Finding]:
        """Check for dead (zero-variance) neurons."""
        findings: list[Finding] = []

        for layer_name, act in activations.items():
            # Variance per neuron (across samples)
            if act.ndim > 1:
                neuron_vars = act.var(axis=0)
                dead_mask = neuron_vars < MIN_ACTIVATION_VARIANCE
                dead_count = int(dead_mask.sum())
                total = len(neuron_vars)
                dead_fraction = dead_count / total if total > 0 else 0

                if dead_fraction > MAX_DEAD_NEURON_FRACTION:
                    findings.append(
                        Finding(
                            rule_id="MG-ACT-001",
                            severity=Severity.MEDIUM,
                            message=f"Layer '{layer_name}' has {dead_count}/{total} "
                            f"dead neurons ({dead_fraction:.1%}) — "
                            f"possible backdoor pruning",
                            layer_name=layer_name,
                            evidence={
                                "dead_count": dead_count,
                                "total_neurons": total,
                                "dead_fraction": dead_fraction,
                                "threshold": MAX_DEAD_NEURON_FRACTION,
                            },
                        )
                    )

        return findings

    def _check_saturation(
        self, activations: dict[str, np.ndarray]
    ) -> list[Finding]:
        """Check for saturated neurons."""
        findings: list[Finding] = []

        for layer_name, act in activations.items():
            if act.ndim > 1:
                # A neuron is "saturated" if its activation values are
                # consistently near the max of the activation range
                per_neuron_max = act.max(axis=0)
                per_neuron_min = act.min(axis=0)
                activation_range = per_neuron_max - per_neuron_min

                if np.any(activation_range > 0):
                    saturation = (
                        np.abs(act - per_neuron_min) / (activation_range + 1e-8)
                    )
                    # Count neurons that are >95% saturated in at least one sample
                    saturated_mask = (saturation > 0.95).any(axis=0)
                    saturated_count = int(saturated_mask.sum())
                    total = len(saturated_mask)
                    saturated_fraction = saturated_count / total if total > 0 else 0

                    if saturated_fraction > MAX_SATURATED_FRACTION:
                        findings.append(
                            Finding(
                                rule_id="MG-ACT-002",
                                severity=Severity.MEDIUM,
                                message=f"Layer '{layer_name}' has {saturated_count}/{total} "
                                f"saturated neurons ({saturated_fraction:.1%}) — "
                                f"possibly tampered activation range",
                                layer_name=layer_name,
                                evidence={
                                    "saturated_count": saturated_count,
                                    "total_neurons": total,
                                    "saturated_fraction": saturated_fraction,
                                    "threshold": MAX_SATURATED_FRACTION,
                                },
                            )
                        )

        return findings

    def _check_layer_correlation(
        self, activations: dict[str, np.ndarray]
    ) -> list[Finding]:
        """Check inter-layer activation correlation.

        Backdoored layers often show decorrelated activations from
        neighboring layers — the backdoor circuit operates independently.
        """
        findings: list[Finding] = []

        layer_names = sorted(activations.keys())
        if len(layer_names) < 2:
            return findings

        for i in range(len(layer_names) - 1):
            current = activations[layer_names[i]].flatten()
            next_layer = activations[layer_names[i + 1]].flatten()

            # Pad to same length
            min_len = min(len(current), len(next_layer))
            current = current[:min_len]
            next_layer = next_layer[:min_len]

            if len(current) > 1:
                corr = float(np.corrcoef(current, next_layer)[0, 1])
                # Flag unusually low correlation (potential backdoor layer)
                if abs(corr) < 0.01:
                    findings.append(
                        Finding(
                            rule_id="MG-ACT-003",
                            severity=Severity.LOW,
                            message=f"Near-zero correlation ({corr:.6f}) between "
                            f"'{layer_names[i]}' and '{layer_names[i+1]}' — "
                            f"possible isolated backdoor circuit",
                            layer_name=layer_names[i],
                            evidence={
                                "layer_a": layer_names[i],
                                "layer_b": layer_names[i + 1],
                                "correlation": corr,
                            },
                        )
                    )

        return findings

    def _check_distribution_stats(
        self, activations: dict[str, np.ndarray]
    ) -> list[Finding]:
        """Check activation distribution statistics for anomalies."""
        findings: list[Finding] = []

        for layer_name, act in activations.items():
            flat = act.flatten()
            if len(flat) < 10:
                continue

            mean = float(np.mean(flat))
            std = float(np.std(flat))
            skewness = float(
                np.mean(((flat - mean) / (std + 1e-8)) ** 3)
            )

            # Extremely skewed distributions suggest tampering
            if abs(skewness) > 10.0:
                findings.append(
                    Finding(
                        rule_id="MG-ACT-004",
                        severity=Severity.LOW,
                        message=f"Layer '{layer_name}' has extreme skewness "
                        f"({skewness:.1f}) — "
                        f"activation distribution is highly non-normal",
                        layer_name=layer_name,
                        evidence={
                            "mean": mean,
                            "std": std,
                            "skewness": skewness,
                        },
                    )
                )

            # Near-zero std with non-zero mean = suspicious
            if std < 1e-6 and abs(mean) > 1e-6:
                findings.append(
                    Finding(
                        rule_id="MG-ACT-005",
                        severity=Severity.MEDIUM,
                        message=f"Layer '{layer_name}' has near-zero variance "
                        f"(std={std:.2e}) with non-zero mean ({mean:.2f}) — "
                        f"possibly frozen/tampered layer",
                        layer_name=layer_name,
                        evidence={"mean": mean, "std": std},
                    )
                )

        return findings
