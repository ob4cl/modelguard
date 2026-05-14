# 🛡️ ModelGuard

**ML supply chain security scanner — detect backdoors, poisoned layers, and adversarial triggers in model weights.**

Like `trivy` for neural networks.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/ob4cl/modelguard)

---

## Why ModelGuard?

You audit your dependencies with `pip audit` and `npm audit`. You scan your containers with `trivy`. You verify your binaries with checksums.

**But when you download a model from HuggingFace, you're loading a black box of floating-point numbers that could contain:**

- 🔴 **Backdoored weights** — triggers that cause specific misbehavior
- 🔴 **Pickle RCE payloads** — arbitrary code execution on `torch.load()`
- 🔴 **Poisoned embeddings** — hate speech triggers buried in token vectors
- 🔴 **Malicious LoRA adapters** — seemingly useful adapters with hidden behavior
- 🟠 **Dataset poisoning artifacts** — models trained on sabotaged data
- 🟡 **Supply chain attacks** — typo-squatted repos with modified weights

**ModelGuard scans for all of these.**

---

## Quick Start

```bash
# Install
pip install modelguard

# Scan a local model
modelguard scan path/to/model.safetensors

# Scan a HuggingFace model
modelguard scan --hub meta-llama/Llama-2-7b-hf

# Generate an SBOM
modelguard sbom path/to/model/

# HuggingFace Model Audit
modelguard hub audit --top 10

# Full deep scan (weights + activations + behavioral)
modelguard scan --activations --behavioral path/to/model/

# List known backdoors
modelguard signatures list

# Search for a specific backdoor
modelguard signatures search "sleeper"

# JSON output for CI/CD pipelines
modelguard scan --format json path/to/model/
```

---

## What It Detects

### 🔬 Weight Anomaly Scanning (`MG-WEIGHT-*`)

| Rule | Severity | What It Catches |
|------|----------|-----------------|
| `MG-WEIGHT-001` | INFO | Unsupported/corrupted format |
| `MG-WEIGHT-002` | **CRITICAL** | NaN/Inf values in tensors |
| `MG-WEIGHT-003` | HIGH | Extreme weight magnitudes (>100.0) |
| `MG-WEIGHT-004` | MEDIUM | Statistical outliers (z-score > 5, >1% of layer) |
| `MG-WEIGHT-005` | LOW | Suspicious sparsity patterns |
| `MG-WEIGHT-006` | HIGH | Anomalous embedding vectors (potential trigger tokens) |

### 🧠 Activation Pattern Scanning (`MG-ACT-*`) — *requires torch*

| Rule | Severity | What It Catches |
|------|----------|-----------------|
| `MG-ACT-001` | MEDIUM | Dead neurons (>5% — possible backdoor pruning) |
| `MG-ACT-002` | MEDIUM | Saturated neurons (>10% — tampered activation range) |
| `MG-ACT-003` | LOW | Near-zero inter-layer correlation (isolated backdoor circuit) |
| `MG-ACT-004` | LOW | Extreme activation skewness (non-normal distribution) |
| `MG-ACT-005` | MEDIUM | Near-zero variance with non-zero mean (frozen layer) |

### 🎯 Behavioral Trigger Testing (`MG-BEH-*`) — *requires torch*

| Rule | Severity | What It Catches |
|------|----------|-----------------|
| `MG-BEH-001` | varies | Known trigger phrase produces harmful output |
| `MG-BEH-002` | HIGH | Model fails to refuse known-dangerous triggers |

Tests 7 known trigger categories from Sleeper Agents and dataset poisoning research.
Includes: code backdoors, safety overrides, codeword triggers, temporal bypasses, and prompt injection hybrids. |

### 🗃️ Known Backdoor Registry (`MG-2024-*`)

A CVE-style database of documented backdoors and attack techniques:

| ID | Name | Severity |
|----|------|----------|
| `MG-2024-0001` | BadNets — Pixel-Level Trigger | HIGH |
| `MG-2024-0002` | TrojanNN — Weight-Based Injection | HIGH |
| `MG-2024-0003` | Sleeper Agent — Persistent LLM Backdoor | **CRITICAL** |
| `MG-2024-0004` | Dataset Poisoning — Vulnerable Code Gen | **CRITICAL** |
| `MG-2024-0005` | Malicious Pickle Payload (RCE) | **CRITICAL** |
| `MG-2024-0006` | Typo-Squatted Model on HuggingFace | HIGH |
| `MG-2024-0007` | Compromised Fine-Tuning Dataset | HIGH |
| `MG-2024-0008` | Quantization-Time Backdoor Injection | MEDIUM |
| `MG-2024-0009` | Embedding Layer Poisoning | **CRITICAL** |
| `MG-2024-0010` | LoRA Adapter Backdoor | HIGH |

[Contribute signatures →](signatures/registry.yaml)

---

## Architecture

```
ModelGuard
├── Weight Scanner       — Statistical anomaly detection in weight tensors
│   ├── NaN/Inf detection
│   ├── Extreme magnitude checking
│   ├── Z-score outlier analysis
│   ├── Embedding similarity anomaly detection
│   └── Sparsity validation
│
├── Activation Scanner   — Hidden state analysis (requires torch)
│   ├── Dead neuron detection
│   ├── Saturation analysis
│   ├── Inter-layer correlation
│   └── Distribution statistics
│
├── Behavioral Scanner   — Trigger phrase testing (requires torch)
│   ├── 7 known trigger categories
│   ├── Harmful output pattern detection
│   └── Safety bypass detection
│
├── Signature Matcher    — Known-backdoor database matching
│   ├── Exact hash matching (SHA-256)
│   ├── Partial hash matching
│   └── Structural fingerprint matching
│
├── Format Handlers      — Safe model file inspection
│   ├── Safetensors (safe by design)
│   ├── GGUF (llama.cpp format)
│   └── PyTorch (pickle RCE risk detection)
│
├── Hub Auditor          — Bulk HuggingFace model scanning
│
├── SBOM Generator       — CycloneDX 1.5 SBOMs for ML models
│
└── Vulnerability DB     — CVE-style database of known backdoors
```

---

## Use Cases

### CI/CD Pipeline

```yaml
# .github/workflows/model-scan.yml
- name: Scan model for backdoors
  run: |
    pip install modelguard
    modelguard scan --format json --fail-on high ./models/ > scan-results.json
```

### Pre-Deployment Gate

```bash
# Block deployment if any HIGH+ findings
modelguard scan --fail-on high model.safetensors && ./deploy.sh
```

### Model Registry Integration

```python
from modelguard import scan_model

result = scan_model("new_model.safetensors")
if not result.passed:
    raise ValueError(f"Model failed security scan: {result.to_json()}")
```

### HuggingFace Model Audit

```bash
# Audit a model before using it
modelguard scan --hub organization/model-name
modelguard sbom --hub organization/model-name
```

---

## Research Context

ModelGuard builds on established research in ML security:

- **BadNets** (Gu et al., 2017) — First systematic study of neural network backdoors
- **Neural Cleanse** (Wang et al., 2019) — Backdoor detection via trigger reconstruction
- **Trojaning Attack on Neural Networks** (Liu et al., 2018) — Weight modification attacks
- **Sleeper Agents** (Hubinger et al., 2024) — Persistent backdoors in LLMs through safety training
- **Poisoning Web-Scale Datasets** (Carlini et al., 2024) — Practical dataset poisoning at scale

The key insight: **weight-level inspection catches what behavioral testing misses**, and vice versa. ModelGuard combines both approaches.

---

## Installation

```bash
# Basic (safetensors support)
pip install modelguard

# With PyTorch support (for .bin files + activation scanning)
pip install modelguard[torch]

# With GGUF support (for llama.cpp models)
pip install modelguard[gguf]

# Everything
pip install modelguard[all]
```

---

## Development

```bash
git clone https://github.com/ob4cl/modelguard.git
cd modelguard
pip install -e ".[dev]"
pytest
```

---

## Roadmap

- [x] Weight anomaly scanning (6 rules)
- [x] Known backdoor signature database (10 entries)
- [x] Format handlers (safetensors, GGUF, PyTorch)
- [x] SBOM generation (CycloneDX 1.5)
- [x] Activation pattern scanning (5 rules)
- [x] Behavioral trigger testing (7 trigger categories)
- [x] HuggingFace Hub audit command
- [ ] Automated online signature database updates
- [ ] VSCode extension ("ModelGuard: Scan Model")
- [ ] Integration with HuggingFace's model cards
- [ ] Community signature contributions via PRs
- [ ] GitHub Action for CI scanning

---

## Contributing

This is a research project. Contributions welcome in these areas:

1. **New backdoor signatures** — Add entries to `known_backdoors.py`
2. **Detection rules** — New weight/activation analysis techniques
3. **Format support** — MLX, ONNX, CoreML handlers
4. **Research papers** — Cite ModelGuard in your ML security work

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Citation

If you use ModelGuard in your research:

```bibtex
@software{modelguard2024,
  author = {ModelGuard Contributors},
  title = {ModelGuard: ML Supply Chain Security Scanner},
  year = {2024},
  url = {https://github.com/ob4cl/modelguard}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*"The supply chain attack surface for ML models is the same as for software — we just haven't started treating it that way yet."*
