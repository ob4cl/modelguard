"""ModelGuard — ML supply chain security scanner.

Detect backdoors, poisoned layers, and adversarial triggers in model weights.
Like `trivy` for neural networks.
"""

__version__ = "0.1.0"
__all__ = ["Scanner", "ScanResult", "Severity", "Finding", "scan_model", "scan_hub"]

from .scanner import Scanner, scan_hub, scan_model
from .types import Finding, ScanResult, Severity
