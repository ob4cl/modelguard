"""PyTorch checkpoint format handler.

WARNING: PyTorch .bin/.pt files can contain pickle payloads (RCE risk).
Always prefer safetensors. This handler provides safe(r) inspection utilities.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any


def is_torch_checkpoint(path: Path) -> bool:
    """Check if a file is a PyTorch checkpoint (zip-based)."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # PyTorch checkpoints contain specific files
            names = zf.namelist()
            return any("data.pkl" in n or "version" in n for n in names)
    except (zipfile.BadZipFile, Exception):
        return False


def check_pickle_risk(path: Path) -> dict[str, Any]:
    """Check a PyTorch file for pickle-based RCE risk.

    Returns risk assessment and recommendations.
    """
    risks = []
    is_safe = True

    if path.suffix in (".bin", ".pt", ".pth"):
        # These extensions often indicate pickle format
        if path.suffix == ".bin":
            risks.append(
                "File extension .bin — likely contains pickle serialized data. "
                "Loading with torch.load() can execute arbitrary code."
            )
            is_safe = False
        else:
            risks.append(
                f"File extension {path.suffix} — may contain pickle data. "
                "Use safetensors instead."
            )
            is_safe = False

    # Check if it's actually a zip (modern torch.save uses zip)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            has_data_pkl = any("data.pkl" in n for n in zf.namelist())
            if has_data_pkl:
                risks.append(
                    "Contains data.pkl — pickle-serialized tensor data. "
                    "RCE risk on load."
                )
                is_safe = False
    except zipfile.BadZipFile:
        # Not a zip — likely pure pickle, even riskier
        risks.append(
            "Not a zip archive — likely pure pickle format. "
            "EXTREME RCE risk on load."
        )
        is_safe = False

    return {
        "path": str(path),
        "is_safe": is_safe,
        "risks": risks,
        "recommendation": (
            "Convert to safetensors: use 'modelguard convert' "
            "or huggingface_hub's convert tool."
            if not is_safe
            else "File appears safe to load."
        ),
    }
