"""Safetensors format handler.

Safetensors is the recommended format for model distribution.
It's safe by design (no pickle RCE), but weights can still be poisoned.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read the safetensors header without loading tensors into memory.

    Safetensors format:
    - 8 bytes: header_size (little-endian u64)
    - header_size bytes: JSON header
    - remaining bytes: tensor data

    The header contains tensor names, dtypes, and shapes.
    """
    with open(path, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError("File too small to be a valid safetensors file")

        header_size = struct.unpack("<Q", header_size_bytes)[0]

        if header_size > 100 * 1024 * 1024:  # 100 MB header is suspicious
            raise ValueError(f"Header size {header_size} is suspiciously large")

        header_json = f.read(header_size)
        return json.loads(header_json)


def get_tensor_count(path: Path) -> int:
    """Get the number of tensors without loading them."""
    header = read_safetensors_header(path)
    return len(header) if isinstance(header, dict) else 0


def get_total_params(path: Path) -> int:
    """Estimate total parameter count from tensor shapes."""
    header = read_safetensors_header(path)
    total = 0
    for tensor_name, info in header.items():
        if isinstance(info, dict) and "shape" in info:
            shape = info["shape"]
            params = 1
            for dim in shape:
                params *= dim
            total += params
    return total


def list_tensor_names(path: Path) -> list[str]:
    """List all tensor names in the file."""
    header = read_safetensors_header(path)
    return list(header.keys()) if isinstance(header, dict) else []
