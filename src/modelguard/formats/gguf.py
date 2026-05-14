"""GGUF format handler.

GGUF is the standard format for llama.cpp and quantized models.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

# GGUF magic number
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian


def is_gguf(path: Path) -> bool:
    """Check if a file is a valid GGUF file."""
    try:
        with open(path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            return magic == GGUF_MAGIC
    except Exception:
        return False


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Read GGUF metadata without loading tensors.

    GGUF format v3:
    - 4 bytes: magic (0x46554747)
    - 4 bytes: version
    - 8 bytes: tensor_count
    - 8 bytes: metadata_kv_count
    - N metadata key-value pairs
    - Tensor info entries
    """
    try:
        with open(path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                return {"error": "Not a valid GGUF file"}

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            return {
                "format": "GGUF",
                "version": version,
                "tensor_count": tensor_count,
                "metadata_kv_count": kv_count,
            }
    except Exception as e:
        return {"error": str(e)}
