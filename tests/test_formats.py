"""Tests for format handlers."""

import json
import struct

from modelguard.formats.pytorch import check_pickle_risk
from modelguard.formats.safetensors import (
    get_tensor_count,
    get_total_params,
    list_tensor_names,
    read_safetensors_header,
)


class TestSafetensorsFormat:
    def test_invalid_file_raises(self, tmp_path):
        """Reading a non-safetensors file should raise."""
        bad_file = tmp_path / "not_st.bin"
        bad_file.write_bytes(b"not a safetensors file at all")

        with __import__("pytest").raises(Exception):
            read_safetensors_header(bad_file)

    def test_too_small_file_raises(self, tmp_path):
        """A file smaller than 8 bytes can't be safetensors."""
        tiny = tmp_path / "tiny.bin"
        tiny.write_bytes(b"1234567")

        with __import__("pytest").raises(ValueError):
            read_safetensors_header(tiny)

    def test_valid_header_reads_correctly(self, tmp_path):
        """Smoke test with well-formed safetensors header."""
        header = {"weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
        header_json = json.dumps(header).encode()
        header_size = len(header_json)

        st_file = tmp_path / "test.safetensors"
        with open(st_file, "wb") as f:
            f.write(struct.pack("<Q", header_size))
            f.write(header_json)
            f.write(b"\x00" * 24)  # 2*3*4 bytes of fake float32 data

        result = read_safetensors_header(st_file)
        assert "weight" in result
        assert result["weight"]["shape"] == [2, 3]

    def test_tensor_count(self, tmp_path):
        header = {
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        }
        header_json = json.dumps(header).encode()

        st_file = tmp_path / "multi.safetensors"
        with open(st_file, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 8)

        assert get_tensor_count(st_file) == 2

    def test_total_params(self, tmp_path):
        header = {
            "w": {"dtype": "F32", "shape": [100, 50], "data_offsets": [0, 20000]},
        }
        header_json = json.dumps(header).encode()

        st_file = tmp_path / "params.safetensors"
        with open(st_file, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 20000)

        assert get_total_params(st_file) == 5000

    def test_list_tensor_names(self, tmp_path):
        header = {
            "model.layers.0.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "model.layers.0.bias": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        }
        header_json = json.dumps(header).encode()

        st_file = tmp_path / "names.safetensors"
        with open(st_file, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 8)

        names = list_tensor_names(st_file)
        assert "model.layers.0.weight" in names
        assert len(names) == 2


class TestPyTorchFormat:
    def test_bin_files_flagged_as_risky(self, tmp_path):
        """Any .bin file should be flagged for pickle risk."""
        bin_file = tmp_path / "model.bin"
        bin_file.write_bytes(b"pretend this is pickle data")

        result = check_pickle_risk(bin_file)
        assert result["is_safe"] is False
        assert len(result["risks"]) > 0

    def test_pt_files_flagged(self, tmp_path):
        pt_file = tmp_path / "model.pt"
        pt_file.write_bytes(b"torch data")

        result = check_pickle_risk(pt_file)
        assert result["is_safe"] is False
        assert "recommendation" in result

    def test_pth_files_flagged(self, tmp_path):
        pth_file = tmp_path / "model.pth"
        pth_file.write_bytes(b"weights")

        result = check_pickle_risk(pth_file)
        assert result["is_safe"] is False
