from pathlib import Path

import pytest

from ccvr.features import _validated_output_path, stream_extract_features
from ccvr.io import write_json


def test_archive_member_validation(tmp_path: Path) -> None:
    result = _validated_output_path(
        tmp_path, "OpenCLIP/dance/example.npy", "OpenCLIP"
    )
    assert result == (tmp_path / "OpenCLIP" / "dance" / "example.npy").resolve()
    assert (
        _validated_output_path(
            tmp_path, "OpenCLIP/.ipynb_checkpoints/x.json", "OpenCLIP"
        )
        is None
    )
    assert (
        _validated_output_path(tmp_path, "EVA-CLIP/dance/x.npy", "OpenCLIP")
        is None
    )


def test_feature_download_refuses_failed_data_gate(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    write_json(gate / "data_audit.json", {"accepted": False})
    with pytest.raises(RuntimeError, match="data gate"):
        stream_extract_features("OpenCLIP", tmp_path / "features", gate, 8.0)
