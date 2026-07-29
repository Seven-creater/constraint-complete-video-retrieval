import hashlib
import io
import urllib.request
from pathlib import Path

import pytest

from ccvr.features import (
    _ResumableHashingReader,
    _validated_output_path,
    stream_extract_features,
)
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


class _FakeResponse:
    def __init__(self, payload: bytes, status: int, headers: dict[str, str]) -> None:
        self._payload = io.BytesIO(payload)
        self.status = status
        self.headers = headers

    def read(self, size: int = -1) -> bytes:
        return self._payload.read(size)

    def close(self) -> None:
        self._payload.close()

    def getcode(self) -> int:
        return self.status


def test_feature_reader_resumes_at_exact_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[urllib.request.Request] = []
    responses = iter(
        (
            _FakeResponse(b"abc", 200, {}),
            _FakeResponse(b"def", 206, {"Content-Range": "bytes 3-5/6"}),
        )
    )

    def fake_open(request: urllib.request.Request, timeout: int) -> _FakeResponse:
        requests.append(request)
        return next(responses)

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    monkeypatch.setattr("ccvr.features.time.sleep", lambda _: None)
    with _ResumableHashingReader("https://example.test/archive", 6) as reader:
        assert reader.read(6) == b"abcdef"
        assert reader.reconnects == 1
        assert reader.digest.hexdigest() == hashlib.sha256(b"abcdef").hexdigest()
    assert requests[1].get_header("Range") == "bytes=3-"


def test_feature_reader_rejects_ignored_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            _FakeResponse(b"abc", 200, {}),
            _FakeResponse(b"def", 200, {}),
        )
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: next(responses),
    )
    monkeypatch.setattr("ccvr.features.time.sleep", lambda _: None)
    with _ResumableHashingReader("https://example.test/archive", 6) as reader:
        with pytest.raises(RuntimeError, match="exact feature byte range"):
            reader.read(6)
