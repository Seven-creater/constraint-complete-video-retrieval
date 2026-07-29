from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .io import read_json, write_json
from .remote import huggingface_endpoint


DATASET_ID = "debby0527/MUVR"
ARCHIVES = {
    "OpenCLIP": "features/VLMs_1_15/OpenCLIP.tar",
    "EVA-CLIP": "features/VLMs_1_15/EVA-CLIP.tar",
}


class _ResumableHashingReader:
    def __init__(
        self,
        url: str,
        expected_size: int,
        timeout: int = 180,
        maximum_reconnects: int = 12,
    ) -> None:
        self.url = url
        self.expected_size = expected_size
        self.timeout = timeout
        self.maximum_reconnects = maximum_reconnects
        self.stream: BinaryIO | None = None
        self.digest = hashlib.sha256()
        self.bytes_read = 0
        self.reconnects = 0

    def __enter__(self) -> "_ResumableHashingReader":
        self._connect(reconnecting=False)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.stream is not None:
            self.stream.close()

    def _connect(self, reconnecting: bool) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        while True:
            if reconnecting:
                if self.reconnects >= self.maximum_reconnects:
                    raise RuntimeError(
                        "feature stream exhausted its reconnect budget"
                    )
                self.reconnects += 1
                time.sleep(min(2 ** (self.reconnects - 1), 30))
            headers = {
                "Accept-Encoding": "identity",
                "User-Agent": "ccvr/0.1",
            }
            if self.bytes_read:
                headers["Range"] = f"bytes={self.bytes_read}-"
            request = urllib.request.Request(self.url, headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout)
            except (OSError, TimeoutError, urllib.error.URLError):
                reconnecting = True
                continue
            status = int(getattr(response, "status", response.getcode()))
            if self.bytes_read:
                content_range = str(response.headers.get("Content-Range") or "")
                expected_prefix = f"bytes {self.bytes_read}-"
                if status != 206 or not content_range.startswith(expected_prefix):
                    response.close()
                    raise RuntimeError(
                        "server did not honor the exact feature byte range"
                    )
            elif status not in {200, 206}:
                response.close()
                raise RuntimeError(f"unexpected feature response status: {status}")
            self.stream = response
            return

    def read(self, size: int = -1) -> bytes:
        if self.stream is None:
            raise RuntimeError("feature stream is not open")
        remaining_archive = self.expected_size - self.bytes_read
        if remaining_archive <= 0:
            return b""
        requested = remaining_archive if size < 0 else min(size, remaining_archive)
        chunks: list[bytes] = []
        remaining = requested
        while remaining:
            try:
                value = self.stream.read(remaining)
            except (
                EOFError,
                OSError,
                TimeoutError,
                http.client.IncompleteRead,
                urllib.error.URLError,
            ):
                self._connect(reconnecting=True)
                continue
            if not value:
                self._connect(reconnecting=True)
                continue
            self.digest.update(value)
            self.bytes_read += len(value)
            remaining -= len(value)
            chunks.append(value)
        return b"".join(chunks)

    def readable(self) -> bool:
        return True


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "ccvr/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _archive_descriptor(backbone: str, revision: str) -> dict[str, Any]:
    if backbone not in ARCHIVES:
        raise ValueError(f"unsupported backbone: {backbone}")
    endpoint = huggingface_endpoint()
    items = _fetch_json(
        f"{endpoint}/api/datasets/{DATASET_ID}/tree/{revision}/"
        "features/VLMs_1_15?recursive=false&expand=true"
    )
    expected_path = ARCHIVES[backbone]
    for item in items:
        if item.get("path") == expected_path:
            return {
                "path": expected_path,
                "size": int(item["size"]),
                "sha256": str(item["lfs"]["oid"]),
                "commit": str(item["lastCommit"]["id"]),
            }
    raise FileNotFoundError(expected_path)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _validated_output_path(root: Path, member_name: str, backbone: str) -> Path | None:
    parts = PurePosixPath(member_name).parts
    if len(parts) < 3 or parts[0] != backbone:
        return None
    relative = Path(*parts[1:])
    if relative.suffix not in {".npy", ".pkl"}:
        return None
    destination = (root / backbone / relative).resolve()
    expected_root = (root / backbone).resolve()
    if expected_root not in destination.parents:
        raise ValueError(f"unsafe archive member: {member_name}")
    return destination


def stream_extract_features(
    backbone: str,
    output: Path,
    gate_dir: Path,
    maximum_storage_gb: float,
) -> dict[str, Any]:
    gate = read_json(gate_dir / "data_audit.json")
    if not gate.get("accepted"):
        raise RuntimeError("feature extraction is forbidden until the data gate passes")
    revision = str(gate["source_revision"])
    endpoint = huggingface_endpoint()
    descriptor = _archive_descriptor(backbone, revision)
    sentinel = output / backbone / "feature_lock.json"
    if sentinel.exists():
        existing = read_json(sentinel)
        if existing.get("archive_sha256") == descriptor["sha256"]:
            return existing
        raise RuntimeError(f"feature revision drift for {backbone}")
    projected = _directory_size(output.parent) + descriptor["size"]
    maximum_bytes = int(maximum_storage_gb * 1024**3)
    if projected > maximum_bytes:
        raise RuntimeError(
            f"projected storage {projected / 1024**3:.2f}GB exceeds "
            f"{maximum_storage_gb:.2f}GB"
        )

    url = (
        f"{endpoint}/datasets/{DATASET_ID}/resolve/{revision}/"
        f"{descriptor['path']}?download=true"
    )
    file_count = 0
    with _ResumableHashingReader(url, descriptor["size"]) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                destination = _validated_output_path(output, member.name, backbone)
                if destination is None:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read archive member: {member.name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".part")
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
                os.replace(temporary, destination)
                file_count += 1
        while reader.read(1024 * 1024):
            pass
    observed_hash = reader.digest.hexdigest()
    if reader.bytes_read != descriptor["size"]:
        raise RuntimeError(
            f"archive size mismatch: {reader.bytes_read} != {descriptor['size']}"
        )
    if observed_hash != descriptor["sha256"]:
        raise RuntimeError(
            f"archive checksum mismatch: {observed_hash} != {descriptor['sha256']}"
        )
    lock = {
        "backbone": backbone,
        "archive_path": descriptor["path"],
        "archive_size": descriptor["size"],
        "archive_sha256": descriptor["sha256"],
        "archive_commit": descriptor["commit"],
        "dataset_revision": revision,
        "endpoint": endpoint,
        "transport_reconnects": reader.reconnects,
        "extracted_files": file_count,
        "extracted_bytes": _directory_size(output / backbone),
        "data_gate_manifest_sha256": gate["manifest_sha256"],
    }
    write_json(sentinel, lock)
    return lock
