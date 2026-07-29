import json
from pathlib import Path

from ccvr.dataset import audit_dataset
from ccvr.io import sha256_file, write_json


def _write_partition(root: Path, partition: str, query_id: int) -> None:
    directory = root / partition
    directory.mkdir(parents=True)
    queries = [
        {
            "id": query_id,
            "video_name": f"q{query_id}",
            "topic_id": 1,
            "frames_path": f"/frames/q{query_id}",
        }
    ]
    videos = [
        {
            "id": value,
            "video_name": f"v{value}",
            "topic_id": 1,
            "frames_path": f"/frames/v{value}",
        }
        for value in range(1, 9)
    ]
    relationships = [
        {
            "query_id": query_id,
            "rel_lists": {"pos_all": {"id_list": list(range(1, 9))}},
        }
    ]
    tags = [
        {
            "query_id": query_id,
            "tag_lists": {
                "+a": {"id_list": [1, 2, 3, 4]},
                "-a": {"id_list": [5, 6, 7, 8]},
                "+b": {"id_list": [1, 2, 5, 6]},
                "-b": {"id_list": [3, 4, 7, 8]},
                "AND +a +b": {"id_list": [1, 2]},
                "OR +a +b": {"id_list": [1, 2, 3, 4, 5, 6]},
                "AND +a -b": {"id_list": [3, 4]},
            },
        }
    ]
    write_json(directory / "queries_en.json", queries)
    write_json(directory / "videos.json", videos)
    write_json(directory / "query_rel_lists.json", relationships)
    write_json(directory / "query_tag_lists.json", tags)


def test_dataset_audit_is_deterministic_and_detects_gate(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for index, partition in enumerate(("news", "region", "instance", "dance", "others")):
        _write_partition(raw, partition, 10 + index)
    files = {}
    for path in sorted(raw.rglob("*.json")):
        files[path.relative_to(raw).as_posix()] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    write_json(
        raw / "source_lock.json",
        {
            "dataset": "debby0527/MUVR",
            "requested_revision": "main",
            "resolved_revision": "deadbeef",
            "files": files,
        },
    )
    config = {
        "protocol_version": "test",
        "dataset": {
            "revision": "main",
            "split_fractions": [0.6, 0.2, 0.2],
            "minimum_positives": 2,
            "minimum_near_miss_negatives": 2,
            "minimum_prevalence": 0.01,
            "maximum_prevalence": 0.75,
        },
        "data_gate": {
            "minimum_eligible_queries": 10,
            "minimum_and_queries": 10,
            "minimum_or_queries": 5,
            "minimum_negated_queries": 5,
            "required_partitions": 5,
            "required_algebra_agreement": 1.0,
        },
    }
    first = audit_dataset(raw, tmp_path / "run1", config)
    second = audit_dataset(raw, tmp_path / "run2", config)
    assert first["accepted"]
    assert first["algebra_agreement"] == 1.0
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["split_topic_counts"] == {"train": 5}
    decision = json.loads((tmp_path / "run1" / "stage_decision.json").read_text())
    assert decision["status"] == "data_gate_passed"


def test_missing_atomic_literal_fails_cleanly(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for index, partition in enumerate(("news", "region", "instance", "dance", "others")):
        _write_partition(raw, partition, 20 + index)
    tag_path = raw / "news" / "query_tag_lists.json"
    tags = json.loads(tag_path.read_text(encoding="utf-8"))
    tags[0]["tag_lists"]["OR +a +missing"] = {"id_list": [1, 2, 3, 4]}
    write_json(tag_path, tags)
    files = {}
    for path in sorted(raw.rglob("*.json")):
        files[path.relative_to(raw).as_posix()] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    write_json(
        raw / "source_lock.json",
        {
            "dataset": "debby0527/MUVR",
            "requested_revision": "main",
            "resolved_revision": "deadbeef",
            "files": files,
        },
    )
    config = {
        "protocol_version": "test",
        "dataset": {
            "revision": "main",
            "split_fractions": [0.6, 0.2, 0.2],
            "minimum_positives": 2,
            "minimum_near_miss_negatives": 2,
            "minimum_prevalence": 0.01,
            "maximum_prevalence": 0.75,
        },
        "data_gate": {
            "minimum_eligible_queries": 0,
            "minimum_and_queries": 0,
            "minimum_or_queries": 0,
            "minimum_negated_queries": 0,
            "required_partitions": 0,
            "required_algebra_agreement": 1.0,
        },
    }
    report = audit_dataset(raw, tmp_path / "run", config)
    assert not report["accepted"]
    assert report["unresolvable_composites"] == 1
    assert report["status"] == "public_dataset_gate_failed"


def test_unpublished_opposite_polarity_is_inferred_from_base_domain(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    for index, partition in enumerate(("news", "region", "instance", "dance", "others")):
        _write_partition(raw, partition, 30 + index)
    for partition in ("news", "region", "instance", "dance", "others"):
        tag_path = raw / partition / "query_tag_lists.json"
        tags = json.loads(tag_path.read_text(encoding="utf-8"))
        del tags[0]["tag_lists"]["-a"]
        write_json(tag_path, tags)
    files = {}
    for path in sorted(raw.rglob("*.json")):
        files[path.relative_to(raw).as_posix()] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    write_json(
        raw / "source_lock.json",
        {
            "dataset": "debby0527/MUVR",
            "requested_revision": "main",
            "resolved_revision": "deadbeef",
            "files": files,
        },
    )
    config = {
        "protocol_version": "test",
        "dataset": {
            "revision": "main",
            "split_fractions": [0.6, 0.2, 0.2],
            "minimum_positives": 2,
            "minimum_near_miss_negatives": 2,
            "minimum_prevalence": 0.01,
            "maximum_prevalence": 0.75,
        },
        "data_gate": {
            "minimum_eligible_queries": 0,
            "minimum_and_queries": 0,
            "minimum_or_queries": 0,
            "minimum_negated_queries": 0,
            "required_partitions": 0,
            "required_algebra_agreement": 1.0,
        },
    }
    report = audit_dataset(raw, tmp_path / "run", config)
    assert report["accepted"]
    assert report["unresolvable_composites"] == 0
    assert report["algebra_agreement"] == 1.0
