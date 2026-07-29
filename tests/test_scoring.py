import pickle
from pathlib import Path

import numpy as np

from ccvr.io import write_json, write_jsonl
from ccvr.scoring import _method_scores, _normalize, score_backbone


def test_method_scores_are_order_invariant_for_logic() -> None:
    row = {
        "operator": "AND",
        "literals": [
            {"tag": "a", "polarity": "+"},
            {"tag": "b", "polarity": "-"},
        ],
    }
    base = np.asarray([0.2, 0.4])
    raw = np.asarray([[0.8, 0.1], [0.2, 0.9]])
    satisfaction = np.asarray([[0.9, 0.3], [0.7, 0.2]])
    concat = np.asarray([0.5, 0.5])
    first = _method_scores(row, base, raw, satisfaction, concat)
    swapped_row = {**row, "literals": list(reversed(row["literals"]))}
    second = _method_scores(
        swapped_row, base, raw[::-1], satisfaction[::-1], concat
    )
    assert np.allclose(first["fuzzy_logic"], second["fuzzy_logic"])
    for name in first:
        if name.startswith("prototype_"):
            assert np.allclose(first[name], second[name])


def test_normalize_rejects_zero_vectors() -> None:
    try:
        _normalize(np.zeros(3))
    except ValueError as exc:
        assert "zero-norm" in str(exc)
    else:
        raise AssertionError("zero vector was accepted")


def test_score_backbone_runs_end_to_end_on_frozen_features(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    features = tmp_path / "features" / "OpenCLIP"
    run = tmp_path / "run"
    manifest_rows = []
    partitions = ("news", "region", "instance", "dance", "others")
    for partition_index, partition in enumerate(partitions):
        queries = []
        videos = []
        for split_index, split in enumerate(("dev", "test")):
            query_id = 1000 + partition_index * 10 + split_index
            topic_id = split_index + 1
            query_name = f"q_{partition}_{split}"
            queries.append(
                {
                    "id": query_id,
                    "topic_id": topic_id,
                    "frames_path": f"/frames/{query_name}",
                }
            )
            videos.append(
                {
                    "id": query_id,
                    "topic_id": topic_id,
                    "frames_path": f"/frames/{query_name}",
                }
            )
            query_feature = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            partition_features = features / partition
            partition_features.mkdir(parents=True, exist_ok=True)
            np.save(partition_features / f"{query_name}.npy", query_feature)
            with (partition_features / f"{query_name}_query_dict.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "query_prompt": np.asarray([1.0, 0.0], dtype=np.float32),
                        "tags": {
                            "a": np.asarray([1.0, 0.0], dtype=np.float32),
                            "b": np.asarray([0.0, 1.0], dtype=np.float32),
                        },
                    },
                    handle,
                )
            candidate_ids = []
            for candidate_offset, vector in enumerate(
                (
                    (1.0, 0.0),
                    (0.8, 0.2),
                    (0.2, 0.8),
                    (0.0, 1.0),
                ),
                start=1,
            ):
                candidate_id = (
                    partition_index * 100 + split_index * 10 + candidate_offset
                )
                candidate_name = f"v_{partition}_{split}_{candidate_offset}"
                candidate_ids.append(candidate_id)
                videos.append(
                    {
                        "id": candidate_id,
                        "topic_id": topic_id,
                        "frames_path": f"/frames/{candidate_name}",
                    }
                )
                np.save(
                    partition_features / f"{candidate_name}.npy",
                    np.asarray([vector, vector], dtype=np.float32),
                )
            common = {
                "source_query_id": query_id,
                "query_video_name": query_name,
                "topic_id": f"{partition}:{topic_id}",
                "partition": partition,
                "split": split,
                "partial_negatives": candidate_ids[1:],
                "source_revision": "frozen",
            }
            manifest_rows.extend(
                [
                    {
                        **common,
                        "query_id": f"{partition}:{query_id}:atom",
                        "operator": "ATOM",
                        "expression": "+a",
                        "literals": [{"tag": "a", "polarity": "+"}],
                        "positives": candidate_ids[:2],
                        "eligible_for_diagnostic": False,
                    },
                    {
                        **common,
                        "query_id": f"{partition}:{query_id}:and",
                        "operator": "AND",
                        "expression": "AND +a -b",
                        "literals": [
                            {"tag": "a", "polarity": "+"},
                            {"tag": "b", "polarity": "-"},
                        ],
                        "positives": candidate_ids[:1],
                        "eligible_for_diagnostic": True,
                    },
                ]
            )
        write_json(raw / partition / "queries_en.json", queries)
        write_json(raw / partition / "videos.json", videos)
    manifest_path = run / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    write_json(
        run / "data_audit.json",
        {"accepted": True, "manifest_sha256": "manifest-hash"},
    )
    write_json(
        features / "feature_lock.json",
        {"data_gate_manifest_sha256": "manifest-hash"},
    )
    config = {
        "budget": {"maximum_gpu_hours": 4},
        "problem_gate": {
            "minimum_compositional_map_drop": 0.0,
            "minimum_constraint_violation_at_10": 0.0,
            "minimum_supporting_partitions": 0,
            "simple_solution_maximum_violation_at_10": -1.0,
            "simple_solution_minimum_oracle_relative_map": 2.0,
            "bootstrap_samples": 10,
            "bootstrap_seed": 1,
        },
        "prototype_gate": {
            "minimum_absolute_map_gain": -1.0,
            "minimum_relative_violation_reduction": -1.0,
            "minimum_supporting_partitions": 0,
            "maximum_atomic_map_drop": 1.0,
            "seeds": [1, 2, 3],
            "maximum_parameters": 2000000,
        },
    }
    summary = score_backbone(
        "OpenCLIP",
        raw,
        manifest_path,
        features,
        run / "openclip",
        config,
        "cpu",
    )
    assert summary["runtime"]["metric_rows"] > 0
    assert (run / "openclip" / "per_query_metrics.jsonl").exists()
    assert summary["prototype_gate"]["parameter_count"] == 2
