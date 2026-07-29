from __future__ import annotations

import math
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .budget import ExperimentBudget
from .io import read_json, read_jsonl, write_json
from .metrics import (
    aggregate_metrics,
    bootstrap_mean_ci,
    monotone_score,
    rank_metrics,
)


BASELINE_METHODS = (
    "ignore_conditions",
    "official_sum",
    "condition_sum",
    "fuzzy_logic",
    "text_concat",
)
GRID_BASE_WEIGHTS = (0.25, 0.5, 0.75)
GRID_TAUS = (0.02, 0.05, 0.1, 0.2)


class _JsonlSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.handle: Any = None
        self.count = 0

    def __enter__(self) -> "_JsonlSink":
        import json

        self._json = json
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temporary.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(
            self._json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        self.handle.write("\n")
        self.count += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.handle.close()
        if exc_type is None:
            os.replace(self.temporary, self.path)


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    array = array.reshape(-1, array.shape[-1])
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("zero-norm feature vector")
    return array / norm


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or "query_prompt" not in value or "tags" not in value:
        raise ValueError(f"unexpected query feature payload: {path}")
    return value


def _feature_path(feature_root: Path, partition: str, video_name: str) -> Path:
    path = feature_root / partition / f"{video_name}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _query_payload_path(
    feature_root: Path, partition: str, video_name: str
) -> Path:
    path = feature_root / partition / f"{video_name}_query_dict.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _partition_metadata(
    raw: Path,
    partition: str,
) -> tuple[np.ndarray, list[str], dict[int, str]]:
    videos = read_json(raw / partition / "videos.json")
    queries = read_json(raw / partition / "queries_en.json")
    query_ids = {int(row["id"]) for row in queries}
    candidate_ids: list[int] = []
    candidate_names: list[str] = []
    names_by_id: dict[int, str] = {}
    for video in videos:
        video_id = int(video["id"])
        name = Path(str(video["frames_path"])).name
        names_by_id[video_id] = name
        if video_id not in query_ids:
            candidate_ids.append(video_id)
            candidate_names.append(name)
    return np.asarray(candidate_ids, dtype=np.int64), candidate_names, names_by_id


def _load_partition_features(
    feature_root: Path,
    partition: str,
    candidate_names: list[str],
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for name in candidate_names:
        array = _normalize(np.load(_feature_path(feature_root, partition, name)))
        if expected_shape is None:
            expected_shape = array.shape
        if array.shape != expected_shape:
            raise ValueError(
                f"inconsistent feature shape in {partition}: "
                f"{array.shape} != {expected_shape}"
            )
        arrays.append(array)
    if not arrays:
        raise ValueError(f"partition has no database features: {partition}")
    return np.stack(arrays)


def _torch_module(device: str) -> Any | None:
    if device == "cpu":
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA scoring requires torch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch


def _similarities(
    query_vectors: np.ndarray,
    target_features: np.ndarray,
    device: str,
    chunk_size: int = 2048,
) -> np.ndarray:
    query = _normalize(query_vectors)
    output: list[np.ndarray] = []
    torch = _torch_module(device)
    if torch is None:
        for start in range(0, len(target_features), chunk_size):
            target = target_features[start : start + chunk_size]
            values = np.einsum("qd,nfd->qnf", query, target, optimize=True)
            output.append(values.max(axis=2))
    else:
        query_tensor = torch.from_numpy(query).to(device)
        with torch.inference_mode():
            for start in range(0, len(target_features), chunk_size):
                target = torch.from_numpy(
                    target_features[start : start + chunk_size]
                ).to(device)
                values = torch.einsum("qd,nfd->qnf", query_tensor, target)
                output.append(values.amax(dim=2).cpu().numpy())
                del target, values
    return np.concatenate(output, axis=1)


def _visual_similarity(
    query_frames: np.ndarray,
    target_features: np.ndarray,
    device: str,
    chunk_size: int = 2048,
) -> np.ndarray:
    query = _normalize(query_frames)
    output: list[np.ndarray] = []
    torch = _torch_module(device)
    if torch is None:
        for start in range(0, len(target_features), chunk_size):
            target = target_features[start : start + chunk_size]
            values = np.einsum("qd,nfd->qnf", query, target, optimize=True)
            output.append(values.max(axis=(0, 2)))
    else:
        query_tensor = torch.from_numpy(query).to(device)
        with torch.inference_mode():
            for start in range(0, len(target_features), chunk_size):
                target = torch.from_numpy(
                    target_features[start : start + chunk_size]
                ).to(device)
                values = torch.einsum("qd,nfd->qnf", query_tensor, target)
                output.append(values.amax(dim=(0, 2)).cpu().numpy())
                del target, values
    return np.concatenate(output)


def _literal_satisfaction(
    row: dict[str, Any],
    raw_tag_scores: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    raw_values: list[np.ndarray] = []
    satisfaction: list[np.ndarray] = []
    for literal in row["literals"]:
        tag = str(literal["tag"])
        if tag not in raw_tag_scores:
            raise KeyError(f"query feature payload lacks tag: {tag}")
        raw = raw_tag_scores[tag]
        normalized = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
        raw_values.append(raw)
        satisfaction.append(
            normalized if literal["polarity"] == "+" else 1.0 - normalized
        )
    return np.stack(raw_values), np.stack(satisfaction)


def _method_scores(
    row: dict[str, Any],
    base_scores: np.ndarray,
    raw_literals: np.ndarray,
    satisfaction: np.ndarray,
    concat_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    signs = np.asarray(
        [1.0 if literal["polarity"] == "+" else -1.0 for literal in row["literals"]]
    )[:, None]
    base_normalized = np.clip((base_scores + 1.0) / 2.0, 0.0, 1.0)
    operator = str(row["operator"])
    if operator == "AND":
        hard_logic = satisfaction.min(axis=0)
    elif operator == "OR":
        hard_logic = satisfaction.max(axis=0)
    else:
        hard_logic = satisfaction[0]
    scores = {
        "ignore_conditions": base_scores,
        "official_sum": base_scores + 0.3 * np.sum(signs * raw_literals, axis=0),
        "condition_sum": 0.5 * base_normalized + 0.5 * satisfaction.mean(axis=0),
        "fuzzy_logic": 0.5 * base_normalized + 0.5 * hard_logic,
        "text_concat": concat_scores,
    }
    for base_weight in GRID_BASE_WEIGHTS:
        for tau in GRID_TAUS:
            name = f"prototype_bw{base_weight:.2f}_tau{tau:.2f}"
            scores[name] = monotone_score(
                base_scores,
                satisfaction,
                operator,
                base_weight=base_weight,
                tau=tau,
            )
    return scores


def _combined_query_vectors(
    rows: list[dict[str, Any]],
    query_text: np.ndarray,
    tag_vectors: dict[str, np.ndarray],
) -> np.ndarray:
    base = _normalize(query_text)[0]
    output: list[np.ndarray] = []
    for row in rows:
        additions = []
        for literal in row["literals"]:
            vector = _normalize(tag_vectors[str(literal["tag"])])[0]
            additions.append(
                vector if literal["polarity"] == "+" else -vector
            )
        combined = base + np.mean(additions, axis=0)
        output.append(_normalize(combined)[0])
    return np.stack(output)


def _labels(candidate_ids: np.ndarray, positives: Iterable[int]) -> np.ndarray:
    positive_set = set(int(value) for value in positives)
    return np.asarray(
        [int(candidate_id) in positive_set for candidate_id in candidate_ids],
        dtype=np.int8,
    )


def _score_backbone_impl(
    backbone: str,
    raw: Path,
    manifest_path: Path,
    feature_root: Path,
    output: Path,
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    gate_dir = manifest_path.parent
    data_gate = read_json(gate_dir / "data_audit.json")
    if not data_gate.get("accepted"):
        raise RuntimeError("scoring is forbidden until the data gate passes")
    feature_lock = read_json(feature_root / "feature_lock.json")
    if feature_lock.get("data_gate_manifest_sha256") != data_gate["manifest_sha256"]:
        raise RuntimeError("feature lock was built for a different manifest")
    if backbone == "EVA-CLIP":
        openclip_summary = read_json(output.parent / "openclip" / "summary.json")
        if not openclip_summary["problem_gate"]["accepted"]:
            raise RuntimeError("EVA-CLIP is forbidden until OpenCLIP passes")

    manifest = read_jsonl(manifest_path)
    selected = [
        row
        for row in manifest
        if row["split"] in {"dev", "test"}
        and (
            row["operator"] == "ATOM"
            or bool(row["eligible_for_diagnostic"])
        )
    ]
    rows_by_partition_source: dict[
        tuple[str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in selected:
        rows_by_partition_source[
            (str(row["partition"]), int(row["source_query_id"]))
        ].append(row)

    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "per_query_metrics.jsonl"
    scores_path = output / "top10_scores.jsonl"
    metric_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with _JsonlSink(metrics_path) as metric_sink, _JsonlSink(scores_path) as score_sink:
        for partition in ("news", "region", "instance", "dance", "others"):
            candidate_ids, candidate_names, _ = _partition_metadata(raw, partition)
            target_features = _load_partition_features(
                feature_root, partition, candidate_names
            )
            source_groups = sorted(
                (
                    (source_id, rows)
                    for (row_partition, source_id), rows
                    in rows_by_partition_source.items()
                    if row_partition == partition
                ),
                key=lambda item: item[0],
            )
            for source_id, rows in source_groups:
                query_video_name = str(rows[0]["query_video_name"])
                query_visual = np.load(
                    _feature_path(feature_root, partition, query_video_name)
                )
                payload = _load_pickle(
                    _query_payload_path(feature_root, partition, query_video_name)
                )
                query_text = _normalize(np.asarray(payload["query_prompt"]))[0]
                tag_vectors = {
                    str(tag): _normalize(np.asarray(vector))[0]
                    for tag, vector in dict(payload["tags"]).items()
                }
                required_tags = sorted(
                    {
                        str(literal["tag"])
                        for row in rows
                        for literal in row["literals"]
                    }
                )
                missing_tags = sorted(set(required_tags) - set(tag_vectors))
                if missing_tags:
                    raise KeyError(
                        f"{partition}:{source_id} lacks tag embeddings: {missing_tags}"
                    )
                visual_scores = _visual_similarity(
                    query_visual, target_features, device
                )
                text_scores = _similarities(
                    query_text, target_features, device
                )[0]
                base_scores = (visual_scores + text_scores) / 2.0
                tag_matrix = np.stack([tag_vectors[tag] for tag in required_tags])
                tag_score_matrix = _similarities(
                    tag_matrix, target_features, device
                )
                raw_tag_scores = {
                    tag: tag_score_matrix[index]
                    for index, tag in enumerate(required_tags)
                }
                combined_vectors = _combined_query_vectors(
                    rows, query_text, tag_vectors
                )
                combined_text_scores = _similarities(
                    combined_vectors, target_features, device
                )
                concat_scores = (
                    combined_text_scores + visual_scores[None, :]
                ) / 2.0

                for row_index, row in enumerate(rows):
                    labels = _labels(candidate_ids, row["positives"])
                    if labels.sum() == 0:
                        if row["operator"] == "ATOM":
                            continue
                        raise ValueError(f"eligible query has no database positives: {row['query_id']}")
                    raw_literals, satisfaction = _literal_satisfaction(
                        row, raw_tag_scores
                    )
                    methods = _method_scores(
                        row,
                        base_scores,
                        raw_literals,
                        satisfaction,
                        concat_scores[row_index],
                    )
                    methods["oracle"] = labels.astype(np.float64) + base_scores * 1e-6
                    near_misses = set(int(value) for value in row["partial_negatives"])
                    negated = any(
                        literal["polarity"] == "-" for literal in row["literals"]
                    )
                    for method, method_values in methods.items():
                        result = rank_metrics(
                            method_values,
                            labels,
                            candidate_ids,
                            near_misses,
                        )
                        metric_row = {
                            "query_id": row["query_id"],
                            "source_query_id": source_id,
                            "topic_id": row["topic_id"],
                            "partition": partition,
                            "split": row["split"],
                            "operator": row["operator"],
                            "contains_negation": negated,
                            "method": method,
                            **result,
                        }
                        metric_rows.append(metric_row)
                        metric_sink.write(metric_row)
                        if (
                            row["split"] == "test"
                            and row["eligible_for_diagnostic"]
                            and method in BASELINE_METHODS
                        ):
                            order = np.argsort(-method_values, kind="stable")[:10]
                            logic_values = (
                                satisfaction.min(axis=0)
                                if row["operator"] == "AND"
                                else satisfaction.max(axis=0)
                                if row["operator"] == "OR"
                                else satisfaction[0]
                            )
                            for candidate_index in order:
                                candidate_id = int(candidate_ids[candidate_index])
                                score_sink.write(
                                    {
                                        "query_id": row["query_id"],
                                        "candidate_id": candidate_id,
                                        "backbone": backbone,
                                        "method": method,
                                        "base_score": float(base_scores[candidate_index]),
                                        "literal_scores": [
                                            float(values[candidate_index])
                                            for values in raw_literals
                                        ],
                                        "logic_score": float(
                                            logic_values[candidate_index]
                                        ),
                                        "label": int(labels[candidate_index]),
                                        "violation_type": (
                                            "none"
                                            if labels[candidate_index]
                                            else "near_miss"
                                            if candidate_id in near_misses
                                            else "base_irrelevant"
                                        ),
                                    }
                                )
            del target_features

    summary = analyze_backbone(metric_rows, backbone, config)
    summary["runtime"] = {
        "wall_seconds": time.monotonic() - started,
        "device": device,
        "gpu_hours": (
            (time.monotonic() - started) / 3600.0 if device != "cpu" else 0.0
        ),
        "metric_rows": len(metric_rows),
    }
    write_json(output / "summary.json", summary)
    return summary


def score_backbone(
    backbone: str,
    raw: Path,
    manifest_path: Path,
    feature_root: Path,
    output: Path,
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    maximum_gpu_hours = float(config["budget"]["maximum_gpu_hours"])
    with ExperimentBudget(
        run_dir=output.parent,
        maximum_gpu_hours=maximum_gpu_hours,
        stage=f"score_{backbone}",
    ):
        return _score_backbone_impl(
            backbone=backbone,
            raw=raw,
            manifest_path=manifest_path,
            feature_root=feature_root,
            output=output,
            config=config,
            device=device,
        )


def _mean(rows: list[dict[str, Any]], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows])) if rows else math.nan


def _method_rows(
    rows: list[dict[str, Any]],
    method: str,
    *,
    split: str,
    atomic: bool,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["method"] == method
        and row["split"] == split
        and (row["operator"] == "ATOM") == atomic
    ]


def _stratum(
    rows: list[dict[str, Any]],
    *,
    partition: str | None = None,
    operator: str | None = None,
    negated: bool | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (partition is None or row["partition"] == partition)
        and (operator is None or row["operator"] == operator)
        and (negated is None or bool(row["contains_negation"]) == negated)
    ]


def analyze_backbone(
    metric_rows: list[dict[str, Any]],
    backbone: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    problem = config["problem_gate"]
    method_summaries: dict[str, dict[str, float]] = {}
    for method in BASELINE_METHODS:
        composite = _method_rows(metric_rows, method, split="test", atomic=False)
        atomic = _method_rows(metric_rows, method, split="test", atomic=True)
        method_summaries[method] = {
            "composite_map": _mean(composite, "map"),
            "atomic_map": _mean(atomic, "map"),
            "compositional_map_drop": _mean(atomic, "map")
            - _mean(composite, "map"),
            "constraint_violation_at_10": _mean(
                composite, "constraint_violation_at_10"
            ),
            "ndcg_at_10": _mean(composite, "ndcg_at_10"),
            "recall_at_200": _mean(composite, "recall_at_200"),
        }
    strongest = max(
        BASELINE_METHODS,
        key=lambda method: method_summaries[method]["composite_map"],
    )
    strongest_test = _method_rows(
        metric_rows, strongest, split="test", atomic=False
    )
    strongest_atomic = _method_rows(
        metric_rows, strongest, split="test", atomic=True
    )
    partition_results: dict[str, dict[str, float | bool]] = {}
    supporting_partitions = 0
    for partition in ("news", "region", "instance", "dance", "others"):
        composite = _stratum(strongest_test, partition=partition)
        atomic = _stratum(strongest_atomic, partition=partition)
        gap = _mean(atomic, "map") - _mean(composite, "map")
        violation = _mean(composite, "constraint_violation_at_10")
        supports = (
            gap >= float(problem["minimum_compositional_map_drop"])
            and violation >= float(problem["minimum_constraint_violation_at_10"])
        )
        supporting_partitions += int(supports)
        partition_results[partition] = {
            "compositional_map_drop": gap,
            "constraint_violation_at_10": violation,
            "supports_problem": supports,
        }
    topic_atomic: dict[str, list[float]] = defaultdict(list)
    topic_composite: dict[str, list[float]] = defaultdict(list)
    for row in strongest_atomic:
        topic_atomic[str(row["topic_id"])].append(float(row["map"]))
    for row in strongest_test:
        topic_composite[str(row["topic_id"])].append(float(row["map"]))
    topic_differences = [
        float(np.mean(topic_atomic[topic]) - np.mean(topic_composite[topic]))
        for topic in sorted(set(topic_atomic) & set(topic_composite))
    ]
    ci_low, ci_high = bootstrap_mean_ci(
        topic_differences,
        int(problem["bootstrap_samples"]),
        int(problem["bootstrap_seed"]),
    )
    and_rows = _stratum(strongest_test, operator="AND")
    negated_rows = _stratum(strongest_test, negated=True)
    atomic_mean = _mean(strongest_atomic, "map")
    strata = {
        "AND": {
            "compositional_map_drop": atomic_mean - _mean(and_rows, "map"),
            "constraint_violation_at_10": _mean(
                and_rows, "constraint_violation_at_10"
            ),
        },
        "NOT": {
            "compositional_map_drop": atomic_mean - _mean(negated_rows, "map"),
            "constraint_violation_at_10": _mean(
                negated_rows, "constraint_violation_at_10"
            ),
        },
    }
    required_gap = float(problem["minimum_compositional_map_drop"])
    required_violation = float(problem["minimum_constraint_violation_at_10"])
    strata_pass = all(
        values["compositional_map_drop"] >= required_gap
        and values["constraint_violation_at_10"] >= required_violation
        for values in strata.values()
    )
    simple_solution = any(
        values["constraint_violation_at_10"]
        <= float(problem["simple_solution_maximum_violation_at_10"])
        and values["composite_map"]
        >= float(problem["simple_solution_minimum_oracle_relative_map"])
        for values in method_summaries.values()
    )
    problem_accepted = (
        not simple_solution
        and method_summaries[strongest]["compositional_map_drop"] >= required_gap
        and method_summaries[strongest]["constraint_violation_at_10"]
        >= required_violation
        and supporting_partitions
        >= int(problem["minimum_supporting_partitions"])
        and strata_pass
        and ci_low > 0.0
    )
    prototype = _select_prototype(metric_rows, strongest, config)
    return {
        "backbone": backbone,
        "method_summaries": method_summaries,
        "strongest_simple_method": strongest,
        "partition_results": partition_results,
        "strata": strata,
        "bootstrap_compositional_drop_ci95": [ci_low, ci_high],
        "simple_solution": simple_solution,
        "problem_gate": {
            "accepted": problem_accepted,
            "supporting_partitions": supporting_partitions,
            "required_partitions": int(problem["minimum_supporting_partitions"]),
        },
        "prototype_gate": prototype,
    }


def _select_prototype(
    rows: list[dict[str, Any]],
    strongest_simple: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    prototype_config = config["prototype_gate"]
    grid_methods = sorted(
        {
            str(row["method"])
            for row in rows
            if str(row["method"]).startswith("prototype_")
        }
    )
    dev_topics = sorted(
        {
            str(row["topic_id"])
            for row in rows
            if row["split"] == "dev" and row["operator"] != "ATOM"
        }
    )
    baseline_test = _method_rows(
        rows, strongest_simple, split="test", atomic=False
    )
    baseline_atomic = _method_rows(
        rows, strongest_simple, split="test", atomic=True
    )
    baseline_map = _mean(baseline_test, "map")
    baseline_violation = _mean(baseline_test, "constraint_violation_at_10")
    baseline_atomic_map = _mean(baseline_atomic, "map")
    seed_results: list[dict[str, Any]] = []
    for seed in prototype_config["seeds"]:
        generator = np.random.default_rng(int(seed))
        sample_count = max(1, math.ceil(len(dev_topics) * 0.8))
        sampled_topics = set(
            generator.choice(dev_topics, size=sample_count, replace=False).tolist()
        )
        best_method = max(
            grid_methods,
            key=lambda method: _mean(
                [
                    row
                    for row in _method_rows(
                        rows, method, split="dev", atomic=False
                    )
                    if str(row["topic_id"]) in sampled_topics
                ],
                "map",
            ),
        )
        test = _method_rows(rows, best_method, split="test", atomic=False)
        atomic = _method_rows(rows, best_method, split="test", atomic=True)
        test_map = _mean(test, "map")
        violation = _mean(test, "constraint_violation_at_10")
        gain = test_map - baseline_map
        violation_reduction = (
            (baseline_violation - violation) / baseline_violation
            if baseline_violation > 0
            else 0.0
        )
        atomic_drop = baseline_atomic_map - _mean(atomic, "map")
        supporting = 0
        for partition in ("news", "region", "instance", "dance", "others"):
            candidate_partition = _stratum(test, partition=partition)
            baseline_partition = _stratum(baseline_test, partition=partition)
            partition_gain = _mean(candidate_partition, "map") - _mean(
                baseline_partition, "map"
            )
            baseline_partition_violation = _mean(
                baseline_partition, "constraint_violation_at_10"
            )
            partition_violation = _mean(
                candidate_partition, "constraint_violation_at_10"
            )
            relative_reduction = (
                (baseline_partition_violation - partition_violation)
                / baseline_partition_violation
                if baseline_partition_violation > 0
                else 0.0
            )
            supporting += int(
                partition_gain
                >= float(prototype_config["minimum_absolute_map_gain"])
                and relative_reduction
                >= float(
                    prototype_config["minimum_relative_violation_reduction"]
                )
            )
        strata_pass = True
        for operator, negated in (("AND", None), (None, True)):
            candidate = _stratum(test, operator=operator, negated=negated)
            baseline = _stratum(
                baseline_test, operator=operator, negated=negated
            )
            stratum_gain = _mean(candidate, "map") - _mean(baseline, "map")
            baseline_cv = _mean(baseline, "constraint_violation_at_10")
            candidate_cv = _mean(candidate, "constraint_violation_at_10")
            reduction = (
                (baseline_cv - candidate_cv) / baseline_cv
                if baseline_cv > 0
                else 0.0
            )
            strata_pass &= (
                stratum_gain
                >= float(prototype_config["minimum_absolute_map_gain"])
                and reduction
                >= float(
                    prototype_config["minimum_relative_violation_reduction"]
                )
            )
        accepted = (
            gain >= float(prototype_config["minimum_absolute_map_gain"])
            and violation_reduction
            >= float(prototype_config["minimum_relative_violation_reduction"])
            and supporting
            >= int(prototype_config["minimum_supporting_partitions"])
            and atomic_drop
            <= float(prototype_config["maximum_atomic_map_drop"])
            and strata_pass
        )
        seed_results.append(
            {
                "seed": int(seed),
                "selected_method": best_method,
                "map_gain": gain,
                "relative_violation_reduction": violation_reduction,
                "atomic_map_drop": atomic_drop,
                "supporting_partitions": supporting,
                "strata_pass": bool(strata_pass),
                "accepted": bool(accepted),
            }
        )
    return {
        "accepted": all(result["accepted"] for result in seed_results),
        "parameter_count": 2,
        "maximum_parameters": int(prototype_config["maximum_parameters"]),
        "seed_results": seed_results,
    }
