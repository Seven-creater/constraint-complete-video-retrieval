from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def rank_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_ids: np.ndarray,
    near_miss_ids: set[int],
    k: int = 10,
) -> dict[str, float]:
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    positive_count = int(labels.sum())
    if positive_count == 0:
        raise ValueError("ranking query has no positive candidates")
    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order].astype(np.float64)
    precision = np.cumsum(ranked_labels) / (np.arange(len(labels)) + 1)
    average_precision = float(np.sum(precision * ranked_labels) / positive_count)
    recall_200 = float(ranked_labels[:200].sum() / positive_count)
    top = ranked_labels[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float(np.sum(top * discounts))
    ideal_count = min(positive_count, k)
    idcg = float(np.sum(discounts[:ideal_count]))
    top_ids = candidate_ids[order[:k]]
    return {
        "map": average_precision,
        "recall_at_200": recall_200,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "constraint_violation_at_10": float(1.0 - top.mean()),
        "near_miss_at_10": float(
            sum(int(candidate_id) in near_miss_ids for candidate_id in top_ids)
            / len(top_ids)
        ),
    }


def aggregate_metrics(
    rows: Iterable[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    metric_names = (
        "map",
        "recall_at_200",
        "ndcg_at_10",
        "constraint_violation_at_10",
        "near_miss_at_10",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, values in sorted(grouped.items(), key=lambda item: item[0]):
        record = {key: value for key, value in zip(group_keys, group)}
        record["count"] = len(values)
        for metric in metric_names:
            record[metric] = float(np.mean([float(row[metric]) for row in values]))
        output.append(record)
    return output


def bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = generator.choice(array, size=len(array), replace=True).mean()
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def soft_logic(literal_scores: np.ndarray, operator: str, tau: float) -> np.ndarray:
    if literal_scores.ndim != 2:
        raise ValueError("literal scores must have shape [literal, candidate]")
    if operator == "ATOM":
        return literal_scores[0]
    if operator == "AND":
        scaled = -literal_scores / tau
        maximum = np.max(scaled, axis=0)
        return -tau * (
            maximum
            + np.log(np.mean(np.exp(scaled - maximum), axis=0))
        )
    if operator == "OR":
        scaled = literal_scores / tau
        maximum = np.max(scaled, axis=0)
        return tau * (
            maximum
            + np.log(np.mean(np.exp(scaled - maximum), axis=0))
        )
    raise ValueError(f"unsupported operator: {operator}")


def monotone_score(
    base_scores: np.ndarray,
    literal_scores: np.ndarray,
    operator: str,
    base_weight: float,
    tau: float,
) -> np.ndarray:
    if not 0.0 <= base_weight <= 1.0:
        raise ValueError("base weight must be in [0, 1]")
    base = np.clip((base_scores + 1.0) / 2.0, 1e-6, 1.0)
    logic = np.clip(soft_logic(literal_scores, operator, tau), 1e-6, 1.0)
    return np.power(base, base_weight) * np.power(logic, 1.0 - base_weight)

