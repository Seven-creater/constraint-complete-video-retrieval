from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, write_json, write_jsonl
from .logic import (
    evaluate_expression,
    operator_aware_near_misses,
    parse_expression,
)
from .remote import huggingface_endpoint


DATASET_ID = "debby0527/MUVR"
PARTITIONS = ("news", "region", "instance", "dance", "others")
METADATA_FILES = (
    "queries_en.json",
    "query_rel_lists.json",
    "query_tag_lists.json",
    "videos.json",
)


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "ccvr/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ccvr/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    os.replace(temporary, destination)


def fetch_metadata(output: Path, revision: str = "main") -> dict[str, Any]:
    endpoint = huggingface_endpoint()
    resolved_revision = revision
    if revision == "main":
        api = _fetch_json(f"{endpoint}/api/datasets/{DATASET_ID}")
        resolved_revision = str(api.get("sha") or revision)
    files: dict[str, dict[str, Any]] = {}
    for partition in PARTITIONS:
        for filename in METADATA_FILES:
            relative = Path(partition) / filename
            destination = output / relative
            url = (
                f"{endpoint}/datasets/{DATASET_ID}/resolve/"
                f"{resolved_revision}/annotations/retrieval/{partition}/{filename}"
            )
            if not destination.exists():
                _download(url, destination)
            files[relative.as_posix()] = {
                "sha256": sha256_file(destination),
                "size": destination.stat().st_size,
            }
    lock = {
        "dataset": DATASET_ID,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "endpoint": endpoint,
        "files": files,
    }
    write_json(output / "source_lock.json", lock)
    return lock


def _topic_splits(
    topics_by_partition: dict[str, set[int]],
    fractions: tuple[float, float, float],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for partition, topic_ids in topics_by_partition.items():
        ranked = sorted(
            topic_ids,
            key=lambda topic_id: hashlib.sha256(
                f"{partition}:{topic_id}".encode("utf-8")
            ).hexdigest(),
        )
        count = len(ranked)
        train_end = round(count * fractions[0])
        dev_end = train_end + round(count * fractions[1])
        for index, topic_id in enumerate(ranked):
            split = "train" if index < train_end else "dev" if index < dev_end else "test"
            mapping[f"{partition}:{topic_id}"] = split
    return mapping


def _query_key(partition: str, query_id: int, expression: str) -> str:
    suffix = hashlib.sha256(expression.encode("utf-8")).hexdigest()[:12]
    return f"{partition}:{query_id}:{suffix}"


def audit_dataset(
    raw: Path,
    output: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source_lock_path = raw / "source_lock.json"
    if not source_lock_path.exists():
        raise FileNotFoundError(f"missing source lock: {source_lock_path}")
    source_lock = read_json(source_lock_path)
    expected_revision = str(config["dataset"]["revision"])
    if (
        source_lock.get("requested_revision") != expected_revision
        and source_lock.get("resolved_revision") != expected_revision
    ):
        raise ValueError("metadata revision differs from preregistered revision")

    loaded: dict[str, dict[str, Any]] = {}
    topics_by_partition: dict[str, set[int]] = defaultdict(set)
    for partition in PARTITIONS:
        queries = read_json(raw / partition / "queries_en.json")
        relationships = read_json(raw / partition / "query_rel_lists.json")
        tag_lists = read_json(raw / partition / "query_tag_lists.json")
        videos = read_json(raw / partition / "videos.json")
        query_by_id = {int(item["id"]): item for item in queries}
        relation_by_query = {int(item["query_id"]): item for item in relationships}
        video_by_id = {int(item["id"]): item for item in videos}
        videos_by_topic: dict[int, list[int]] = defaultdict(list)
        for video in videos:
            topic_id = int(video["topic_id"])
            topics_by_partition[partition].add(topic_id)
            videos_by_topic[topic_id].append(int(video["id"]))
        loaded[partition] = {
            "query_by_id": query_by_id,
            "query_ids": set(query_by_id),
            "relation_by_query": relation_by_query,
            "tag_lists": tag_lists,
            "video_by_id": video_by_id,
            "videos_by_topic": videos_by_topic,
        }

    fractions = tuple(float(value) for value in config["dataset"]["split_fractions"])
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must contain three values summing to one")
    split_by_topic = _topic_splits(topics_by_partition, fractions)  # type: ignore[arg-type]

    dataset_config = config["dataset"]
    min_positives = int(dataset_config["minimum_positives"])
    min_near_misses = int(dataset_config["minimum_near_miss_negatives"])
    min_prevalence = float(dataset_config["minimum_prevalence"])
    max_prevalence = float(dataset_config["maximum_prevalence"])

    rows: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    unresolvable: list[dict[str, Any]] = []
    compared_composites = 0

    for partition in PARTITIONS:
        values = loaded[partition]
        for tag_item in values["tag_lists"]:
            query_id = int(tag_item["query_id"])
            if query_id not in values["query_by_id"]:
                raise ValueError(f"{partition}: tag query {query_id} lacks query metadata")
            query = values["query_by_id"][query_id]
            topic_id = int(query["topic_id"])
            topic_key = f"{partition}:{topic_id}"
            relation_item = values["relation_by_query"].get(query_id)
            if relation_item is None:
                raise ValueError(f"{partition}: query {query_id} lacks relationships")
            base_relevant = set(
                int(value)
                for value in relation_item["rel_lists"]["pos_all"]["id_list"]
            )
            base_relevant -= values["query_ids"]
            expressions = {
                source: parse_expression(source)
                for source in tag_item["tag_lists"].keys()
            }
            atomic_sets = {
                expression.literals[0].key: set(
                    int(value)
                    for value in tag_item["tag_lists"][source]["id_list"]
                )
                - values["query_ids"]
                for source, expression in expressions.items()
                if expression.operator == "ATOM"
            }
            all_tags = {
                literal.tag
                for expression in expressions.values()
                for literal in expression.literals
            }
            for tag in all_tags:
                positive_key = ("+", tag)
                negative_key = ("-", tag)
                if positive_key not in atomic_sets and negative_key in atomic_sets:
                    atomic_sets[positive_key] = base_relevant - atomic_sets[negative_key]
                if negative_key not in atomic_sets and positive_key in atomic_sets:
                    atomic_sets[negative_key] = base_relevant - atomic_sets[positive_key]
            for source, expression in expressions.items():
                published = set(
                    int(value)
                    for value in tag_item["tag_lists"][source]["id_list"]
                ) - values["query_ids"]
                raw_counts["queries"] += 1
                raw_counts[expression.operator] += 1
                raw_counts["negated"] += int(expression.contains_negation)
                expression_resolvable = True
                if expression.operator != "ATOM":
                    compared_composites += 1
                    try:
                        recomputed = evaluate_expression(expression, atomic_sets)
                    except KeyError as exc:
                        expression_resolvable = False
                        recomputed = set()
                        unresolvable.append(
                            {
                                "partition": partition,
                                "query_id": query_id,
                                "expression": source,
                                "error": str(exc),
                            }
                        )
                    if expression_resolvable and recomputed != published:
                        mismatches.append(
                            {
                                "partition": partition,
                                "query_id": query_id,
                                "expression": source,
                                "missing": sorted(published - recomputed),
                                "unexpected": sorted(recomputed - published),
                            }
                        )
                if expression_resolvable:
                    near_misses, negative_definition = operator_aware_near_misses(
                        expression,
                        published,
                        base_relevant,
                        atomic_sets,
                    )
                else:
                    near_misses = base_relevant - published
                    negative_definition = "unresolvable_missing_atomic_literal"
                prevalence = (
                    len(published) / len(base_relevant) if base_relevant else 0.0
                )
                eligible = (
                    expression.operator in {"AND", "OR"}
                    and expression_resolvable
                    and len(published) >= min_positives
                    and len(near_misses) >= min_near_misses
                    and min_prevalence <= prevalence <= max_prevalence
                )
                if eligible:
                    eligible_counts["queries"] += 1
                    eligible_counts[expression.operator] += 1
                    eligible_counts["negated"] += int(expression.contains_negation)
                    eligible_counts[f"partition:{partition}"] += 1
                video_by_id = values["video_by_id"]
                query_video = video_by_id.get(query_id, query)
                rows.append(
                    {
                        "query_id": _query_key(partition, query_id, source),
                        "source_query_id": query_id,
                        "query_video_name": str(
                            Path(str(query_video["frames_path"])).name
                        ),
                        "topic_id": topic_key,
                        "partition": partition,
                        "split": split_by_topic[topic_key],
                        "operator": expression.operator,
                        "expression": source,
                        "literals": [
                            literal.as_dict() for literal in expression.literals
                        ],
                        "positives": sorted(published),
                        "partial_negatives": sorted(near_misses),
                        "negative_definition": negative_definition,
                        "base_positives": sorted(base_relevant),
                        "prevalence_within_base_positives": prevalence,
                        "eligible_for_diagnostic": eligible,
                        "source_revision": source_lock["resolved_revision"],
                    }
                )

    rows.sort(
        key=lambda row: (
            row["partition"],
            row["topic_id"],
            row["source_query_id"],
            row["expression"],
        )
    )
    manifest_path = output / "manifest.jsonl"
    row_count = write_jsonl(manifest_path, rows)
    algebra_failures = len(mismatches) + len(unresolvable)
    agreement = (
        (compared_composites - algebra_failures) / compared_composites
        if compared_composites
        else 0.0
    )
    gate_config = config["data_gate"]
    gate_checks = {
        "minimum_eligible_queries": eligible_counts["queries"]
        >= int(gate_config["minimum_eligible_queries"]),
        "minimum_and_queries": eligible_counts["AND"]
        >= int(gate_config["minimum_and_queries"]),
        "minimum_or_queries": eligible_counts["OR"]
        >= int(gate_config["minimum_or_queries"]),
        "minimum_negated_queries": eligible_counts["negated"]
        >= int(gate_config["minimum_negated_queries"]),
        "required_partitions": sum(
            eligible_counts[f"partition:{partition}"] > 0 for partition in PARTITIONS
        )
        >= int(gate_config["required_partitions"]),
        "required_algebra_agreement": agreement
        == float(gate_config["required_algebra_agreement"]),
    }
    accepted = all(gate_checks.values())
    report = {
        "protocol_version": config["protocol_version"],
        "status": "data_gate_passed" if accepted else "public_dataset_gate_failed",
        "accepted": accepted,
        "source_revision": source_lock["resolved_revision"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": row_count,
        "raw_counts": dict(raw_counts),
        "eligible_counts": dict(eligible_counts),
        "paper_reported_query_count": 93885,
        "observed_query_count": raw_counts["queries"],
        "paper_observed_count_difference": 93885 - raw_counts["queries"],
        "compared_composites": compared_composites,
        "algebra_mismatches": len(mismatches),
        "unresolvable_composites": len(unresolvable),
        "algebra_agreement": agreement,
        "mismatch_examples": (mismatches + unresolvable)[:20],
        "gate_checks": gate_checks,
        "split_topic_counts": dict(
            Counter(split_by_topic.values())
        ),
    }
    write_json(output / "source_lock.json", source_lock)
    write_json(output / "data_audit.json", report)
    write_json(
        output / "stage_decision.json",
        {
            "status": report["status"],
            "next_stage": "openclip_problem_gate" if accepted else None,
            "thresholds_changed": False,
        },
    )
    _write_audit_markdown(output / "data_audit.md", report)
    return report


def _write_audit_markdown(path: Path, report: dict[str, Any]) -> None:
    checks = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in report["gate_checks"].items()
    )
    content = (
        "# MUVR public data gate\n\n"
        f"Status: **{report['status']}**\n\n"
        f"- Frozen revision: `{report['source_revision']}`\n"
        f"- Observed queries: {report['observed_query_count']}\n"
        f"- Paper-reported upper bound: {report['paper_reported_query_count']}\n"
        f"- Eligible diagnostic queries: "
        f"{report['eligible_counts'].get('queries', 0)}\n"
        f"- Algebra agreement: {report['algebra_agreement']:.6f}\n"
        f"- Manifest SHA-256: `{report['manifest_sha256']}`\n\n"
        "## Preregistered checks\n\n"
        f"{checks}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
