from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from .io import read_json, write_json


def audit_literature(matrix: Path, output: Path) -> dict[str, Any]:
    with matrix.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    collisions = [
        row for row in rows if int(row.get("collision_axes") or 0) >= 3
    ]
    report = {
        "status": "literature_gate_passed" if not collisions else "literature_collision",
        "accepted": not collisions,
        "works_checked": len(rows),
        "three_axis_collisions": collisions,
    }
    write_json(output, report)
    return report


def finalize_direction(run: Path) -> dict[str, Any]:
    data = read_json(run / "data_audit.json")
    literature = read_json(run / "literature_audit.json")
    budget_path = run / "budget_ledger.json"
    openclip_path = run / "openclip" / "summary.json"
    eva_path = run / "eva_clip" / "summary.json"
    if budget_path.exists() and read_json(budget_path).get("exceeded", False):
        status = "gpu_budget_exceeded"
        reason = "the frozen four GPU-hour budget was exceeded"
    elif not literature["accepted"]:
        status = "literature_collision"
        reason = "nearest work covers at least three preregistered novelty axes"
    elif not data["accepted"]:
        status = "public_dataset_gate_failed"
        reason = "public MUVR release did not pass the frozen data gate"
    elif not openclip_path.exists():
        status = "direction_selection_incomplete"
        reason = "OpenCLIP problem gate has not completed"
    else:
        openclip = read_json(openclip_path)
        if openclip["simple_solution"]:
            status = "solved_by_simple_logic"
            reason = "a parameter-free method crossed the simple-solution threshold"
        elif not openclip["problem_gate"]["accepted"]:
            status = "problem_gate_failed"
            reason = "OpenCLIP did not establish the preregistered problem effect"
        elif not eva_path.exists():
            status = "direction_selection_incomplete"
            reason = "EVA-CLIP replication is required"
        else:
            eva = read_json(eva_path)
            if eva["simple_solution"]:
                status = "solved_by_simple_logic"
                reason = "a parameter-free method crossed the simple-solution threshold"
            elif not eva["problem_gate"]["accepted"]:
                status = "problem_gate_failed"
                reason = "the problem effect did not replicate on EVA-CLIP"
            elif not (
                openclip["prototype_gate"]["accepted"]
                and eva["prototype_gate"]["accepted"]
            ):
                status = "problem_confirmed_method_foothold_missing"
                reason = "the problem replicated but the minimum method gate failed"
            else:
                status = "direction_confirmed"
                reason = "data, problem, replication, and prototype gates all passed"
    confirmed = status == "direction_confirmed"
    decision = {
        "status": status,
        "reason": reason,
        "direction": (
            "constraint_complete_multimodal_untrimmed_video_retrieval"
            if confirmed
            else None
        ),
        "paper_core": "monotone_literal_coverage" if confirmed else None,
        "thresholds_changed": False,
        "human_annotations": 0,
        "omni_calls": 0,
        "temporal_cooccurrence_claim": False,
    }
    write_json(run / "direction_decision.json", decision)
    _write_markdown(run / "direction_decision.md", decision)
    return decision


def _write_markdown(path: Path, decision: dict[str, Any]) -> None:
    content = (
        "# Research direction decision\n\n"
        f"Status: **{decision['status']}**\n\n"
        f"Reason: {decision['reason']}\n\n"
        f"- Direction: `{decision['direction']}`\n"
        f"- Paper core: `{decision['paper_core']}`\n"
        "- Thresholds changed: no\n"
        "- Human annotations: 0\n"
        "- Omni calls: 0\n"
        "- Temporal co-occurrence claim: no\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
