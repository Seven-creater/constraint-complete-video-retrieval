from pathlib import Path

from ccvr.decision import audit_literature, finalize_direction
from ccvr.io import write_json


def test_literature_gate_rejects_three_axis_collision(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.csv"
    matrix.write_text(
        "work,collision_axes\nsafe,2\ncollision,3\n", encoding="utf-8"
    )
    report = audit_literature(matrix, tmp_path / "audit.json")
    assert not report["accepted"]
    assert report["three_axis_collisions"][0]["work"] == "collision"


def test_final_decision_requires_both_backbones_and_prototypes(
    tmp_path: Path,
) -> None:
    write_json(tmp_path / "data_audit.json", {"accepted": True})
    write_json(tmp_path / "literature_audit.json", {"accepted": True})
    for directory in ("openclip", "eva_clip"):
        write_json(
            tmp_path / directory / "summary.json",
            {
                "simple_solution": False,
                "problem_gate": {"accepted": True},
                "prototype_gate": {"accepted": True},
            },
        )
    decision = finalize_direction(tmp_path)
    assert decision["status"] == "direction_confirmed"
    assert decision["temporal_cooccurrence_claim"] is False

