from pathlib import Path

import pytest

from ccvr.budget import ExperimentBudget


def test_budget_records_usage_and_blocks_duplicate(tmp_path: Path) -> None:
    with ExperimentBudget(tmp_path, maximum_gpu_hours=4.0, stage="first"):
        with pytest.raises(RuntimeError, match="duplicate"):
            with ExperimentBudget(tmp_path, maximum_gpu_hours=4.0, stage="duplicate"):
                pass

    assert not (tmp_path / ".scoring.lock").exists()
    ledger = (tmp_path / "budget_ledger.json").read_text(encoding="utf-8")
    assert '"stage": "first"' in ledger
