from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .io import write_json


class ExperimentBudget:
    """Shared GPU-hour ledger plus a fail-closed single-scorer lock."""

    def __init__(self, run_dir: Path, maximum_gpu_hours: float, stage: str) -> None:
        self.run_dir = run_dir
        self.maximum_gpu_hours = float(maximum_gpu_hours)
        self.stage = stage
        self.ledger_path = run_dir / "budget_ledger.json"
        self.lock_path = run_dir / ".scoring.lock"
        self._lock_fd: int | None = None
        self._started = 0.0

    def __enter__(self) -> "ExperimentBudget":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._lock_fd = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                f"Refusing duplicate scoring process; lock exists: {self.lock_path}"
            ) from exc

        os.write(self._lock_fd, f"{os.getpid()}\n".encode("utf-8"))
        ledger = self._read_ledger()
        consumed = float(ledger.get("consumed_gpu_hours", 0.0))
        if consumed >= self.maximum_gpu_hours:
            self._release_lock()
            raise RuntimeError(
                f"GPU budget exhausted: {consumed:.6f} >= {self.maximum_gpu_hours:.6f}"
            )
        self._started = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> None:
        elapsed_hours = max(0.0, time.monotonic() - self._started) / 3600.0
        ledger = self._read_ledger()
        consumed = float(ledger.get("consumed_gpu_hours", 0.0)) + elapsed_hours
        entries = list(ledger.get("entries", []))
        entries.append(
            {
                "stage": self.stage,
                "pid": os.getpid(),
                "gpu_hours": elapsed_hours,
                "status": "ok" if exc is None else "error",
                "error": None if exc is None else str(exc),
            }
        )
        write_json(
            self.ledger_path,
            {
                "maximum_gpu_hours": self.maximum_gpu_hours,
                "consumed_gpu_hours": consumed,
                "remaining_gpu_hours": max(0.0, self.maximum_gpu_hours - consumed),
                "exceeded": consumed > self.maximum_gpu_hours,
                "entries": entries,
            },
        )
        self._release_lock()

    def _read_ledger(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            return {
                "maximum_gpu_hours": self.maximum_gpu_hours,
                "consumed_gpu_hours": 0.0,
                "remaining_gpu_hours": self.maximum_gpu_hours,
                "exceeded": False,
                "entries": [],
            }
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        self.lock_path.unlink(missing_ok=True)
