"""Execution log writer with [PASS]/[FAIL] events."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class RunLogger:
    """Append-only run.log with structured event helpers."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.path.write_text("", encoding="utf-8")

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def log(self, message: str) -> None:
        line = f"[{self._ts()}] {message}"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def event(self, name: str) -> None:
        self.log(f"EVENT {name}")

    def pass_(self, name: str, detail: str = "") -> None:
        suffix = f" {detail}" if detail else ""
        self.log(f"[PASS] {name}{suffix}")

    def fail(self, name: str, detail: str = "") -> None:
        suffix = f" {detail}" if detail else ""
        self.log(f"[FAIL] {name}{suffix}")

    def info(self, message: str) -> None:
        self.log(message)
