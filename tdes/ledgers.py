"""Append-only consumption, learning, and OPUS ledgers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class JsonlLedger:
    """Append-only JSONL ledger with offset tracking."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        self._offset = self._count_lines()

    def _count_lines(self) -> int:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0
        with self.path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    @property
    def offset(self) -> int:
        return self._offset

    def append(self, record: Dict[str, Any]) -> int:
        """Append a record and return its 0-based offset."""
        offset = self._offset
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self._offset += 1
        return offset

    def truncate_to(self, offset: int) -> None:
        """Keep only the first `offset` records (for resume/fork hygiene)."""
        records = self.read_all()[:offset]
        with self.path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        self._offset = len(records)

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        out: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def read_range(self, start: int, end: int) -> List[Dict[str, Any]]:
        return self.read_all()[start:end]

    def get(self, offset: int) -> Optional[Dict[str, Any]]:
        rows = self.read_all()
        if 0 <= offset < len(rows):
            return rows[offset]
        return None


class LedgerSuite:
    """Bundles consumption, learning, and OPUS audit ledgers."""

    def __init__(self, ledger_dir: Path):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.consumption = JsonlLedger(self.ledger_dir / "consumption.jsonl")
        self.learning = JsonlLedger(self.ledger_dir / "learning.jsonl")
        self.opus = JsonlLedger(self.ledger_dir / "opus_audit.jsonl")

    def record_opus(self, decision: Dict[str, Any]) -> int:
        return self.opus.append(decision)

    def record_consumption(self, record: Dict[str, Any]) -> int:
        return self.consumption.append(record)

    def record_learning(self, record: Dict[str, Any]) -> int:
        return self.learning.append(record)

    def consumption_offset(self) -> int:
        return self.consumption.offset
