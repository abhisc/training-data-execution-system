"""Evaluation and validation firewall."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from .sharding import ShardManifest


@dataclass
class EvalRegistryEntry:
    shard_id: str
    content_hash: str
    benchmark_id: Optional[str]
    role: str
    never_train: bool = True


class EvalFirewall:
    """Registry of never-train shards with executable admission checks."""

    def __init__(self) -> None:
        self.entries: Dict[str, EvalRegistryEntry] = {}
        self.blocked_events: List[Dict] = []

    def register_from_manifests(self, manifests: Sequence[ShardManifest]) -> int:
        count = 0
        for m in manifests:
            if m.never_train or m.role in ("eval", "val"):
                self.entries[m.shard_id] = EvalRegistryEntry(
                    shard_id=m.shard_id,
                    content_hash=m.content_hash,
                    benchmark_id=m.benchmark_id,
                    role=m.role,
                    never_train=True,
                )
                count += 1
        return count

    def is_blocked(self, shard_id: str) -> bool:
        return shard_id in self.entries

    def blocked_ids(self) -> Set[str]:
        return set(self.entries.keys())

    def check_admission(self, shard_id: str, context: str = "dataloader") -> bool:
        """Return True if allowed for training; False if blocked.

        On block, records a rejection event.
        """
        if shard_id in self.entries:
            entry = self.entries[shard_id]
            event = {
                "event": "eval_shard_blocked",
                "shard_id": shard_id,
                "content_hash": entry.content_hash,
                "benchmark_id": entry.benchmark_id,
                "role": entry.role,
                "context": context,
                "reason": "never_train flag set in evaluation/validation registry",
            }
            self.blocked_events.append(event)
            return False
        return True

    def assert_no_eval_in_training(
        self, shard_ids: Sequence[str], context: str = "batch"
    ) -> None:
        for sid in shard_ids:
            if not self.check_admission(sid, context=context):
                raise PermissionError(
                    f"Evaluation/validation shard blocked from training: {sid}"
                )

    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "entries": {k: asdict(v) for k, v in self.entries.items()},
            "blocked_events": self.blocked_events,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EvalFirewall":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fw = cls()
        for k, v in data.get("entries", {}).items():
            fw.entries[k] = EvalRegistryEntry(**v)
        fw.blocked_events = list(data.get("blocked_events", []))
        return fw
