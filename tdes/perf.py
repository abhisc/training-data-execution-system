"""Throughput and packing efficiency reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .ledgers import LedgerSuite
from .opus import OpusGate


def compute_performance(
    perf_events: Sequence[Dict[str, Any]],
    opus: OpusGate,
    ledgers: LedgerSuite,
    seq_len: int,
) -> Dict[str, Any]:
    events = list(perf_events)
    loader_s = sum(e.get("loader_s", 0.0) for e in events)
    train_s = sum(e.get("train_s", 0.0) for e in events)
    total_s = loader_s + train_s
    raw_tokens = sum(e.get("raw_tokens", 0) for e in events)
    useful_tokens = sum(e.get("useful_tokens", 0) for e in events)
    accepted = [e for e in events if e.get("accepted")]
    rejected = [e for e in events if not e.get("accepted")]

    utils = [e.get("utilization", 0.0) for e in events if e.get("accepted")]
    avg_util = sum(utils) / len(utils) if utils else 0.0

    decisions = [d.decision for d in opus.decisions]
    n_dec = len(decisions) or 1
    reject_rate = decisions.count("reject") / n_dec
    defer_rate = decisions.count("defer") / n_dec
    accept_rate = decisions.count("accept") / n_dec
    floor_rate = decisions.count("floor_override") / n_dec

    # GPU idle proxy: loader wait relative to train
    gpu_idle_s = loader_s  # time GPU waits on loader in this mock
    loader_wait_s = loader_s

    # Cache hit rate: deterministic rebuilds always hit (demo reports 1.0 when stream exists)
    stream_path = ledgers.ledger_dir / "batch_stream.jsonl"
    cache_hit_rate = 1.0 if stream_path.exists() else 0.0

    raw_tps = raw_tokens / total_s if total_s > 0 else 0.0
    useful_tps = useful_tokens / total_s if total_s > 0 else 0.0
    accepted_tokens = sum(e.get("useful_tokens", 0) for e in accepted)
    accepted_tps = accepted_tokens / total_s if total_s > 0 else 0.0

    report = {
        "steps_observed": len(events),
        "accepted_steps": len(accepted),
        "rejected_or_deferred_steps": len(rejected),
        "seq_len": seq_len,
        "raw_tokens": raw_tokens,
        "useful_loss_bearing_tokens": useful_tokens,
        "total_time_s": total_s,
        "loader_time_s": loader_s,
        "train_time_s": train_s,
        "raw_tokens_per_sec": raw_tps,
        "useful_loss_bearing_tokens_per_sec": useful_tps,
        "accepted_tokens_per_sec": accepted_tps,
        "opus_rejection_rate": reject_rate,
        "opus_defer_rate": defer_rate,
        "opus_accept_rate": accept_rate,
        "opus_floor_override_rate": floor_rate,
        "gpu_idle_time_s": gpu_idle_s,
        "loader_wait_time_s": loader_wait_s,
        "cache_hit_rate": cache_hit_rate,
        "packing_utilization_avg": avg_util,
        "decision_counts": {
            "accept": decisions.count("accept"),
            "reject": decisions.count("reject"),
            "defer": decisions.count("defer"),
            "floor_override": decisions.count("floor_override"),
        },
    }
    return report


def save_performance(report: Dict[str, Any], path: Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
