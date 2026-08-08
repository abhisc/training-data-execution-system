"""Crash, resume, replay, and fork orchestration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ledgers import LedgerSuite
from .mixture import MixtureSchedule
from .model import MockModel
from .opus import OpusGate
from .packing import PackedBatch
from .trainer import LoaderCursor, SimulatedCrash, Trainer
from .firewall import EvalFirewall
from .sharding import ShardManifest


def clone_trainer_for_resume(
    source: Trainer,
    ckpt_id: str,
    ledgers: LedgerSuite,
    opus: Optional[OpusGate] = None,
    branch_id: Optional[str] = None,
) -> Tuple[Trainer, Dict[str, Any]]:
    """Create a fresh trainer and restore from checkpoint."""
    model = MockModel(dim=source.model.dim, seed=0)
    new_opus = opus or OpusGate(
        accept_threshold=source.opus.accept_threshold,
        defer_threshold=source.opus.defer_threshold,
        force_reject_every=source.opus.force_reject_every,
        force_defer_every=source.opus.force_defer_every,
    )
    t = Trainer(
        manifests=source.all_manifests,
        schedule=source.schedule,
        firewall=source.firewall,
        model=model,
        ledgers=ledgers,
        opus=new_opus,
        seq_len=source.seq_len,
        docs_per_batch=source.docs_per_batch,
        run_id=source.run_id,
        branch_id=branch_id or source.branch_id,
        checkpoint_dir=source.checkpoint_dir,
    )
    payload = t.load_checkpoint(ckpt_id)
    # Truncate ledgers to checkpoint offsets for a clean resume branch view
    ledgers.consumption.truncate_to(int(payload["ledger_offset"]))
    ledgers.learning.truncate_to(int(payload["learning_offset"]))
    ledgers.opus.truncate_to(int(payload["opus_offset"]))
    if branch_id:
        t.branch_id = branch_id
    return t, payload


def expected_stream_record(stream: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    for rec in stream:
        if rec["global_step"] == step:
            return rec
    return None


def prove_resume(
    original_stream: List[Dict[str, Any]],
    resumed: Trainer,
) -> Dict[str, Any]:
    """Build the next batch after resume and compare to original stream."""
    step = resumed.global_step
    # Peek-build without permanently needing side effects beyond cursor:
    # build_batch_at with use_live_cursor=True advances cursor — that is correct
    # for the resumed next batch.
    cursor_before = resumed.cursor.to_dict()
    batch, plan = resumed.build_batch_at(step, use_live_cursor=True)
    # Restore cursor so caller can train normally from this step
    resumed.cursor = LoaderCursor.from_dict(cursor_before)

    expected = expected_stream_record(original_stream, step)
    if expected is None:
        return {
            "ok": False,
            "reason": f"no original stream record for step {step}",
            "step": step,
        }

    matched = (
        batch.batch_id == expected["batch_id"]
        and batch.batch_hash == expected["batch_hash"]
        and batch.shard_ids == expected["shard_ids"]
        and batch.token_spans == expected["token_spans"]
    )
    return {
        "ok": matched,
        "step": step,
        "expected_batch_id": expected["batch_id"],
        "resumed_batch_id": batch.batch_id,
        "expected_batch_hash": expected["batch_hash"],
        "resumed_batch_hash": batch.batch_hash,
        "expected_shard_ids": expected["shard_ids"],
        "resumed_shard_ids": batch.shard_ids,
        "checkpoint_id": resumed.last_checkpoint_id,
    }


def replay_interval(
    trainer_template: Trainer,
    start_step: int,
    end_step: int,
    original_stream: List[Dict[str, Any]],
    start_cursor: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Reconstruct batches for [start_step, end_step) and compare hashes."""
    # Fresh cursor replay from step 0 to start_step, or use provided cursor
    model = MockModel(dim=trainer_template.model.dim, seed=0)
    # Use a throwaway ledger dir under checkpoint parent
    tmp_ledgers = LedgerSuite(
        trainer_template.ledgers.ledger_dir / "_replay_tmp"
    )
    # Clear throwaway
    for p in tmp_ledgers.ledger_dir.glob("*.jsonl"):
        p.write_text("", encoding="utf-8")
    tmp_ledgers = LedgerSuite(trainer_template.ledgers.ledger_dir / "_replay_tmp")

    t = Trainer(
        manifests=trainer_template.all_manifests,
        schedule=trainer_template.schedule,
        firewall=trainer_template.firewall,
        model=model,
        ledgers=tmp_ledgers,
        opus=OpusGate(
            accept_threshold=trainer_template.opus.accept_threshold,
            defer_threshold=trainer_template.opus.defer_threshold,
            force_reject_every=trainer_template.opus.force_reject_every,
            force_defer_every=trainer_template.opus.force_defer_every,
        ),
        seq_len=trainer_template.seq_len,
        docs_per_batch=trainer_template.docs_per_batch,
        run_id="replay",
        branch_id="replay",
        checkpoint_dir=None,
    )
    if start_cursor is not None:
        t.cursor = LoaderCursor.from_dict(start_cursor)
        t.global_step = start_step
    else:
        # Fast-forward cursor by building (and discarding) batches 0..start_step-1
        for s in range(start_step):
            t.build_batch_at(s, use_live_cursor=True)
        t.global_step = start_step

    comparisons: List[Dict[str, Any]] = []
    all_ok = True
    for s in range(start_step, end_step):
        batch, _plan = t.build_batch_at(s, use_live_cursor=True)
        expected = expected_stream_record(original_stream, s)
        if expected is None:
            comparisons.append({"step": s, "ok": False, "reason": "missing original"})
            all_ok = False
            continue
        ok = (
            batch.batch_id.split(":")[-1] == expected["batch_id"].split(":")[-1]
            and batch.batch_hash == expected["batch_hash"]
            and batch.shard_ids == expected["shard_ids"]
            and batch.token_spans == expected["token_spans"]
        )
        # Note: batch_id includes branch prefix; compare step suffix + hash
        comparisons.append(
            {
                "step": s,
                "ok": ok,
                "original_batch_id": expected["batch_id"],
                "replay_batch_id": batch.batch_id,
                "original_hash": expected["batch_hash"],
                "replay_hash": batch.batch_hash,
                "original_spans": expected["token_spans"],
                "replay_spans": batch.token_spans,
            }
        )
        if not ok:
            all_ok = False

    return {
        "ok": all_ok,
        "start_step": start_step,
        "end_step": end_step,
        "comparisons": comparisons,
    }


def fork_from_checkpoint(
    source: Trainer,
    ckpt_id: str,
    new_branch_id: str,
    ledger_dir: Path,
    n_steps: int = 5,
) -> Dict[str, Any]:
    """Restore checkpoint and train a divergent branch."""
    ledgers = LedgerSuite(ledger_dir)
    # Copy opus settings; do not truncate main ledgers — use branch-specific suite
    forked, payload = clone_trainer_for_resume(
        source,
        ckpt_id,
        ledgers=ledgers,
        branch_id=new_branch_id,
    )
    forked.run_id = f"{source.run_id}_fork"
    forked.branch_id = new_branch_id
    start = forked.global_step
    forked.train_steps(n_steps=n_steps, checkpoint_every=0, crash_at=None)
    return {
        "ok": True,
        "parent_checkpoint": ckpt_id,
        "parent_step": int(payload["global_step"]),
        "branch_id": new_branch_id,
        "fork_start_step": start,
        "fork_end_step": forked.global_step,
        "fork_stream": forked.batch_stream,
        "consumption_records": ledgers.consumption.read_all(),
    }


def run_crash_and_resume(
    trainer: Trainer,
    total_steps: int,
    crash_at: int,
    checkpoint_every: int = 10,
) -> Dict[str, Any]:
    """Train with a simulated crash, then resume and prove next-batch match."""
    assert crash_at > checkpoint_every, "crash_at must be after at least one checkpoint"

    try:
        trainer.train_steps(
            n_steps=total_steps,
            checkpoint_every=checkpoint_every,
            crash_at=crash_at,
        )
        crashed = False
    except SimulatedCrash as e:
        crashed = True
        crash_msg = str(e)

    original_stream = list(trainer.batch_stream)
    if not trainer.checkpoints:
        raise RuntimeError("No checkpoints saved before crash")

    # Use the last checkpoint at or before crash_at
    ckpt_id = trainer.checkpoints[-1]
    ckpt_meta = json.loads(
        (trainer.checkpoint_dir / f"{ckpt_id}.json").read_text(encoding="utf-8")
    )
    resume_step = int(ckpt_meta["global_step"])

    # Fresh ledger suite for resumed run (prove reconstruction)
    resume_ledger_dir = trainer.ledgers.ledger_dir / "resume_run"
    resume_ledgers = LedgerSuite(resume_ledger_dir)
    resumed, payload = clone_trainer_for_resume(trainer, ckpt_id, resume_ledgers)

    proof = prove_resume(original_stream, resumed)

    # Continue a few steps after resume to show continuity
    resumed.train_steps(n_steps=3, checkpoint_every=0)

    return {
        "crashed": crashed,
        "crash_at": crash_at,
        "crash_message": crash_msg if crashed else None,
        "checkpoint_id": ckpt_id,
        "resume_step": resume_step,
        "proof": proof,
        "original_stream_len": len(original_stream),
        "resumed_stream": resumed.batch_stream,
    }
