"""Training loop, dataloader cursor, and checkpointing."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .firewall import EvalFirewall
from .ledgers import LedgerSuite
from .mixture import MixtureSchedule, TRAIN_LANES
from .model import MockModel, StepLossResult
from .opus import OpusGate, OpusDecision
from .packing import DocumentPiece, PackedBatch, pack_documents
from .sharding import ShardManifest, load_shard_tokens


@dataclass
class LoaderCursor:
    """Per-lane document cursor for deterministic sampling."""

    lane_indices: Dict[str, int] = field(default_factory=dict)

    def next_index(self, lane: str, n: int) -> int:
        idx = self.lane_indices.get(lane, 0)
        self.lane_indices[lane] = idx + 1
        return idx % max(1, n)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.lane_indices)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "LoaderCursor":
        return cls(lane_indices=dict(data))


class SimulatedCrash(Exception):
    """Raised deliberately to demonstrate crash recovery."""


class Trainer:
    """Deterministic training orchestrator."""

    def __init__(
        self,
        manifests: Sequence[ShardManifest],
        schedule: MixtureSchedule,
        firewall: EvalFirewall,
        model: MockModel,
        ledgers: LedgerSuite,
        opus: OpusGate,
        seq_len: int = 64,
        docs_per_batch: int = 3,
        run_id: str = "run_main",
        branch_id: str = "main",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.all_manifests = list(manifests)
        self.train_manifests = [m for m in manifests if not m.never_train]
        self.by_lane: Dict[str, List[ShardManifest]] = {l: [] for l in TRAIN_LANES}
        for m in self.train_manifests:
            lane = m.capability_lane
            if lane in self.by_lane:
                self.by_lane[lane].append(m)

        self.schedule = schedule
        self.firewall = firewall
        self.model = model
        self.ledgers = ledgers
        self.opus = opus
        self.seq_len = seq_len
        self.docs_per_batch = docs_per_batch
        self.run_id = run_id
        self.branch_id = branch_id
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.cursor = LoaderCursor()
        self.last_checkpoint_id: Optional[str] = None
        self.checkpoints: List[str] = []
        self.perf_events: List[Dict[str, Any]] = []
        self.lane_counts: Dict[str, int] = {l: 0 for l in TRAIN_LANES}
        self.accepted_steps = 0
        self._crash_at: Optional[int] = None
        # Every schedule step (accepted or not) for resume/replay proofs
        self.batch_stream: List[Dict[str, Any]] = []
        self.stream_path = (
            self.ledgers.ledger_dir / "batch_stream.jsonl" if ledgers else None
        )

    def recent_lane_shares(self) -> Dict[str, float]:
        total = sum(self.lane_counts.values()) or 1
        return {l: self.lane_counts.get(l, 0) / total for l in TRAIN_LANES}

    def _select_docs(self, lane: str) -> List[DocumentPiece]:
        pool = self.by_lane.get(lane) or self.train_manifests
        if not pool:
            raise RuntimeError(f"No training shards for lane {lane}")
        docs: List[DocumentPiece] = []
        for _ in range(self.docs_per_batch):
            idx = self.cursor.next_index(lane, len(pool))
            m = pool[idx]
            # Firewall check before packing
            if not self.firewall.check_admission(m.shard_id, context="select_docs"):
                raise PermissionError(f"Blocked shard reached selection: {m.shard_id}")
            tokens = load_shard_tokens(m).tolist()
            prompt_end = None
            if m.spans and m.spans[0].get("prompt_end") is not None:
                prompt_end = m.spans[0]["prompt_end"]
            docs.append(
                DocumentPiece(
                    doc_id=m.document_ids[0],
                    shard_id=m.shard_id,
                    tokens=tokens,
                    prompt_end=prompt_end,
                    lane=lane,
                )
            )
        return docs

    def build_batch_at(
        self,
        step: int,
        cursor: Optional[LoaderCursor] = None,
        use_live_cursor: bool = True,
    ) -> Tuple[PackedBatch, Any]:
        """Build the batch that would be produced at `step`.

        If use_live_cursor, advances self.cursor. Otherwise uses a provided
        cursor snapshot (for replay) without mutating trainer state beyond
        a temporary copy.
        """
        plan = self.schedule.get(step)
        if use_live_cursor:
            docs = self._select_docs(plan.lane)
        else:
            assert cursor is not None
            # Temporarily swap cursor
            saved = self.cursor
            self.cursor = cursor
            docs = self._select_docs(plan.lane)
            self.cursor = saved

        batch_id = f"{self.branch_id}:step{step:05d}"
        batch = pack_documents(
            docs,
            seq_len=self.seq_len,
            batch_id=batch_id,
            lane=plan.lane,
            policy=plan.packing_policy,
        )
        return batch, plan

    def _write_consumption(
        self,
        batch: PackedBatch,
        plan: Any,
        decision: OpusDecision,
        checkpoint_id: Optional[str],
        microbatch_id: int = 0,
    ) -> int:
        record = {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": self.global_step,
            "checkpoint_id": checkpoint_id,
            "microbatch_id": microbatch_id,
            "batch_id": batch.batch_id,
            "packed_sample_ids": batch.packed_sample_ids,
            "shard_ids": batch.shard_ids,
            "token_spans": batch.token_spans,
            "batch_hash": batch.batch_hash,
            "loss_mask_hash": batch.loss_mask_hash,
            "attention_policy": batch.attention_policy,
            "position_policy": "per_segment_reset",
            "packing_policy": batch.packing_policy,
            "mixture_lane": batch.lane,
            "curriculum_stage": plan.stage,
            "opus_decision_id": decision.decision_id,
            "opus_decision": decision.decision,
            "seq_len": batch.seq_len,
            "useful_tokens": batch.useful_tokens,
            "pad_count": batch.pad_count,
            "utilization": batch.utilization,
        }
        return self.ledgers.record_consumption(record)

    def _write_learning(
        self,
        batch: PackedBatch,
        result: StepLossResult,
        consumption_offset: int,
    ) -> int:
        # Sample-level and a compact token-level trace (non-zero loss positions)
        token_trace = []
        for i, (loss, m) in enumerate(zip(result.token_losses, batch.loss_mask)):
            if m == 1:
                token_trace.append({"index": i, "token_id": batch.token_ids[i], "loss": loss})
        # Cap trace size for artifact readability
        if len(token_trace) > 32:
            token_trace = token_trace[:16] + token_trace[-16:]

        record = {
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": self.global_step,
            "batch_id": batch.batch_id,
            "batch_hash": batch.batch_hash,
            "shard_ids": batch.shard_ids,
            "consumption_offset": consumption_offset,
            "mean_loss": result.mean_loss,
            "perplexity": result.perplexity,
            "gradient_norm": result.gradient_norm,
            "gradient_alignment": result.gradient_alignment,
            "useful_tokens": result.useful_tokens,
            "token_loss_trace": token_trace,
        }
        return self.ledgers.record_learning(record)

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        if not self.checkpoint_dir:
            raise RuntimeError("checkpoint_dir not set")
        ckpt_id = tag or f"ckpt_step{self.global_step:05d}"
        path = self.checkpoint_dir / f"{ckpt_id}.npz"
        payload = {
            "checkpoint_id": ckpt_id,
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": self.global_step,
            "ledger_offset": self.ledgers.consumption_offset(),
            "learning_offset": self.ledgers.learning.offset,
            "opus_offset": self.ledgers.opus.offset,
            "loader_cursor": self.cursor.to_dict(),
            "lane_counts": dict(self.lane_counts),
            "accepted_steps": self.accepted_steps,
            "model": self.model.state_dict(),
            "last_checkpoint_id": self.last_checkpoint_id,
        }
        # numpy-aware save
        np.savez_compressed(path, payload=np.array(payload, dtype=object))
        # Also write a JSON sidecar without large arrays for human audit
        meta = {
            "checkpoint_id": ckpt_id,
            "run_id": self.run_id,
            "branch_id": self.branch_id,
            "global_step": self.global_step,
            "ledger_offset": payload["ledger_offset"],
            "learning_offset": payload["learning_offset"],
            "opus_offset": payload["opus_offset"],
            "loader_cursor": payload["loader_cursor"],
            "lane_counts": payload["lane_counts"],
            "accepted_steps": payload["accepted_steps"],
            "model_step_count": self.model.step_count,
            "weights_shape": list(self.model.weights.shape),
            # Portable relative path (no machine-local absolute paths).
            "path": f"checkpoints/{ckpt_id}.npz",
        }
        meta_path = self.checkpoint_dir / f"{ckpt_id}.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.last_checkpoint_id = ckpt_id
        self.checkpoints.append(ckpt_id)
        return ckpt_id

    def load_checkpoint(self, ckpt_id: str) -> Dict[str, Any]:
        assert self.checkpoint_dir
        path = self.checkpoint_dir / f"{ckpt_id}.npz"
        data = np.load(path, allow_pickle=True)
        payload = data["payload"].item()
        self.global_step = int(payload["global_step"])
        self.cursor = LoaderCursor.from_dict(payload["loader_cursor"])
        self.lane_counts = dict(payload["lane_counts"])
        self.accepted_steps = int(payload["accepted_steps"])
        self.branch_id = payload["branch_id"]
        self.run_id = payload["run_id"]
        self.last_checkpoint_id = payload["checkpoint_id"]
        self.model.load_state_dict(payload["model"])
        return payload

    def train_steps(
        self,
        n_steps: int,
        checkpoint_every: int = 10,
        crash_at: Optional[int] = None,
    ) -> None:
        self._crash_at = crash_at
        end = self.global_step + n_steps
        while self.global_step < end:
            if crash_at is not None and self.global_step == crash_at:
                raise SimulatedCrash(
                    f"Simulated crash at global_step={self.global_step} "
                    f"after checkpoint={self.last_checkpoint_id}"
                )
            self._train_one_step(checkpoint_every=checkpoint_every)

    def _record_stream(self, batch: PackedBatch, plan: Any, decision: OpusDecision) -> None:
        rec = {
            "global_step": self.global_step,
            "batch_id": batch.batch_id,
            "batch_hash": batch.batch_hash,
            "shard_ids": list(batch.shard_ids),
            "token_spans": batch.token_spans,
            # Bit-exact resume/replay evidence: token ids + sequence positions.
            "token_ids": list(batch.token_ids),
            "position_ids": list(batch.position_ids),
            "loss_mask": list(batch.loss_mask),
            "segment_ids": list(batch.segment_ids),
            "lane": batch.lane,
            "stage": plan.stage,
            "packing_policy": batch.packing_policy,
            "opus_decision": decision.decision,
            "branch_id": self.branch_id,
            "loader_cursor_after": self.cursor.to_dict(),
        }
        self.batch_stream.append(rec)
        if self.stream_path:
            with self.stream_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    def _train_one_step(self, checkpoint_every: int = 10) -> Optional[PackedBatch]:
        t0 = time.perf_counter()
        step = self.global_step
        batch, plan = self.build_batch_at(step, use_live_cursor=True)
        t1 = time.perf_counter()

        decision = self.opus.evaluate(
            batch,
            global_step=step,
            protected_floors=plan.protected_floors,
            recent_lane_shares=self.recent_lane_shares(),
        )
        self.ledgers.record_opus(decision.to_dict())
        self._record_stream(batch, plan, decision)

        if not self.opus.is_trainable(decision):
            self.perf_events.append(
                {
                    "step": step,
                    "loader_s": t1 - t0,
                    "train_s": 0.0,
                    "useful_tokens": 0,
                    "raw_tokens": batch.seq_len,
                    "accepted": False,
                    "decision": decision.decision,
                    "utilization": batch.utilization,
                }
            )
            self.global_step += 1
            if checkpoint_every and self.global_step % checkpoint_every == 0:
                self.save_checkpoint()
            return None

        # Firewall: ensure no eval shards slipped in
        self.firewall.assert_no_eval_in_training(batch.shard_ids, context="train_step")

        result = self.model.forward(batch)
        self.model.backward_and_step(batch, result)
        t2 = time.perf_counter()

        offset = self._write_consumption(
            batch, plan, decision, checkpoint_id=self.last_checkpoint_id
        )
        self._write_learning(batch, result, consumption_offset=offset)

        self.lane_counts[batch.lane] = self.lane_counts.get(batch.lane, 0) + 1
        self.accepted_steps += 1
        self.perf_events.append(
            {
                "step": step,
                "loader_s": t1 - t0,
                "train_s": t2 - t1,
                "useful_tokens": batch.useful_tokens,
                "raw_tokens": batch.seq_len - batch.pad_count,
                "accepted": True,
                "decision": decision.decision,
                "utilization": batch.utilization,
            }
        )

        self.global_step += 1
        if checkpoint_every and self.global_step % checkpoint_every == 0:
            self.save_checkpoint()
        return batch
