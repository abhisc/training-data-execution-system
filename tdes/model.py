"""NumPy MockModel with deterministic, content-linked losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .packing import PackedBatch
from .tokenizer import PAD_ID


@dataclass
class StepLossResult:
    mean_loss: float
    token_losses: List[float]
    perplexity: float
    gradient_norm: float
    gradient_alignment: float
    useful_tokens: int


class MockModel:
    """Simulated model: real weight/optimizer arrays, deterministic losses.

    Loss for token t decays with exposure count of its shard and depends on
    a hash-mix of (weights snapshot, token id, position). Fully reproducible.
    """

    def __init__(self, dim: int = 32, seed: int = 42):
        self.dim = dim
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(dim, dim).astype(np.float64) * 0.02
        self.opt_state = {
            "m": np.zeros_like(self.weights),
            "v": np.zeros_like(self.weights),
            "t": 0,
            "lr": 1e-3,
        }
        self.rng_state = rng.get_state()
        self.exposure: Dict[str, int] = {}
        self.step_count = 0

    def _token_loss(self, token_id: int, pos: int, shard_id: str) -> float:
        exp = self.exposure.get(shard_id, 0)
        # Base CE-like loss ~ log(vocab) scaled; vocab proxy = 512 → ~6.24
        wsum = float(np.sum(self.weights) % 1.0)
        mix = ((token_id * 131 + pos * 17 + int(wsum * 1e6)) % 1000) / 1000.0
        base = 6.2 + 0.8 * mix
        # Decay with exposure (overfitting / memorization)
        decay = 1.0 / (1.0 + 0.15 * exp)
        return float(base * decay)

    def forward(self, batch: PackedBatch) -> StepLossResult:
        losses: List[float] = []
        # Use first shard for exposure if multiple
        primary_shard = batch.shard_ids[0] if batch.shard_ids else "unknown"
        for i, (tid, m, pos) in enumerate(
            zip(batch.token_ids, batch.loss_mask, batch.position_ids)
        ):
            if m == 0 or tid == PAD_ID:
                losses.append(0.0)
                continue
            # Map token to owning shard via spans
            shard = primary_shard
            for span in batch.token_spans:
                if span["start"] <= i < span["end"]:
                    shard = span["shard_id"]
                    break
            losses.append(self._token_loss(int(tid), int(pos), shard))

        useful = [l for l, m in zip(losses, batch.loss_mask) if m == 1]
        mean_loss = float(np.mean(useful)) if useful else 0.0
        ppl = float(np.exp(min(mean_loss, 20.0))) if useful else 1.0

        # Deterministic fake grad metrics from weights + loss
        g = mean_loss * (0.1 + abs(float(self.weights[0, 0])))
        grad_norm = float(abs(g) * np.sqrt(self.dim))
        grad_align = float(np.tanh(1.0 / (1.0 + mean_loss)))

        return StepLossResult(
            mean_loss=mean_loss,
            token_losses=losses,
            perplexity=ppl,
            gradient_norm=grad_norm,
            gradient_alignment=grad_align,
            useful_tokens=len(useful),
        )

    def backward_and_step(self, batch: PackedBatch, result: StepLossResult) -> None:
        """Update mock weights/optimizer and bump shard exposure."""
        self.opt_state["t"] += 1
        t = self.opt_state["t"]
        lr = self.opt_state["lr"]
        # Synthetic gradient proportional to mean loss
        grad = np.full_like(self.weights, result.mean_loss * 1e-3)
        grad[0, 0] += result.mean_loss * 1e-4
        m = self.opt_state["m"]
        v = self.opt_state["v"]
        beta1, beta2 = 0.9, 0.999
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
        mhat = m / (1 - beta1 ** t)
        vhat = v / (1 - beta2 ** t)
        self.weights -= lr * mhat / (np.sqrt(vhat) + 1e-8)

        for sid in batch.shard_ids:
            self.exposure[sid] = self.exposure.get(sid, 0) + 1
        self.step_count += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.copy(),
            "opt_state": {
                "m": self.opt_state["m"].copy(),
                "v": self.opt_state["v"].copy(),
                "t": self.opt_state["t"],
                "lr": self.opt_state["lr"],
            },
            "rng_state": self.rng_state,
            "exposure": dict(self.exposure),
            "step_count": self.step_count,
            "dim": self.dim,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.dim = state["dim"]
        self.weights = np.array(state["weights"], dtype=np.float64, copy=True)
        opt = state["opt_state"]
        self.opt_state = {
            "m": np.array(opt["m"], dtype=np.float64, copy=True),
            "v": np.array(opt["v"], dtype=np.float64, copy=True),
            "t": int(opt["t"]),
            "lr": float(opt["lr"]),
        }
        self.rng_state = state["rng_state"]
        self.exposure = dict(state["exposure"])
        self.step_count = int(state["step_count"])
