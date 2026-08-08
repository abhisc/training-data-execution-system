"""Packing policies, loss masks, attention segment ids, position ids."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tokenizer import PAD_ID, EOS_ID


@dataclass
class DocumentPiece:
    """A document (or span) to pack."""

    doc_id: str
    shard_id: str
    tokens: List[int]
    prompt_end: Optional[int] = None  # loss only after this within tokens
    lane: str = "web"


@dataclass
class PackedBatch:
    batch_id: str
    token_ids: List[int]
    loss_mask: List[int]
    position_ids: List[int]
    segment_ids: List[int]  # attention boundaries: same id can attend
    attention_policy: str
    packing_policy: str
    pad_count: int
    useful_tokens: int  # loss-bearing tokens
    packed_sample_ids: List[str]
    shard_ids: List[str]
    token_spans: List[Dict[str, Any]]
    lane: str
    seq_len: int
    batch_hash: str = ""
    loss_mask_hash: str = ""

    def compute_hashes(self) -> None:
        raw = (
            np.asarray(self.token_ids, dtype=np.uint32).tobytes()
            + np.asarray(self.loss_mask, dtype=np.uint8).tobytes()
            + np.asarray(self.position_ids, dtype=np.uint32).tobytes()
            + np.asarray(self.segment_ids, dtype=np.uint32).tobytes()
        )
        self.batch_hash = hashlib.sha256(raw).hexdigest()
        self.loss_mask_hash = hashlib.sha256(
            np.asarray(self.loss_mask, dtype=np.uint8).tobytes()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @property
    def utilization(self) -> float:
        if self.seq_len == 0:
            return 0.0
        return (self.seq_len - self.pad_count) / self.seq_len


def _loss_mask_for_doc(tokens: List[int], prompt_end: Optional[int]) -> List[int]:
    n = len(tokens)
    if prompt_end is None:
        return [1] * n
    pe = max(0, min(prompt_end, n))
    return [0] * pe + [1] * (n - pe)


def pack_pad_only(
    docs: Sequence[DocumentPiece],
    seq_len: int,
    batch_id: str,
    lane: str,
) -> PackedBatch:
    """Take first doc, pad/truncate to seq_len."""
    if not docs:
        tokens = [PAD_ID] * seq_len
        loss = [0] * seq_len
        pos = list(range(seq_len))
        seg = [0] * seq_len
        return PackedBatch(
            batch_id=batch_id,
            token_ids=tokens,
            loss_mask=loss,
            position_ids=pos,
            segment_ids=seg,
            attention_policy="causal_full",
            packing_policy="pad_only",
            pad_count=seq_len,
            useful_tokens=0,
            packed_sample_ids=[],
            shard_ids=[],
            token_spans=[],
            lane=lane,
            seq_len=seq_len,
        )

    doc = docs[0]
    toks = list(doc.tokens[:seq_len])
    mask = _loss_mask_for_doc(toks, doc.prompt_end)
    pad = seq_len - len(toks)
    tokens = toks + [PAD_ID] * pad
    loss = mask + [0] * pad
    pos = list(range(len(toks))) + [0] * pad
    seg = [1] * len(toks) + [0] * pad
    spans = [
        {
            "doc_id": doc.doc_id,
            "shard_id": doc.shard_id,
            "start": 0,
            "end": len(toks),
            "src_start": 0,
            "src_end": len(toks),
        }
    ]
    batch = PackedBatch(
        batch_id=batch_id,
        token_ids=tokens,
        loss_mask=loss,
        position_ids=pos,
        segment_ids=seg,
        attention_policy="causal_full",
        packing_policy="pad_only",
        pad_count=pad,
        useful_tokens=sum(loss),
        packed_sample_ids=[doc.doc_id],
        shard_ids=[doc.shard_id],
        token_spans=spans,
        lane=lane,
        seq_len=seq_len,
    )
    batch.compute_hashes()
    return batch


def pack_concat_chop(
    docs: Sequence[DocumentPiece],
    seq_len: int,
    batch_id: str,
    lane: str,
) -> PackedBatch:
    """Concatenate docs with EOS separators, chop to seq_len."""
    tokens: List[int] = []
    loss: List[int] = []
    pos: List[int] = []
    seg: List[int] = []
    spans: List[Dict[str, Any]] = []
    sample_ids: List[str] = []
    shard_ids: List[str] = []
    seg_id = 0

    for doc in docs:
        if len(tokens) >= seq_len:
            break
        seg_id += 1
        doc_mask = _loss_mask_for_doc(doc.tokens, doc.prompt_end)
        start = len(tokens)
        take = min(len(doc.tokens), seq_len - len(tokens))
        tokens.extend(doc.tokens[:take])
        loss.extend(doc_mask[:take])
        pos.extend(range(take))
        seg.extend([seg_id] * take)
        spans.append(
            {
                "doc_id": doc.doc_id,
                "shard_id": doc.shard_id,
                "start": start,
                "end": start + take,
                "src_start": 0,
                "src_end": take,
            }
        )
        sample_ids.append(doc.doc_id)
        if doc.shard_id not in shard_ids:
            shard_ids.append(doc.shard_id)

    pad = seq_len - len(tokens)
    tokens = tokens + [PAD_ID] * pad
    loss = loss + [0] * pad
    pos = pos + [0] * pad
    seg = seg + [0] * pad
    batch = PackedBatch(
        batch_id=batch_id,
        token_ids=tokens,
        loss_mask=loss,
        position_ids=pos,
        segment_ids=seg,
        attention_policy="causal_segmented",
        packing_policy="concat_chop",
        pad_count=pad,
        useful_tokens=sum(loss),
        packed_sample_ids=sample_ids,
        shard_ids=shard_ids,
        token_spans=spans,
        lane=lane,
        seq_len=seq_len,
    )
    batch.compute_hashes()
    return batch


def pack_greedy(
    docs: Sequence[DocumentPiece],
    seq_len: int,
    batch_id: str,
    lane: str,
) -> PackedBatch:
    """Place docs sequentially into the first available space."""
    return _pack_fit(docs, seq_len, batch_id, lane, policy="greedy", sort=False)


def pack_best_fit(
    docs: Sequence[DocumentPiece],
    seq_len: int,
    batch_id: str,
    lane: str,
) -> PackedBatch:
    """Sort by length descending and pack tightly."""
    return _pack_fit(docs, seq_len, batch_id, lane, policy="best_fit", sort=True)


def pack_structure_preserving(
    docs: Sequence[DocumentPiece],
    seq_len: int,
    batch_id: str,
    lane: str,
) -> PackedBatch:
    """Pack with isolated segments; no cross-doc attention (segment ids differ)."""
    batch = _pack_fit(
        docs, seq_len, batch_id, lane, policy="structure_preserving", sort=False
    )
    batch.attention_policy = "block_diagonal_segments"
    batch.compute_hashes()
    return batch


def _pack_fit(
    docs: Sequence[DocumentPiece],
    seq_len: int,
    batch_id: str,
    lane: str,
    policy: str,
    sort: bool,
) -> PackedBatch:
    items = list(docs)
    if sort:
        items = sorted(items, key=lambda d: len(d.tokens), reverse=True)

    tokens: List[int] = []
    loss: List[int] = []
    pos: List[int] = []
    seg: List[int] = []
    spans: List[Dict[str, Any]] = []
    sample_ids: List[str] = []
    shard_ids: List[str] = []
    remaining = seq_len
    seg_id = 0

    # For best-fit: choose the doc that leaves the smallest positive remainder
    pool = list(items)
    while pool and remaining > 0:
        if policy == "best_fit":
            candidates = [d for d in pool if len(d.tokens) <= remaining]
            if not candidates:
                # take truncated longest that fits partially
                candidates = sorted(pool, key=lambda d: len(d.tokens), reverse=True)[:1]
            best = min(candidates, key=lambda d: remaining - min(len(d.tokens), remaining))
            doc = best
            pool.remove(doc)
        else:
            doc = pool.pop(0)

        take = min(len(doc.tokens), remaining)
        if take <= 0:
            break
        seg_id += 1
        doc_mask = _loss_mask_for_doc(doc.tokens[:take], doc.prompt_end)
        start = len(tokens)
        tokens.extend(doc.tokens[:take])
        loss.extend(doc_mask)
        # Position ids reset per segment for structure-preserving / best isolation
        pos.extend(range(take))
        seg.extend([seg_id] * take)
        spans.append(
            {
                "doc_id": doc.doc_id,
                "shard_id": doc.shard_id,
                "start": start,
                "end": start + take,
                "src_start": 0,
                "src_end": take,
            }
        )
        sample_ids.append(doc.doc_id)
        if doc.shard_id not in shard_ids:
            shard_ids.append(doc.shard_id)
        remaining -= take

    pad = seq_len - len(tokens)
    tokens = tokens + [PAD_ID] * pad
    loss = loss + [0] * pad
    pos = pos + [0] * pad
    seg = seg + [0] * pad

    attn = "causal_segmented"
    if policy == "structure_preserving":
        attn = "block_diagonal_segments"

    batch = PackedBatch(
        batch_id=batch_id,
        token_ids=tokens,
        loss_mask=loss,
        position_ids=pos,
        segment_ids=seg,
        attention_policy=attn,
        packing_policy=policy,
        pad_count=pad,
        useful_tokens=int(sum(loss)),
        packed_sample_ids=sample_ids,
        shard_ids=shard_ids,
        token_spans=spans,
        lane=lane,
        seq_len=seq_len,
    )
    batch.compute_hashes()
    return batch


POLICIES = {
    "pad_only": pack_pad_only,
    "concat_chop": pack_concat_chop,
    "greedy": pack_greedy,
    "best_fit": pack_best_fit,
    "structure_preserving": pack_structure_preserving,
}


def pack_documents(
    docs: Sequence[DocumentPiece],
    seq_len: int,
    batch_id: str,
    lane: str,
    policy: str,
) -> PackedBatch:
    if policy not in POLICIES:
        raise ValueError(f"Unknown packing policy: {policy}")
    return POLICIES[policy](docs, seq_len, batch_id, lane)


def attention_allows_cross_segment(batch: PackedBatch, i: int, j: int) -> bool:
    """Whether token i may attend to token j under the batch attention policy."""
    if j > i:
        return False  # causal
    if batch.token_ids[i] == PAD_ID or batch.token_ids[j] == PAD_ID:
        return False
    if batch.attention_policy in ("block_diagonal_segments", "causal_segmented"):
        return batch.segment_ids[i] == batch.segment_ids[j] and batch.segment_ids[i] != 0
    return True
