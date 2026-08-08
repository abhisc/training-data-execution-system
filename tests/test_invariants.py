"""Invariant tests for the Training Data Execution System."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tdes.firewall import EvalFirewall
from tdes.ledgers import LedgerSuite
from tdes.mixture import MixtureSchedule, TRAIN_LANES
from tdes.model import MockModel
from tdes.opus import OpusGate
from tdes.packing import (
    DocumentPiece,
    attention_allows_cross_segment,
    pack_documents,
)
from tdes.runtime import prove_resume, replay_interval, run_crash_and_resume
from tdes.sharding import (
    create_shards_from_corpus,
    load_shard_tokens,
    validate_all_manifests,
)
from tdes.tokenizer import train_and_freeze_tokenizer
from tdes.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("tdes")
    tok = train_and_freeze_tokenizer(CORPUS, base / "tok" / "tokenizer.json", vocab_size=512)
    manifests = create_shards_from_corpus(
        CORPUS, tok, base / "shards", base / "manifests"
    )
    return {"base": base, "tokenizer": tok, "manifests": manifests}


def test_tokenizer_hash_stable(workspace):
    tok = workspace["tokenizer"]
    assert tok.verify_hash()
    # Re-hash file matches stored hash
    from tdes.tokenizer import sha256_file

    assert sha256_file(tok.path) == tok.tokenizer_hash


def test_manifest_integrity(workspace):
    manifests = workspace["manifests"]
    tok = workspace["tokenizer"]
    result = validate_all_manifests(manifests, tok.tokenizer_hash)
    assert result["ok"], result["errors"]
    # Immutability: mutating file breaks validation
    m = manifests[0]
    path = Path(m.shard_path)
    original = path.read_bytes()
    path.write_bytes(original + b"\x00\x00\x00\x00")
    result2 = validate_all_manifests([m], tok.tokenizer_hash)
    path.write_bytes(original)  # restore
    assert not result2["ok"]


def test_firewall_blocks_eval(workspace):
    manifests = workspace["manifests"]
    fw = EvalFirewall()
    fw.register_from_manifests(manifests)
    eval_ids = [m.shard_id for m in manifests if m.never_train]
    assert eval_ids
    assert fw.check_admission(eval_ids[0], context="test") is False
    assert fw.blocked_events
    train_id = next(m.shard_id for m in manifests if not m.never_train)
    assert fw.check_admission(train_id) is True


def test_packing_masks_and_segments(workspace):
    manifests = [m for m in workspace["manifests"] if not m.never_train]
    docs = []
    for m in manifests[:3]:
        toks = load_shard_tokens(m).tolist()
        pe = m.spans[0].get("prompt_end") if m.spans else None
        docs.append(
            DocumentPiece(m.document_ids[0], m.shard_id, toks, pe, m.capability_lane)
        )
    batch = pack_documents(docs, 64, "t:struct", "agentic", "structure_preserving")
    assert len(batch.token_ids) == 64
    assert batch.useful_tokens == sum(batch.loss_mask)
    # No cross-segment attention
    for i in range(64):
        for j in range(i + 1):
            if (
                batch.segment_ids[i]
                and batch.segment_ids[j]
                and batch.segment_ids[i] != batch.segment_ids[j]
            ):
                assert not attention_allows_cross_segment(batch, i, j)
    # Position ids reset per segment
    for seg in set(batch.segment_ids) - {0}:
        idxs = [k for k, s in enumerate(batch.segment_ids) if s == seg]
        assert [batch.position_ids[k] for k in idxs] == list(range(len(idxs)))


def test_agentic_loss_mask_zeros_prompt(workspace):
    agentic = [
        m
        for m in workspace["manifests"]
        if m.capability_lane == "agentic" and not m.never_train
    ]
    assert agentic
    m = agentic[0]
    pe = m.spans[0].get("prompt_end")
    assert pe is not None and pe > 0
    toks = load_shard_tokens(m).tolist()
    doc = DocumentPiece(m.document_ids[0], m.shard_id, toks, pe, "agentic")
    batch = pack_documents([doc], 64, "t:sft", "agentic", "pad_only")
    assert all(v == 0 for v in batch.loss_mask[: min(pe, 64)])


def test_mixture_floors_and_shares():
    sched = MixtureSchedule(total_steps=60)
    sched.compile()
    shares = sched.planned_shares()
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    for stage in sched.stages:
        for lane, floor in stage.protected_floors.items():
            # Over full stage window, scheduled share should be near weight and >= ~half floor
            stage_shares = {}
            counts = {l: 0 for l in TRAIN_LANES}
            n = 0
            for plan in sched.steps:
                if plan.stage == stage.name:
                    counts[plan.lane] += 1
                    n += 1
            if n:
                stage_shares = {l: counts[l] / n for l in TRAIN_LANES}
                assert stage_shares.get(lane, 0) >= floor * 0.5


def test_opus_all_decision_types(workspace, tmp_path):
    manifests = workspace["manifests"]
    fw = EvalFirewall()
    fw.register_from_manifests(manifests)
    ledgers = LedgerSuite(tmp_path / "ledgers")
    schedule = MixtureSchedule(total_steps=30)
    schedule.compile()
    trainer = Trainer(
        manifests=manifests,
        schedule=schedule,
        firewall=fw,
        model=MockModel(seed=1),
        ledgers=ledgers,
        opus=OpusGate(),
        seq_len=64,
        checkpoint_dir=tmp_path / "ckpt",
    )
    trainer.train_steps(n_steps=25, checkpoint_every=10)
    decisions = {d.decision for d in trainer.opus.decisions}
    assert {"accept", "reject", "defer", "floor_override"} <= decisions


def test_resume_no_skip_or_duplicate(workspace, tmp_path):
    manifests = workspace["manifests"]
    fw = EvalFirewall()
    fw.register_from_manifests(manifests)
    ledgers = LedgerSuite(tmp_path / "ledgers")
    schedule = MixtureSchedule(total_steps=40)
    schedule.compile()
    trainer = Trainer(
        manifests=manifests,
        schedule=schedule,
        firewall=fw,
        model=MockModel(seed=2),
        ledgers=ledgers,
        opus=OpusGate(),
        seq_len=64,
        checkpoint_dir=tmp_path / "ckpt",
    )
    result = run_crash_and_resume(
        trainer, total_steps=30, crash_at=15, checkpoint_every=10
    )
    assert result["crashed"]
    assert result["proof"]["ok"], result["proof"]
    # Original stream has exactly one record for the resume step; hash matches proof
    step = result["proof"]["step"]
    at_step = [r for r in trainer.batch_stream if r["global_step"] == step]
    assert len(at_step) == 1
    assert at_step[0]["batch_hash"] == result["proof"]["expected_batch_hash"]
    assert at_step[0]["batch_hash"] == result["proof"]["resumed_batch_hash"]
    # Bit-exact: token ids, positions, and masks must match (no single-token drift)
    assert result["proof"]["token_ids_match"] is True
    assert result["proof"]["position_ids_match"] is True
    assert result["proof"]["loss_mask_match"] is True
    assert "token_ids" in at_step[0] and "position_ids" in at_step[0]
    # Immediate predecessor exists (no skipped step in the original stream)
    assert any(r["global_step"] == step - 1 for r in trainer.batch_stream)


def test_replay_hashes_match(workspace, tmp_path):
    manifests = workspace["manifests"]
    fw = EvalFirewall()
    fw.register_from_manifests(manifests)
    ledgers = LedgerSuite(tmp_path / "ledgers")
    schedule = MixtureSchedule(total_steps=30)
    schedule.compile()
    trainer = Trainer(
        manifests=manifests,
        schedule=schedule,
        firewall=fw,
        model=MockModel(seed=3),
        ledgers=ledgers,
        opus=OpusGate(),
        seq_len=64,
        checkpoint_dir=tmp_path / "ckpt",
    )
    trainer.train_steps(n_steps=20, checkpoint_every=10)
    proof = replay_interval(
        trainer, start_step=3, end_step=12, original_stream=trainer.batch_stream
    )
    assert proof["ok"], proof
    for cmp in proof["comparisons"]:
        assert cmp["ok"]
        assert cmp["token_ids_match"] is True
        assert cmp["position_ids_match"] is True
        assert cmp["loss_mask_match"] is True
        assert cmp["original_hash"] == cmp["replay_hash"]


def test_manifest_paths_are_not_machine_local(workspace):
    """On-disk manifests must use portable relative shard_path values."""
    manifests_dir = workspace["base"] / "manifests"
    for path in manifests_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        shard_path = data["shard_path"]
        assert not Path(shard_path).is_absolute(), shard_path
        assert "Users" not in shard_path and "home" not in shard_path.lower()
        assert shard_path.startswith("shards/")


def test_no_eval_in_consumption(workspace, tmp_path):
    manifests = workspace["manifests"]
    fw = EvalFirewall()
    fw.register_from_manifests(manifests)
    blocked = fw.blocked_ids()
    ledgers = LedgerSuite(tmp_path / "ledgers")
    schedule = MixtureSchedule(total_steps=20)
    schedule.compile()
    trainer = Trainer(
        manifests=manifests,
        schedule=schedule,
        firewall=fw,
        model=MockModel(seed=4),
        ledgers=ledgers,
        opus=OpusGate(),
        seq_len=64,
        checkpoint_dir=tmp_path / "ckpt",
    )
    trainer.train_steps(n_steps=15, checkpoint_every=5)
    for row in ledgers.consumption.read_all():
        assert not (set(row["shard_ids"]) & blocked)


def test_evidence_points_to_real_artifacts():
    artifacts = ROOT / "submission_artifacts"
    if not (artifacts / "evidence.json").exists():
        pytest.skip("Run python run_demo.py first")
    evidence = json.loads((artifacts / "evidence.json").read_text())
    assert evidence["all_passed"]
    log = (artifacts / "run.log").read_text()
    for name in [
        "tokenizer_hash_verified",
        "eval_shard_blocked",
        "checkpoint_saved",
        "resume_next_batch_matched",
        "replay_hash_matched",
    ]:
        assert f"[PASS] {name}" in log
    # Required chronological EVENT narrative
    events = [
        "shards created",
        "manifests validated",
        "evaluation data blocked",
        "mixture compiled",
        "batches packed",
        "OPUS decisions recorded",
        "checkpoint saved",
        "crash simulated",
        "run resumed",
        "historical stream replayed",
        "branch forked",
        "audit completed",
        "performance measured",
    ]
    positions = [log.index(f"EVENT {e}") for e in events]
    assert positions == sorted(positions), list(zip(events, positions))
    assert (artifacts / "performance.json").exists()
    assert (artifacts / "evidence.md").exists()
    assert list((artifacts / "manifests").glob("*.json"))
    assert (artifacts / "ledgers" / "consumption.jsonl").exists()
    assert (artifacts / "ledgers" / "learning.jsonl").exists()
    # No machine-local absolute paths in submitted manifests / checkpoints
    for path in (artifacts / "manifests").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert not Path(data["shard_path"]).is_absolute()
    for path in (artifacts / "checkpoints").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert not Path(data["path"]).is_absolute()
