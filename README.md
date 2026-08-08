# Training Data Execution System (TDES) — V5 Session 6

A small but complete, auditable training-data pipeline that proves correctness,
reproducibility, and efficiency — not scale.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
```

This single command regenerates the full `submission_artifacts/` tree:

```
submission_artifacts/
  run.log
  evidence.json
  evidence.md
  manifests/
  ledgers/
  checkpoints/
  performance.json
```

Run automated invariant tests:

```bash
pytest -q
```

## Architecture

```
documents → tokenized shards → manifests → mixture schedule → packing
         → OPUS gate → batches → training → consumption/learning ledgers
         → checkpoint → crash → resume → replay → fork → audit
```

| Module | Role |
|---|---|
| `tdes/tokenizer.py` | Train and freeze a local BPE; `tokenizer_hash = sha256(tokenizer.json)` |
| `tdes/sharding.py` | Immutable `.bin` shards + JSON manifests with all required provenance fields |
| `tdes/firewall.py` | Eval/val registry with `never_train`; executable dataloader admission check |
| `tdes/mixture.py` | Curriculum stages, lane weights, protected floors, compiled per-step schedule |
| `tdes/packing.py` | `pad_only`, `concat_chop`, `greedy`, `best_fit`, `structure_preserving` + masks |
| `tdes/opus.py` | Deterministic quality gate: accept / reject / defer / floor_override |
| `tdes/ledgers.py` | Append-only JSONL consumption, learning, and OPUS audit ledgers |
| `tdes/model.py` | NumPy `MockModel` with deterministic, content-linked losses |
| `tdes/trainer.py` | Step loop; checkpoints tied to ledger offsets + loader cursor |
| `tdes/runtime.py` | Crash simulation, resume proof, interval replay, branch fork |
| `tdes/perf.py` | Packing utilization and useful loss-bearing tokens/sec |
| `tdes/evidence.py` | Builds `evidence.json` / `evidence.md` by reading artifacts (never hardcoded) |

## Design decisions

1. **Determinism without global seeds.** Batch composition is a pure function of
   (manifests, mixture schedule, packing policy, step index, loader cursor).
   `batch_hash = sha256(token_ids ‖ loss_mask ‖ position_ids ‖ segment_ids)`.

2. **Mock model, real data system.** The NumPy model simulates forward/backward
   with reproducible losses that decay with shard exposure. The data path —
   shards, firewall, packing, OPUS, ledgers, checkpoint/resume/replay — is real.

3. **Frozen tokenizer.** A small BPE is trained on the bundled corpus and written
   to disk. Every manifest stores `tokenizer_hash`; the demo verifies it before
   training.

4. **Evaluation firewall.** Eval/val shards are registered with `never_train=true`.
   The loader probes and blocks them; consumption ledgers are audited to prove
   zero leakage.

5. **Protected floors + OPUS.** Rejected/deferred candidates on under-floor lanes
   can be rescued via `floor_override`, with reasons recorded in the OPUS audit trail.

6. **Crash / resume / replay / fork.** Checkpoints store model, optimizer, RNG,
   loader cursor, and ledger offsets. Resume proves the next batch id/hash matches
   the original stream. Replay reconstructs an interval from shards + schedule.
   Fork starts a new `branch_id` from an earlier checkpoint.

## Required PASS markers in `run.log`

```
[PASS] tokenizer_hash_verified
[PASS] eval_shard_blocked
[PASS] checkpoint_saved
[PASS] resume_next_batch_matched
[PASS] replay_hash_matched
```

## License

Demo code for educational submission.
