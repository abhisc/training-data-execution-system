"""Immutable tokenized shards and JSON manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .tokenizer import FrozenTokenizer, sha256_bytes, sha256_file, sha256_text


LANE_META = {
    "web": {"language": "en", "script": "latin", "capability_lane": "web"},
    "code": {"language": "en", "script": "latin", "capability_lane": "code"},
    "math": {"language": "en", "script": "latin", "capability_lane": "math"},
    "indic": {"language": "multi", "script": "indic", "capability_lane": "indic"},
    "agentic": {"language": "en", "script": "latin", "capability_lane": "agentic"},
    "reasoning": {"language": "en", "script": "latin", "capability_lane": "reasoning"},
    "eval": {"language": "en", "script": "latin", "capability_lane": "eval"},
    "val": {"language": "en", "script": "latin", "capability_lane": "val"},
}

CLEANING_PIPELINE = "normalize_whitespace_v1"
CLEANING_PIPELINE_HASH = sha256_text(CLEANING_PIPELINE)


@dataclass
class DocumentSpan:
    document_id: str
    source_id: str
    start: int
    end: int
    prompt_end: Optional[int] = None  # for SFT/agentic: loss starts after this


@dataclass
class ShardManifest:
    shard_id: str
    source_ids: List[str]
    document_ids: List[str]
    content_hash: str
    tokenizer_hash: str
    token_count: int
    language: str
    script: str
    capability_lane: str
    license: str
    provenance_tier: str
    cleaning_pipeline_hash: str
    deduplication_status: str
    contamination_status: str
    evaluation_or_test_overlap_status: str
    never_train: bool
    role: str  # train | eval | val
    shard_path: str
    spans: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShardManifest":
        return cls(**data)


def _infer_prompt_end(text: str, token_ids: List[int], tokenizer: FrozenTokenizer) -> Optional[int]:
    """For agentic docs, mark loss start after the first 'Assistant:' turn if present."""
    marker = "Assistant:"
    idx = text.find(marker)
    if idx < 0:
        return None
    prefix = text[: idx + len(marker)]
    prefix_ids = tokenizer.encode(prefix, add_eos=False)
    # Cap within document length
    return min(len(prefix_ids), len(token_ids))


def create_shards_from_corpus(
    corpus_dir: Path,
    tokenizer: FrozenTokenizer,
    shards_dir: Path,
    manifests_dir: Path,
) -> List[ShardManifest]:
    """Tokenize corpus into immutable .bin shards and write JSON manifests."""
    corpus_dir = Path(corpus_dir)
    shards_dir = Path(shards_dir)
    manifests_dir = Path(manifests_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifests: List[ShardManifest] = []

    for lane_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        lane = lane_dir.name
        meta = LANE_META.get(lane, {"language": "en", "script": "latin", "capability_lane": lane})
        role = "train"
        never_train = False
        eval_overlap = "none"
        contamination = "clean"
        benchmark_id = None

        if lane == "eval":
            role = "eval"
            never_train = True
            eval_overlap = "eval_holdout"
            contamination = "eval_registered"
        elif lane == "val":
            role = "val"
            never_train = True
            eval_overlap = "val_holdout"
            contamination = "val_registered"

        for doc_path in sorted(lane_dir.glob("*.txt")):
            text = doc_path.read_text(encoding="utf-8")
            token_ids = tokenizer.encode(text, add_eos=True)
            arr = np.asarray(token_ids, dtype=np.uint32)

            document_id = f"{lane}/{doc_path.stem}"
            source_id = f"corpus:{lane}:{doc_path.name}"
            shard_id = f"shard_{lane}_{doc_path.stem}"
            shard_path = shards_dir / f"{shard_id}.bin"
            arr.tofile(shard_path)

            content_hash = sha256_bytes(arr.tobytes())
            prompt_end = None
            if lane == "agentic":
                prompt_end = _infer_prompt_end(text, token_ids, tokenizer)

            if lane == "eval":
                benchmark_id = f"demo_benchmark_{doc_path.stem}"
            elif lane == "val":
                benchmark_id = f"demo_validation_{doc_path.stem}"
            else:
                benchmark_id = None

            span = DocumentSpan(
                document_id=document_id,
                source_id=source_id,
                start=0,
                end=len(token_ids),
                prompt_end=prompt_end,
            )

            # Portable relative path for on-disk manifests (no machine-local abs paths).
            # Keep an absolute path in-memory for reliable loading during this process.
            rel_shard_path = f"{shards_dir.name}/{shard_id}.bin"
            abs_shard_path = str(shard_path.resolve())

            manifest = ShardManifest(
                shard_id=shard_id,
                source_ids=[source_id],
                document_ids=[document_id],
                content_hash=content_hash,
                tokenizer_hash=tokenizer.tokenizer_hash,
                token_count=int(len(token_ids)),
                language=meta["language"],
                script=meta["script"],
                capability_lane=meta["capability_lane"],
                license="CC-BY-4.0-demo",
                provenance_tier="synthetic_demo",
                cleaning_pipeline_hash=CLEANING_PIPELINE_HASH,
                deduplication_status="unique",
                contamination_status=contamination,
                evaluation_or_test_overlap_status=eval_overlap,
                never_train=never_train,
                role=role,
                shard_path=abs_shard_path,
                spans=[asdict(span)],
                benchmark_id=benchmark_id,
            )

            man_path = manifests_dir / f"{shard_id}.json"
            on_disk = manifest.to_dict()
            on_disk["shard_path"] = rel_shard_path
            man_path.write_text(json.dumps(on_disk, indent=2), encoding="utf-8")
            manifests.append(manifest)

    return manifests


def load_manifests(manifests_dir: Path) -> List[ShardManifest]:
    manifests_dir = Path(manifests_dir)
    out: List[ShardManifest] = []
    for path in sorted(manifests_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.append(ShardManifest.from_dict(data))
    return out


def resolve_shard_path(
    manifest: ShardManifest,
    base_dir: Optional[Path] = None,
) -> Path:
    """Resolve shard_path whether absolute or portable-relative."""
    p = Path(manifest.shard_path)
    if p.is_file():
        return p

    candidates: List[Path] = []
    if base_dir is not None:
        base_dir = Path(base_dir)
        candidates.extend(
            [
                base_dir / p,
                base_dir / p.name,
                base_dir / "shards" / p.name,
            ]
        )
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / p,
            cwd / "submission_artifacts" / p,
            cwd / "submission_artifacts" / "shards" / p.name,
            cwd / "shards" / p.name,
        ]
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    return p


def load_shard_tokens(
    manifest: ShardManifest,
    base_dir: Optional[Path] = None,
) -> np.ndarray:
    return np.fromfile(resolve_shard_path(manifest, base_dir), dtype=np.uint32)


def validate_manifest(
    manifest: ShardManifest,
    expected_tokenizer_hash: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> List[str]:
    """Return list of validation errors (empty if valid)."""
    errors: List[str] = []
    path = resolve_shard_path(manifest, base_dir)
    if not path.exists():
        errors.append(f"missing shard file: {manifest.shard_path}")
        return errors

    tokens = load_shard_tokens(manifest, base_dir)
    if int(tokens.size) != manifest.token_count:
        errors.append(
            f"{manifest.shard_id}: token_count mismatch "
            f"{tokens.size} != {manifest.token_count}"
        )
    actual_hash = sha256_bytes(tokens.tobytes())
    if actual_hash != manifest.content_hash:
        errors.append(f"{manifest.shard_id}: content_hash mismatch")
    if expected_tokenizer_hash and manifest.tokenizer_hash != expected_tokenizer_hash:
        errors.append(f"{manifest.shard_id}: tokenizer_hash mismatch")
    required = [
        "shard_id",
        "source_ids",
        "document_ids",
        "content_hash",
        "tokenizer_hash",
        "token_count",
        "language",
        "script",
        "capability_lane",
        "license",
        "provenance_tier",
        "cleaning_pipeline_hash",
        "deduplication_status",
        "contamination_status",
        "evaluation_or_test_overlap_status",
    ]
    data = manifest.to_dict()
    for key in required:
        if key not in data or data[key] in (None, "", []):
            errors.append(f"{manifest.shard_id}: missing field {key}")
    return errors


def validate_all_manifests(
    manifests: Sequence[ShardManifest],
    expected_tokenizer_hash: Optional[str] = None,
) -> Dict[str, Any]:
    all_errors: List[str] = []
    for m in manifests:
        all_errors.extend(validate_manifest(m, expected_tokenizer_hash))
    return {
        "ok": len(all_errors) == 0,
        "count": len(manifests),
        "errors": all_errors,
    }
