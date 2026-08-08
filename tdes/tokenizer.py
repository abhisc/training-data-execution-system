"""Frozen BPE tokenizer with content-hash integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[EOS]", "[BOS]", "[SEP]"]
PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
BOS_ID = 3
SEP_ID = 4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


class FrozenTokenizer:
    """Thin wrapper around a frozen HuggingFace tokenizers BPE model."""

    def __init__(self, tokenizer: Tokenizer, path: Path, tokenizer_hash: str):
        self.tokenizer = tokenizer
        self.path = Path(path)
        self.tokenizer_hash = tokenizer_hash
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.eos_id = EOS_ID
        self.bos_id = BOS_ID
        self.sep_id = SEP_ID

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_eos and (not ids or ids[-1] != self.eos_id):
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def verify_hash(self) -> bool:
        return sha256_file(self.path) == self.tokenizer_hash


def collect_corpus_files(corpus_dir: Path) -> List[Path]:
    corpus_dir = Path(corpus_dir)
    files: List[Path] = []
    for path in sorted(corpus_dir.rglob("*.txt")):
        files.append(path)
    return files


def train_and_freeze_tokenizer(
    corpus_dir: Path,
    out_path: Path,
    vocab_size: int = 512,
) -> FrozenTokenizer:
    """Train a small BPE on the local corpus and freeze it to disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_corpus_files(corpus_dir)
    if not files:
        raise ValueError(f"No corpus files under {corpus_dir}")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
        min_frequency=1,
    )
    tokenizer.train([str(p) for p in files], trainer)
    tokenizer.save(str(out_path))

    tokenizer_hash = sha256_file(out_path)
    meta = {
        "tokenizer_path": str(out_path),
        "tokenizer_hash": tokenizer_hash,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "frozen": True,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return FrozenTokenizer(tokenizer, out_path, tokenizer_hash)


def load_frozen_tokenizer(path: Path) -> FrozenTokenizer:
    path = Path(path)
    tokenizer = Tokenizer.from_file(str(path))
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tokenizer_hash = meta["tokenizer_hash"]
    else:
        tokenizer_hash = sha256_file(path)
    return FrozenTokenizer(tokenizer, path, tokenizer_hash)
