"""Corpus hashing helpers shared by dataset builders and runtime validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def corpus_content_hash(passages: list[dict[str, Any]]) -> str:
    """Return the deterministic hash used by corpus and index manifests."""
    hasher = hashlib.sha256()
    for passage in passages:
        serialized = json.dumps(
            passage,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        hasher.update(serialized.encode("utf-8"))
        hasher.update(b"\n")
    return f"sha256:{hasher.hexdigest()}"
