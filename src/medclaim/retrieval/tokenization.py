"""Deterministic tokenization for sparse retrieval."""

from __future__ import annotations

import re
import unicodedata

TOKENIZER_NAME = "simple-alphanumeric-v1"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize with NFKC normalization and lowercase ASCII alphanumerics."""
    if not isinstance(text, str):
        return []
    normalized = unicodedata.normalize("NFKC", text)
    return TOKEN_PATTERN.findall(normalized.lower())
