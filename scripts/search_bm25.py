#!/usr/bin/env python3
"""Search a validated BM25 index and print structured JSON results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.retrieval.bm25 import BM25Error, BM25Retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = BM25Retriever(args.index_dir, args.corpus_dir).search(
            args.query, args.top_k
        )
        serialized = json.dumps(
            result, ensure_ascii=False, allow_nan=False, indent=2
        )
        if args.output is None:
            print(serialized)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
    except (BM25Error, OSError, ValueError) as exc:
        if isinstance(exc, BM25Error):
            message = str(exc)
        else:
            message = f"BM25_OUTPUT_WRITE_FAILED: {exc}"
        print(f"Error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
