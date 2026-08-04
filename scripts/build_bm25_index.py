#!/usr/bin/env python3
"""Build an immutable BM25 index over a versioned passage corpus."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.retrieval.bm25 import BM25Error, build_bm25_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/indexes/bm25")
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--epsilon", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        output_dir = build_bm25_index(
            corpus_dir=args.corpus_dir,
            output_root=args.output_root,
            version=args.version,
            k1=args.k1,
            b=args.b,
            epsilon=args.epsilon,
        )
    except BM25Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    import json

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    print("BM25 index created successfully.\n")
    print(f"Index version: {manifest['index_version']}")
    print(f"Corpus version: {manifest['corpus']['version']}")
    print(f"Passages indexed: {manifest['corpus']['passage_count']}")
    print(f"Tokenizer: {manifest['configuration']['tokenizer']}")
    print(f"BM25 k1: {manifest['configuration']['k1']}")
    print(f"BM25 b: {manifest['configuration']['b']}")
    print(f"Output: {output_dir}")
    print(f"Build time: {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
