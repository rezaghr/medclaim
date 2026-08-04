#!/usr/bin/env python3
"""Search a validated FAISS dense index and print structured JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.retrieval.dense import DenseError, DenseRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = DenseRetriever(
            args.index_dir, args.corpus_dir, device=args.device
        ).search(args.query, args.top_k)
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        if args.output is None:
            print(serialized)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
    except (DenseError, OSError, ValueError) as exc:
        message = str(exc) if isinstance(exc, DenseError) else f"DENSE_OUTPUT_WRITE_FAILED: {exc}"
        print(f"Error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
