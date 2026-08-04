#!/usr/bin/env python3
"""Evaluate BM25 retrieval against SciFact gold evidence passage mappings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.evaluation import EvaluationError, evaluate_bm25  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--gold-evidence", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = evaluate_bm25(
            claims_path=args.claims,
            gold_evidence_path=args.gold_evidence,
            corpus_dir=args.corpus_dir,
            index_dir=args.index_dir,
            split=args.split,
            output_dir=args.output_dir,
            ks=args.ks,
        )
    except EvaluationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("BM25 retrieval evaluation completed successfully.\n")
    print(json.dumps(metrics, ensure_ascii=False, allow_nan=False, indent=2))
    print(f"\nOutput: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
