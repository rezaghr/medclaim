#!/usr/bin/env python3
"""Compare BM25, dense, and hybrid retrieval on one SciFact split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/retrieval/hybrid_scifact_v1.json"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.evaluation import EvaluationError, compare_retrieval  # noqa: E402
from medclaim.retrieval.configuration import (  # noqa: E402
    RetrievalConfigurationError,
    load_retrieval_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--gold-evidence", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--bm25-index-dir", type=Path, required=True)
    parser.add_argument("--dense-index-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sparse-top-k", type=int)
    parser.add_argument("--dense-top-k", type=int)
    parser.add_argument("--fusion-top-k", type=int)
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = load_retrieval_settings(args.config).with_overrides(
            sparse_top_k=args.sparse_top_k,
            dense_top_k=args.dense_top_k,
            fusion_top_k=args.fusion_top_k,
            rrf_k=args.rrf_k,
        )
        metrics = compare_retrieval(
            claims_path=args.claims,
            gold_evidence_path=args.gold_evidence,
            corpus_dir=args.corpus_dir,
            bm25_index_dir=args.bm25_index_dir,
            dense_index_dir=args.dense_index_dir,
            split=args.split,
            output_dir=args.output_dir,
            ks=args.ks,
            settings=settings,
            device=args.device,
        )
    except (EvaluationError, RetrievalConfigurationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Retrieval comparison completed successfully.\n")
    print(json.dumps(metrics, ensure_ascii=False, allow_nan=False, indent=2))
    print(f"\nOutput: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
