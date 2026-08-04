#!/usr/bin/env python3
"""Search BM25 and dense indexes independently, then fuse them with RRF."""

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

from medclaim.retrieval.bm25 import BM25Error, BM25Retriever  # noqa: E402
from medclaim.retrieval.configuration import (  # noqa: E402
    RetrievalConfigurationError,
    load_retrieval_settings,
)
from medclaim.retrieval.dense import DenseError, DenseRetriever  # noqa: E402
from medclaim.retrieval.hybrid import HybridError, HybridRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--bm25-index-dir", type=Path, required=True)
    parser.add_argument("--dense-index-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sparse-top-k", type=int)
    parser.add_argument("--dense-top-k", type=int)
    parser.add_argument("--fusion-top-k", type=int)
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path)
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
        sparse = BM25Retriever(args.bm25_index_dir, args.corpus_dir)
        dense = DenseRetriever(
            args.dense_index_dir, args.corpus_dir, device=args.device
        )
        result = HybridRetriever(
            sparse,
            dense,
            sparse_top_k=settings.sparse_top_k,
            dense_top_k=settings.dense_top_k,
            fusion_top_k=settings.fusion_top_k,
            rrf_k=settings.rrf_k,
        ).search(args.query)
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        if args.output is None:
            print(serialized)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
    except (
        BM25Error,
        DenseError,
        HybridError,
        RetrievalConfigurationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
