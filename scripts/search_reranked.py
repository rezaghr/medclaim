#!/usr/bin/env python3
"""Search hybrid candidates and optionally rerank them with a cross-encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_HYBRID_CONFIG = PROJECT_ROOT / "configs/retrieval/hybrid_scifact_v1.json"
DEFAULT_RERANKER_CONFIG = (
    PROJECT_ROOT / "configs/reranking/scifact_cross_encoder_v1.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.reranking import CrossEncoderReranker, RerankerError  # noqa: E402
from medclaim.reranking.models import (  # noqa: E402
    RerankingConfigurationError,
    load_reranking_configuration,
)
from medclaim.retrieval.bm25 import BM25Error, BM25Retriever  # noqa: E402
from medclaim.retrieval.configuration import (  # noqa: E402
    RetrievalConfigurationError,
    load_retrieval_settings,
)
from medclaim.retrieval.dense import DenseError, DenseRetriever  # noqa: E402
from medclaim.retrieval.hybrid import HybridError, HybridRetriever  # noqa: E402
from medclaim.retrieval.reranked import (  # noqa: E402
    RerankedRetrievalError,
    RerankedRetriever,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--bm25-index-dir", type=Path, required=True)
    parser.add_argument("--dense-index-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--maximum-input-length", type=int)
    parser.add_argument("--disable-reranking", action="store_true")
    parser.add_argument("--hybrid-config", type=Path, default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument(
        "--reranker-config", type=Path, default=DEFAULT_RERANKER_CONFIG
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        hybrid_settings = load_retrieval_settings(args.hybrid_config)
        reranking = load_reranking_configuration(args.reranker_config).with_overrides(
            enabled=False if args.disable_reranking else None,
            model_id=args.model,
            model_revision=args.model_revision,
            candidate_count=args.candidate_count,
            final_evidence_k=args.top_k,
            batch_size=args.batch_size,
            device=args.device,
            maximum_input_length=args.maximum_input_length,
        )
        sparse = BM25Retriever(args.bm25_index_dir, args.corpus_dir)
        dense = DenseRetriever(
            args.dense_index_dir, args.corpus_dir, device=reranking.device
        )
        hybrid = HybridRetriever(
            sparse,
            dense,
            sparse_top_k=hybrid_settings.sparse_top_k,
            dense_top_k=hybrid_settings.dense_top_k,
            fusion_top_k=max(
                hybrid_settings.fusion_top_k, reranking.candidate_count
            ),
            rrf_k=hybrid_settings.rrf_k,
        )
        reranker = (
            CrossEncoderReranker(
                model_id=reranking.model_id,
                model_revision=reranking.model_revision,
                device=reranking.device,
                batch_size=reranking.batch_size,
                maximum_input_length=reranking.maximum_input_length,
            )
            if reranking.enabled
            else None
        )
        result = RerankedRetriever(hybrid, reranker, reranking).search(args.query)
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
        RerankerError,
        RerankedRetrievalError,
        RetrievalConfigurationError,
        RerankingConfigurationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
