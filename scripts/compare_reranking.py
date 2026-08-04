#!/usr/bin/env python3
"""Compare hybrid evidence selection with and without cross-encoder reranking."""

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

from medclaim.evaluation import EvaluationError, compare_reranking  # noqa: E402
from medclaim.reranking import CrossEncoderReranker, RerankerError  # noqa: E402
from medclaim.reranking.models import (  # noqa: E402
    RerankingConfigurationError,
    load_reranking_configuration,
)
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
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--final-evidence-k", type=int)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--maximum-input-length", type=int)
    parser.add_argument("--max-claims", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hybrid-config", type=Path, default=DEFAULT_HYBRID_CONFIG)
    parser.add_argument(
        "--reranker-config", type=Path, default=DEFAULT_RERANKER_CONFIG
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        hybrid_settings = load_retrieval_settings(args.hybrid_config)
        reranking = load_reranking_configuration(args.reranker_config).with_overrides(
            enabled=True,
            model_id=args.model,
            model_revision=args.model_revision,
            candidate_count=args.candidate_count,
            final_evidence_k=args.final_evidence_k,
            batch_size=args.batch_size,
            device=args.device,
            maximum_input_length=args.maximum_input_length,
        )
        reranker = CrossEncoderReranker(
            model_id=reranking.model_id,
            model_revision=reranking.model_revision,
            device=reranking.device,
            batch_size=reranking.batch_size,
            maximum_input_length=reranking.maximum_input_length,
        )
        result = compare_reranking(
            claims_path=args.claims,
            gold_evidence_path=args.gold_evidence,
            corpus_dir=args.corpus_dir,
            bm25_index_dir=args.bm25_index_dir,
            dense_index_dir=args.dense_index_dir,
            split=args.split,
            output_dir=args.output_dir,
            reranker=reranker,
            reranking_configuration=reranking,
            retrieval_settings=hybrid_settings,
            max_claims=args.max_claims,
            device=reranking.device,
        )
    except (
        EvaluationError,
        RerankerError,
        RetrievalConfigurationError,
        RerankingConfigurationError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Reranking ablation completed successfully.\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if result["classification_metrics"]["status"] == "not_available":
        print(
            "\nClassification metrics were not run because no verifier is configured.",
            file=sys.stderr,
        )
    print(f"\nOutput: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
