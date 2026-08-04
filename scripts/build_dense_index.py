#!/usr/bin/env python3
"""Build an immutable FAISS dense index over a versioned passage corpus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.retrieval.dense import DenseError, build_dense_index  # noqa: E402
from medclaim.retrieval.embedding import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingError,
    OllamaEmbedder,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/indexes/dense")
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--provider", choices=("sentence-transformers", "ollama"), default="sentence-transformers"
    )
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--document-prefix", default="search_document: ")
    parser.add_argument("--query-prefix", default="search_query: ")
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    try:
        resolved_device = resolve_device(args.device)
        embedder = (
            OllamaEmbedder(
                args.model,
                base_url=args.ollama_base_url,
                timeout_seconds=args.timeout_seconds,
                input_prefix=args.document_prefix,
            )
            if args.provider == "ollama"
            else None
        )
        output_dir = build_dense_index(
            corpus_dir=args.corpus_dir,
            output_root=args.output_root,
            version=args.version,
            model_id=args.model,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            device=args.device,
            embedder=embedder,
            query_prefix=args.query_prefix if args.provider == "ollama" else None,
        )
    except (DenseError, EmbeddingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    print("Dense index created successfully.\n")
    print(f"Index version: {manifest['index_version']}")
    print(f"Corpus version: {manifest['corpus']['version']}")
    print(f"Passages indexed: {manifest['corpus']['passage_count']}")
    print(f"Embedding model: {manifest['embedding']['model_id']}")
    print(f"Embedding dimension: {manifest['embedding']['dimension']}")
    print(f"Device: {'ollama-managed' if args.provider == 'ollama' else resolved_device}")
    print(f"Output: {output_dir}")
    print(f"Build time: {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
