#!/usr/bin/env python3
"""Download the full evidence releases and build reusable retrieval indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medclaim.corpus.full import build_full_corpus, download_sources
from medclaim.retrieval.bm25 import BM25Retriever, build_bm25_index
from medclaim.retrieval.dense import DenseRetriever, build_dense_index
from medclaim.retrieval.embedding import OllamaEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--corpus-version", default="medical-corpus-full-v1")
    parser.add_argument("--bm25-version", default="medical-bm25-full-v1")
    parser.add_argument("--dense-version", default="medical-dense-full-v1")
    parser.add_argument("--max-passage-words", type=int, default=120)
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--document-prefix", default="search_document: ")
    parser.add_argument("--query-prefix", default="search_query: ")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.artifact_root.chmod(0o755)
    corpus_dir = args.artifact_root / "corpora" / args.corpus_version
    if not corpus_dir.is_dir():
        sources = download_sources(args.artifact_root / "downloads")
        corpus_dir = build_full_corpus(
            sources,
            args.artifact_root / "corpora",
            args.corpus_version,
            args.max_passage_words,
        )

    bm25_dir = args.artifact_root / "indexes" / args.bm25_version
    if not bm25_dir.is_dir():
        build_bm25_index(corpus_dir, args.artifact_root / "indexes", args.bm25_version)

    query_embedder = OllamaEmbedder(
        args.embedding_model,
        base_url=args.ollama_base_url,
        timeout_seconds=args.timeout_seconds,
        input_prefix=args.query_prefix,
    )
    dense_dir = args.artifact_root / "indexes" / args.dense_version
    if not dense_dir.is_dir():
        document_embedder = OllamaEmbedder(
            args.embedding_model,
            base_url=args.ollama_base_url,
            timeout_seconds=args.timeout_seconds,
            input_prefix=args.document_prefix,
        )
        build_dense_index(
            corpus_dir=corpus_dir,
            output_root=args.artifact_root / "indexes",
            version=args.dense_version,
            model_id=args.embedding_model,
            batch_size=args.embedding_batch_size,
            embedder=document_embedder,
            query_prefix=args.query_prefix,
            show_progress_bar=False,
        )

    # Loading both indexes verifies checksums and corpus compatibility before the
    # API is allowed to start.
    sparse = BM25Retriever(bm25_dir, corpus_dir)
    DenseRetriever(
        dense_dir,
        corpus_dir,
        embedder=query_embedder,
        corpus_data=(sparse.corpus_manifest, sparse.passages),
    )
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    print(
        "Runtime artifacts ready: "
        f"{manifest['document_count']} documents, {manifest['passage_count']} passages, "
        f"{manifest['claim_count']} claims; BM25={args.bm25_version}, "
        f"dense={args.dense_version}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
