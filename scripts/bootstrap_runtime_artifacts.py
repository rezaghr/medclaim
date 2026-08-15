#!/usr/bin/env python3
"""Create a small, transparent runtime corpus and its retrieval indexes.

The bundled corpus is intentionally small so a clean Docker checkout can start
without downloading external datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medclaim.corpus.scifact_corpus import (
    corpus_content_hash,
    sha256_text,
    whitespace_token_count,
)
from medclaim.retrieval.bm25 import BM25Retriever, build_bm25_index
from medclaim.retrieval.dense import DenseRetriever, build_dense_index
from medclaim.retrieval.embedding import OllamaEmbedder


STARTER_DOCUMENTS = (
    {
        "dataset": "scifact",
        "document_id": "scifact:document:bootstrap-1",
        "title": "Vitamin D and bone health",
        "source_url": "https://ods.od.nih.gov/factsheets/VitaminD-Consumer/",
        "text": (
            "Vitamin D helps the body absorb calcium. Calcium and vitamin D, together "
            "with other factors, help protect bones from becoming thin and brittle."
        ),
    },
    {
        "dataset": "scifact",
        "document_id": "scifact:document:bootstrap-2",
        "title": "Antibiotic use",
        "source_url": "https://www.cdc.gov/antibiotic-use/about/",
        "text": (
            "Antibiotics treat certain infections caused by bacteria. Antibiotics do "
            "not treat viral infections such as colds and flu."
        ),
    },
    {
        "dataset": "scifact",
        "document_id": "scifact:document:bootstrap-3",
        "title": "Smoking and health",
        "source_url": "https://www.cdc.gov/tobacco/about/",
        "text": (
            "Cigarette smoking harms nearly every organ of the body and causes many "
            "diseases, including cancer, heart disease, stroke, and lung disease."
        ),
    },
    {
        "dataset": "healthver",
        "document_id": "healthver:document:bootstrap-1",
        "title": "How vaccines work",
        "source_url": "https://www.who.int/news-room/feature-stories/detail/how-do-vaccines-work",
        "text": (
            "Vaccines train the immune system to recognize a pathogen and create "
            "antibodies without causing the disease that the vaccine prevents."
        ),
    },
    {
        "dataset": "healthver",
        "document_id": "healthver:document:bootstrap-2",
        "title": "High blood pressure",
        "source_url": "https://www.cdc.gov/high-blood-pressure/about/",
        "text": (
            "High blood pressure usually has no warning signs or symptoms. Measuring "
            "blood pressure is the way to know whether it is high."
        ),
    },
    {
        "dataset": "healthver",
        "document_id": "healthver:document:bootstrap-3",
        "title": "Adult sleep duration",
        "source_url": "https://www.cdc.gov/sleep/about/",
        "text": (
            "Adults generally need at least seven hours of sleep each day. Sleep needs "
            "vary by age and adequate sleep supports health and well-being."
        ),
    },
    {
        "dataset": "pubhealth",
        "document_id": "pubhealth:document:bootstrap-1",
        "title": "Hand hygiene",
        "source_url": "https://www.cdc.gov/clean-hands/about/",
        "text": (
            "Washing hands with soap removes germs and helps prevent respiratory and "
            "diarrheal infections from spreading."
        ),
    },
    {
        "dataset": "pubhealth",
        "document_id": "pubhealth:document:bootstrap-2",
        "title": "Physical activity",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
        "text": (
            "Regular physical activity provides significant physical and mental health "
            "benefits and helps prevent or manage several noncommunicable diseases."
        ),
    },
    {
        "dataset": "pubhealth",
        "document_id": "pubhealth:document:bootstrap-3",
        "title": "Tobacco",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/tobacco",
        "text": (
            "All forms of tobacco use are harmful, and there is no safe level of "
            "exposure to tobacco smoke."
        ),
    },
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_starter_corpus(output_root: Path, version: str) -> Path:
    """Build the immutable starter corpus if it is not already present."""
    output_dir = output_root / version
    if output_dir.exists():
        output_dir.chmod(0o755)
        return output_dir

    created_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    documents: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    for passage_number, source in enumerate(STARTER_DOCUMENTS, start=1):
        text = source["text"]
        documents.append(
            {
                **source,
                "source_document_id": source["document_id"].rsplit(":", 1)[-1],
                "source_type": "bootstrap_public_health_reference",
                "publication_year": None,
                "content_hash": sha256_text(text),
                "corpus_version": version,
                "metadata": {"bootstrap": True},
            }
        )
        passages.append(
            {
                # Short opaque IDs are deliberate: small local models copy them
                # more reliably into the strict evidence_used response field.
                "passage_id": f"bootstrap_{passage_number}",
                "document_id": source["document_id"],
                "dataset": source["dataset"],
                "passage_index": 0,
                "text": text,
                "start_char": 0,
                "end_char": len(text),
                "token_count": whitespace_token_count(text),
                "content_hash": sha256_text(text),
                "corpus_version": version,
                "metadata": {
                    "bootstrap": True,
                    "source_url": source["source_url"],
                    "source_type": "bootstrap_public_health_reference",
                    "source_sentence_indices": [],
                    "is_gold_for_any_claim": False,
                },
            }
        )

    manifest = {
        "artifact_type": "medical_evidence_corpus",
        "dataset": "multi_dataset",
        "datasets": ["scifact", "healthver", "pubhealth"],
        "corpus_version": version,
        "created_at": created_at,
        "builder_version": "bootstrap-1.0.0",
        "source_dataset_version": "bundled-starter-evidence-v1",
        "configuration": {"chunking_mode": "one_bootstrap_document_per_passage"},
        "document_count": len(documents),
        "passage_count": len(passages),
        "claim_count": 0,
        "gold_evidence_set_count": 0,
        "content_hash": corpus_content_hash(passages),
        "outputs": {
            "documents": "documents.jsonl",
            "passages": "passages.jsonl",
            "gold_evidence": "gold_evidence.jsonl",
            "quality_report": "quality_report.json",
        },
        "warnings": [
            "Bundled starter corpus only; claims outside these passages may lack evidence."
        ],
    }
    quality_report = {
        "status": "success",
        "corpus_version": version,
        "bootstrap": True,
        "document_count": len(documents),
        "passage_count": len(passages),
        "warning": manifest["warnings"][0],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}.tmp-", dir=output_root))
    try:
        _write_jsonl(temporary / "documents.jsonl", documents)
        _write_jsonl(temporary / "passages.jsonl", passages)
        (temporary / "gold_evidence.jsonl").write_text("", encoding="utf-8")
        _write_json(temporary / "quality_report.json", quality_report)
        _write_json(temporary / "manifest.json", manifest)
        temporary.chmod(0o755)
        os.rename(temporary, output_dir)
    except Exception:
        for child in temporary.iterdir() if temporary.exists() else ():
            child.unlink()
        if temporary.exists():
            temporary.rmdir()
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--corpus-version", default="bootstrap-corpus-v1")
    parser.add_argument("--bm25-version", default="bootstrap-bm25-v1")
    parser.add_argument("--dense-version", default="bootstrap-dense-v1")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--document-prefix", default="search_document: ")
    parser.add_argument("--query-prefix", default="search_query: ")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_dir = build_starter_corpus(args.artifact_root / "corpora", args.corpus_version)
    bm25_dir = args.artifact_root / "indexes" / args.bm25_version
    if not bm25_dir.exists():
        build_bm25_index(
            corpus_dir,
            args.artifact_root / "indexes",
            args.bm25_version,
        )

    dense_dir = args.artifact_root / "indexes" / args.dense_version
    query_embedder = OllamaEmbedder(
        args.embedding_model,
        base_url=args.ollama_base_url,
        timeout_seconds=args.timeout_seconds,
        input_prefix=args.query_prefix,
    )
    if not dense_dir.exists():
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
            batch_size=64,
            embedder=document_embedder,
            query_prefix=args.query_prefix,
            show_progress_bar=False,
        )

    # Fully load both retrievers so a completed init container guarantees that
    # the API can load the exact same immutable artifacts.
    BM25Retriever(bm25_dir, corpus_dir)
    DenseRetriever(dense_dir, corpus_dir, embedder=query_embedder)
    print(
        f"Runtime artifacts ready: {len(STARTER_DOCUMENTS)} starter passages, "
        f"BM25={args.bm25_version}, dense={args.dense_version}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
