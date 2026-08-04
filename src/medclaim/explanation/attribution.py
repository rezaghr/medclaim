"""Resolve verifier-selected passage IDs to authoritative corpus metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medclaim.corpus.scifact_corpus import corpus_content_hash


class AttributionError(Exception):
    """Raised when citations cannot be resolved to one validated corpus."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise AttributionError(f"ATTRIBUTION_CORPUS_INVALID: Could not read {path}: {exc}.") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be an object")
                rows.append(value)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise AttributionError(f"ATTRIBUTION_CORPUS_INVALID: Could not parse {path}: {exc}.") from exc
    return rows


class CorpusResolver:
    def __init__(self, corpus_dir: Path) -> None:
        manifest = _load_json(corpus_dir / "manifest.json")
        passages = _load_jsonl(corpus_dir / "passages.jsonl")
        documents = _load_jsonl(corpus_dir / "documents.jsonl")
        if not isinstance(manifest, dict) or manifest.get("artifact_type") != "medical_evidence_corpus":
            raise AttributionError("ATTRIBUTION_CORPUS_INVALID: Expected a medical evidence corpus.")
        if manifest.get("passage_count") != len(passages) or manifest.get("document_count") != len(documents):
            raise AttributionError("ATTRIBUTION_CORPUS_INVALID: Manifest counts do not match corpus files.")
        if manifest.get("content_hash") != corpus_content_hash(passages):
            raise AttributionError("ATTRIBUTION_CORPUS_INVALID: Corpus passage hash mismatch.")
        self.corpus_version = manifest.get("corpus_version")
        self.passages = self._index(passages, "passage_id", "passage")
        self.documents = self._index(documents, "document_id", "document")
        for passage in passages:
            if passage.get("corpus_version") != self.corpus_version:
                raise AttributionError("ATTRIBUTION_CORPUS_VERSION_MISMATCH: Passage uses another corpus version.")
            if passage.get("document_id") not in self.documents:
                raise AttributionError("ATTRIBUTION_DOCUMENT_NOT_FOUND: Passage document cannot be resolved.")

    @staticmethod
    def _index(rows: list[dict[str, Any]], field: str, kind: str) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for row in rows:
            value = row.get(field)
            if not isinstance(value, str) or not value or value in output:
                raise AttributionError(f"ATTRIBUTION_CORPUS_INVALID: Invalid or duplicate {kind} ID.")
            output[value] = row
        return output

    def resolve(
        self,
        result: dict[str, Any],
        supplied_passages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        supplied_by_id = {
            item.get("passage_id"): item
            for item in supplied_passages
            if isinstance(item, dict) and isinstance(item.get("passage_id"), str)
        }
        if len(supplied_by_id) != len(supplied_passages):
            raise AttributionError("ATTRIBUTION_SUPPLIED_EVIDENCE_INVALID: Supplied passage IDs are invalid or duplicated.")
        used = result.get("evidence_used")
        if not isinstance(used, list) or len(used) != len(set(used)):
            raise AttributionError("ATTRIBUTION_CITATION_INVALID: evidence_used is invalid or duplicated.")
        component_ids: dict[str, list[str]] = {}
        for component in result.get("component_results", []):
            if not isinstance(component, dict):
                continue
            component_id = component.get("component_id")
            for passage_id in component.get("evidence_used", []):
                if isinstance(component_id, str):
                    component_ids.setdefault(passage_id, []).append(component_id)
        output: list[dict[str, Any]] = []
        for passage_id in used:
            if passage_id not in supplied_by_id:
                raise AttributionError(f"ATTRIBUTION_UNSUPPLIED_CITATION: Passage {passage_id!r} was not supplied.")
            passage = self.passages.get(passage_id)
            if passage is None:
                raise AttributionError(f"ATTRIBUTION_UNKNOWN_PASSAGE: Passage {passage_id!r} is not in the corpus.")
            supplied = supplied_by_id[passage_id]
            if supplied.get("corpus_version", self.corpus_version) != self.corpus_version:
                raise AttributionError("ATTRIBUTION_CORPUS_VERSION_MISMATCH: Citation belongs to another corpus version.")
            document = self.documents.get(passage["document_id"])
            if document is None:
                raise AttributionError("ATTRIBUTION_DOCUMENT_NOT_FOUND: Citation document cannot be resolved.")
            metadata = passage.get("metadata", {})
            output.append(
                {
                    "passage_id": passage_id,
                    "document_id": passage["document_id"],
                    "dataset": passage["dataset"],
                    "source_type": metadata.get("source_type", document.get("source_type")),
                    "title": document.get("title"),
                    "source_url": document.get("source_url"),
                    "publication_year": document.get("publication_year"),
                    "text": passage["text"],
                    "used_by_verifier": True,
                    "component_ids": component_ids.get(passage_id, []),
                    "corpus_version": self.corpus_version,
                    "retrieval": {
                        "bm25_rank": supplied.get("bm25_rank"),
                        "bm25_score": supplied.get("bm25_score"),
                        "dense_rank": supplied.get("dense_rank"),
                        "dense_score": supplied.get("dense_score"),
                        "fusion_rank": supplied.get("pre_rerank_rank", supplied.get("rank")),
                        "rrf_score": supplied.get("rrf_score"),
                        "pre_rerank_rank": supplied.get("pre_rerank_rank"),
                        "rerank_rank": supplied.get("rank") if "reranker_score" in supplied else None,
                        "reranker_score": supplied.get("reranker_score"),
                    },
                }
            )
        return output
