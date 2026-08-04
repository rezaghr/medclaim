"""Compare BM25, dense, and hybrid retrieval on identical SciFact claims."""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from medclaim.retrieval.bm25 import BM25Error, BM25Retriever
from medclaim.retrieval.configuration import RetrievalSettings
from medclaim.retrieval.dense import DenseError, DenseRetriever
from medclaim.retrieval.hybrid import HybridError, HybridRetriever

from .bm25_evaluation import (
    EvaluationError,
    _gold_sets_for_claim,
    _index_unique_records,
    _load_jsonl,
    _validate_ks,
    _write_json,
    _write_jsonl,
)
from .retrieval_metrics import (
    any_gold_passage_recall_at_k,
    complete_evidence_recall_at_k,
    reciprocal_rank,
)

METHODS = ("bm25", "dense", "hybrid")


def _prediction_from_result(
    method: str,
    claim: dict[str, Any],
    ordered_gold_sets: list[list[str]],
    result: dict[str, Any],
    ks: list[int],
    split: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    retrieved_ids = [item["passage_id"] for item in result["results"]]
    gold_sets = [set(evidence_set) for evidence_set in ordered_gold_sets]
    gold_ids = set().union(*gold_sets)
    complete_recall = {
        str(k): complete_evidence_recall_at_k(retrieved_ids, gold_sets, k)
        for k in ks
    }
    any_recall = {
        str(k): any_gold_passage_recall_at_k(retrieved_ids, gold_ids, k)
        for k in ks
    }
    first_gold_rank, rr = reciprocal_rank(retrieved_ids, gold_ids)
    score_fields = {
        "bm25": ("rank", "passage_id", "document_id", "bm25_score"),
        "dense": ("rank", "passage_id", "document_id", "dense_score"),
        "hybrid": (
            "rank",
            "passage_id",
            "document_id",
            "bm25_rank",
            "bm25_score",
            "dense_rank",
            "dense_score",
            "rrf_score",
        ),
    }
    latency = (
        result["latency_ms"]["total"]
        if method == "hybrid"
        else result["latency_ms"]
    )
    prediction = {
        "claim_id": claim["claim_id"],
        "claim_text": claim["claim_text"].strip(),
        "split": split,
        "method": method,
        "gold_evidence_sets": ordered_gold_sets,
        "retrieved": [
            {field: item[field] for field in score_fields[method]}
            for item in result["results"]
        ],
        "complete_evidence_recall": complete_recall,
        "any_gold_passage_recall": any_recall,
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": rr,
        "latency_ms": latency,
    }
    max_k = max(ks)
    error = None
    if not complete_recall[str(max_k)]:
        error = {
            "claim_id": claim["claim_id"],
            "method": method,
            "error_type": (
                "PARTIAL_EVIDENCE_SET_RETRIEVED"
                if any_recall[str(max_k)]
                else "NO_GOLD_PASSAGE_IN_TOP_K"
            ),
            "gold_evidence_sets": ordered_gold_sets,
            "retrieved_passage_ids": retrieved_ids,
        }
    return prediction, error


def _method_metrics(
    predictions: list[dict[str, Any]], ks: list[int]
) -> dict[str, Any]:
    count = len(predictions)
    latencies = [prediction["latency_ms"] for prediction in predictions]
    metrics = {
        "complete_evidence_recall_at_k": {
            str(k): sum(
                prediction["complete_evidence_recall"][str(k)]
                for prediction in predictions
            )
            / count
            for k in ks
        },
        "any_gold_passage_recall_at_k": {
            str(k): sum(
                prediction["any_gold_passage_recall"][str(k)]
                for prediction in predictions
            )
            / count
            for k in ks
        },
        "mrr": mean(prediction["reciprocal_rank"] for prediction in predictions),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
    }
    if any(
        not math.isfinite(value)
        for value in (
            metrics["mrr"],
            metrics["mean_latency_ms"],
            metrics["median_latency_ms"],
        )
    ):
        raise EvaluationError(
            "EVALUATION_INVALID_METRIC: Comparison produced a non-finite metric."
        )
    return metrics


def _write_comparison_csv(
    metrics: dict[str, Any], ks: list[int], path: Path
) -> None:
    fieldnames = ["method"]
    fieldnames.extend(f"recall_at_{k}" for k in ks)
    fieldnames.extend(f"any_passage_recall_at_{k}" for k in ks)
    fieldnames.extend(("mrr", "mean_latency_ms", "median_latency_ms"))
    try:
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for method in METHODS:
                method_metrics = metrics["methods"][method]
                row: dict[str, Any] = {"method": method}
                row.update(
                    {
                        f"recall_at_{k}": method_metrics[
                            "complete_evidence_recall_at_k"
                        ][str(k)]
                        for k in ks
                    }
                )
                row.update(
                    {
                        f"any_passage_recall_at_{k}": method_metrics[
                            "any_gold_passage_recall_at_k"
                        ][str(k)]
                        for k in ks
                    }
                )
                row.update(
                    {
                        "mrr": method_metrics["mrr"],
                        "mean_latency_ms": method_metrics["mean_latency_ms"],
                        "median_latency_ms": method_metrics["median_latency_ms"],
                    }
                )
                writer.writerow(row)
    except (OSError, ValueError) as exc:
        raise EvaluationError(
            f"EVALUATION_OUTPUT_WRITE_FAILED: Could not write {path}: {exc}."
        ) from exc


def compare_retrieval(
    claims_path: Path,
    gold_evidence_path: Path,
    corpus_dir: Path,
    bm25_index_dir: Path,
    dense_index_dir: Path,
    split: str,
    output_dir: Path,
    ks: list[int],
    settings: RetrievalSettings | None = None,
    device: str = "cpu",
    *,
    dense_embedder: Any | None = None,
) -> dict[str, Any]:
    """Evaluate all three retrieval modes on one shared eligible claim set."""
    if output_dir.exists():
        raise EvaluationError(
            "EVALUATION_OUTPUT_EXISTS: "
            f"Experiment output {output_dir.name} already exists."
        )
    if not isinstance(split, str) or not split.strip():
        raise EvaluationError("EVALUATION_INVALID_SPLIT: Split must be non-empty.")
    normalized_split = split.strip()
    sorted_ks = _validate_ks(ks)
    selected_settings = settings or RetrievalSettings()
    if selected_settings.mode != "hybrid":
        raise EvaluationError(
            "EVALUATION_INVALID_CONFIGURATION: Comparison requires hybrid settings."
        )
    if max(sorted_ks) > selected_settings.sparse_top_k + selected_settings.dense_top_k:
        raise EvaluationError(
            "EVALUATION_INVALID_K: Maximum K exceeds available hybrid candidates."
        )

    try:
        sparse_retriever = BM25Retriever(bm25_index_dir, corpus_dir)
        dense_retriever = DenseRetriever(
            dense_index_dir,
            corpus_dir,
            device=device,
            embedder=dense_embedder,
        )
        hybrid_retriever = HybridRetriever(
            sparse_retriever,
            dense_retriever,
            sparse_top_k=selected_settings.sparse_top_k,
            dense_top_k=selected_settings.dense_top_k,
            fusion_top_k=selected_settings.fusion_top_k,
            rrf_k=selected_settings.rrf_k,
        )
    except (BM25Error, DenseError, HybridError) as exc:
        raise EvaluationError(str(exc)) from exc

    claims = _load_jsonl(claims_path, "claims")
    gold_records = _load_jsonl(gold_evidence_path, "gold evidence")
    claims_by_id = _index_unique_records(claims, "claims")
    gold_by_id = _index_unique_records(gold_records, "gold evidence")
    unknown_gold_claims = sorted(set(gold_by_id) - set(claims_by_id))
    if unknown_gold_claims:
        raise EvaluationError(
            "EVALUATION_UNKNOWN_CLAIM: Gold evidence references unknown claim "
            f"{unknown_gold_claims[0]}."
        )
    split_claims = sorted(
        (
            claim
            for claim in claims
            if claim.get("original_split") == normalized_split
        ),
        key=lambda claim: claim["claim_id"],
    )
    exclusions: Counter[str] = Counter()
    eligible: list[tuple[dict[str, Any], list[list[str]]]] = []
    corpus_passage_ids = set(sparse_retriever.passages_by_id)
    for claim in split_claims:
        claim_text = claim.get("claim_text")
        if not isinstance(claim_text, str) or not claim_text.strip():
            exclusions["missing_claim_text"] += 1
            continue
        if claim.get("unified_label") is None:
            exclusions["unlabeled"] += 1
            continue
        gold_record = gold_by_id.get(claim["claim_id"])
        if gold_record is not None and gold_record.get("original_split") not in (
            None,
            normalized_split,
        ):
            raise EvaluationError(
                "EVALUATION_SPLIT_MISMATCH: Claim and gold splits disagree for "
                f"{claim['claim_id']}."
            )
        gold_sets, exclusion_reason = _gold_sets_for_claim(
            gold_record, corpus_passage_ids
        )
        if exclusion_reason is not None:
            exclusions[exclusion_reason] += 1
            continue
        eligible.append((claim, gold_sets))
    if not eligible:
        raise EvaluationError(
            "EVALUATION_NO_ELIGIBLE_CLAIMS: No claims have usable gold evidence."
        )

    predictions: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHODS
    }
    retrieval_errors: list[dict[str, Any]] = []
    max_k = max(sorted_ks)
    for claim, gold_sets in eligible:
        try:
            results = {
                "bm25": sparse_retriever.search(claim["claim_text"], max_k),
                "dense": dense_retriever.search(claim["claim_text"], max_k),
                "hybrid": hybrid_retriever.search(claim["claim_text"], max_k),
            }
        except (BM25Error, DenseError, HybridError) as exc:
            raise EvaluationError(str(exc)) from exc
        for method in METHODS:
            prediction, error = _prediction_from_result(
                method,
                claim,
                gold_sets,
                results[method],
                sorted_ks,
                normalized_split,
            )
            predictions[method].append(prediction)
            if error is not None:
                retrieval_errors.append(error)

    metrics = {
        "dataset": "scifact",
        "split": normalized_split,
        "ks": sorted_ks,
        "total_claims_in_split": len(split_claims),
        "evaluated_claims": len(eligible),
        "excluded_claims": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "methods": {
            method: _method_metrics(predictions[method], sorted_ks)
            for method in METHODS
        },
    }
    dense_embedding = dense_retriever.index_manifest["embedding"]
    manifest = {
        "artifact_type": "retrieval_comparison",
        "experiment_id": output_dir.name,
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "dataset": "scifact",
        "split": normalized_split,
        "ks": sorted_ks,
        "corpus_version": sparse_retriever.corpus_manifest["corpus_version"],
        "corpus_content_hash": sparse_retriever.corpus_manifest["content_hash"],
        "bm25_index_version": sparse_retriever.index_manifest["index_version"],
        "dense_index_version": dense_retriever.index_manifest["index_version"],
        "dense_embedding": {
            "model_id": dense_embedding["model_id"],
            "model_revision": dense_embedding["model_revision"],
            "dimension": dense_embedding["dimension"],
            "normalize_embeddings": dense_embedding["normalize_embeddings"],
        },
        "hybrid_configuration": {
            "sparse_top_k": selected_settings.sparse_top_k,
            "dense_top_k": selected_settings.dense_top_k,
            "fusion_top_k": selected_settings.fusion_top_k,
            "rrf_k": selected_settings.rrf_k,
            "final_evidence_k": selected_settings.final_evidence_k,
        },
        "evaluated_claims": len(eligible),
        "excluded_claims": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "outputs": {
            "bm25_predictions": "bm25_predictions.jsonl",
            "dense_predictions": "dense_predictions.jsonl",
            "hybrid_predictions": "hybrid_predictions.jsonl",
            "metrics": "metrics.json",
            "comparison": "comparison.csv",
            "retrieval_errors": "retrieval_errors.jsonl",
        },
    }

    temporary_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
        )
        for method in METHODS:
            _write_jsonl(
                predictions[method], temporary_dir / f"{method}_predictions.jsonl"
            )
        _write_json(metrics, temporary_dir / "metrics.json")
        _write_comparison_csv(metrics, sorted_ks, temporary_dir / "comparison.csv")
        _write_jsonl(retrieval_errors, temporary_dir / "retrieval_errors.jsonl")
        _write_json(manifest, temporary_dir / "manifest.json")
        os.rename(temporary_dir, output_dir)
    except FileExistsError as exc:
        raise EvaluationError(
            "EVALUATION_OUTPUT_EXISTS: "
            f"Experiment output {output_dir.name} already exists."
        ) from exc
    except EvaluationError:
        raise
    except OSError as exc:
        raise EvaluationError(
            f"EVALUATION_OUTPUT_WRITE_FAILED: Could not create output: {exc}."
        ) from exc
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return metrics
