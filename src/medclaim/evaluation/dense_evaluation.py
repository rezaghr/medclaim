"""Evaluate dense retrieval against SciFact gold evidence mappings."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from medclaim.retrieval.dense import DenseError, DenseRetriever

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


def evaluate_dense(
    claims_path: Path,
    gold_evidence_path: Path,
    corpus_dir: Path,
    index_dir: Path,
    split: str,
    output_dir: Path,
    ks: list[int],
    device: str = "cpu",
    *,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Run one immutable dense retrieval evaluation."""
    if output_dir.exists():
        raise EvaluationError(
            "EVALUATION_OUTPUT_EXISTS: "
            f"Experiment output {output_dir.name} already exists."
        )
    if not isinstance(split, str) or not split.strip():
        raise EvaluationError("EVALUATION_INVALID_SPLIT: Split must be non-empty.")
    normalized_split = split.strip()
    sorted_ks = _validate_ks(ks)
    try:
        retriever = DenseRetriever(
            index_dir, corpus_dir, device=device, embedder=embedder
        )
    except DenseError as exc:
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
    corpus_passage_ids = set(retriever.passages_by_id)
    exclusions: Counter[str] = Counter()
    eligible: list[tuple[dict[str, Any], list[list[str]]]] = []
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
                "EVALUATION_SPLIT_MISMATCH: Claim and gold evidence splits do not "
                f"match for {claim['claim_id']}."
            )
        gold_sets, exclusion_reason = _gold_sets_for_claim(
            gold_record, corpus_passage_ids
        )
        if exclusion_reason is not None:
            exclusions[exclusion_reason] += 1
            continue
        eligible.append((claim, gold_sets))

    predictions: list[dict[str, Any]] = []
    retrieval_errors: list[dict[str, Any]] = []
    max_k = max(sorted_ks)
    for claim, ordered_gold_sets in eligible:
        result = retriever.search(claim["claim_text"], max_k)
        retrieved = result["results"]
        retrieved_ids = [item["passage_id"] for item in retrieved]
        gold_sets = [set(evidence_set) for evidence_set in ordered_gold_sets]
        gold_ids = set().union(*gold_sets)
        complete_recall = {
            str(k): complete_evidence_recall_at_k(retrieved_ids, gold_sets, k)
            for k in sorted_ks
        }
        any_recall = {
            str(k): any_gold_passage_recall_at_k(retrieved_ids, gold_ids, k)
            for k in sorted_ks
        }
        first_gold_rank, rr = reciprocal_rank(retrieved_ids, gold_ids)
        prediction = {
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"].strip(),
            "split": normalized_split,
            "gold_evidence_sets": ordered_gold_sets,
            "retrieved": [
                {
                    "rank": item["rank"],
                    "passage_id": item["passage_id"],
                    "document_id": item["document_id"],
                    "dense_score": item["dense_score"],
                }
                for item in retrieved
            ],
            "complete_evidence_recall": complete_recall,
            "any_gold_passage_recall": any_recall,
            "first_gold_rank": first_gold_rank,
            "reciprocal_rank": rr,
            "latency_ms": result["latency_ms"],
        }
        predictions.append(prediction)
        if not complete_recall[str(max_k)]:
            retrieval_errors.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["claim_text"].strip(),
                    "gold_evidence_sets": ordered_gold_sets,
                    "top_retrieved_passage_ids": retrieved_ids,
                    "first_gold_rank": first_gold_rank,
                    "error_type": (
                        "PARTIAL_EVIDENCE_SET_RETRIEVED"
                        if any_recall[str(max_k)]
                        else "NO_GOLD_PASSAGE_IN_TOP_K"
                    ),
                }
            )

    evaluated_count = len(predictions)
    if evaluated_count == 0:
        raise EvaluationError(
            "EVALUATION_NO_ELIGIBLE_CLAIMS: No claims in the requested split "
            "have usable gold evidence."
        )
    latencies = [prediction["latency_ms"] for prediction in predictions]
    metrics = {
        "experiment": output_dir.name,
        "dataset": "scifact",
        "split": normalized_split,
        "retrieval_mode": "dense",
        "total_claims_in_split": len(split_claims),
        "evaluated_claims": evaluated_count,
        "excluded_claims": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "complete_evidence_recall_at_k": {
            str(k): sum(
                prediction["complete_evidence_recall"][str(k)]
                for prediction in predictions
            )
            / evaluated_count
            for k in sorted_ks
        },
        "any_gold_passage_recall_at_k": {
            str(k): sum(
                prediction["any_gold_passage_recall"][str(k)]
                for prediction in predictions
            )
            / evaluated_count
            for k in sorted_ks
        },
        "mrr": mean(prediction["reciprocal_rank"] for prediction in predictions),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "corpus_version": retriever.corpus_manifest["corpus_version"],
        "index_version": retriever.index_manifest["index_version"],
        "embedding_model": retriever.index_manifest["embedding"]["model_id"],
    }
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in (
            metrics["mrr"],
            metrics["mean_latency_ms"],
            metrics["median_latency_ms"],
        )
    ):
        raise EvaluationError(
            "EVALUATION_INVALID_METRIC: Evaluation produced a non-finite metric."
        )

    embedding_config = retriever.index_manifest["embedding"]
    manifest = {
        "artifact_type": "retrieval_evaluation",
        "experiment_id": output_dir.name,
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "dataset": "scifact",
        "split": normalized_split,
        "retrieval_mode": "dense",
        "ks": sorted_ks,
        "corpus_version": retriever.corpus_manifest["corpus_version"],
        "corpus_content_hash": retriever.corpus_manifest["content_hash"],
        "index_version": retriever.index_manifest["index_version"],
        "embedding": {
            "model_id": embedding_config["model_id"],
            "model_revision": embedding_config["model_revision"],
            "dimension": embedding_config["dimension"],
            "normalize_embeddings": embedding_config["normalize_embeddings"],
        },
        "total_claims_in_split": len(split_claims),
        "evaluated_claims": evaluated_count,
        "excluded_claims": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "outputs": {
            "predictions": "predictions.jsonl",
            "metrics": "metrics.json",
            "retrieval_errors": "retrieval_errors.jsonl",
        },
    }

    temporary_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
        )
        _write_jsonl(predictions, temporary_dir / "predictions.jsonl")
        _write_json(metrics, temporary_dir / "metrics.json")
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
