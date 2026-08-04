"""Hybrid-versus-reranked retrieval and optional verifier ablation."""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from medclaim.reranking.cross_encoder import RerankerError
from medclaim.reranking.models import (
    EvidenceReranker,
    EvidenceVerifier,
    RerankingConfiguration,
)
from medclaim.retrieval.bm25 import BM25Error, BM25Retriever
from medclaim.retrieval.configuration import RetrievalSettings
from medclaim.retrieval.dense import DenseError, DenseRetriever
from medclaim.retrieval.hybrid import HybridError, HybridRetriever

from .bm25_evaluation import (
    EvaluationError,
    _gold_sets_for_claim,
    _index_unique_records,
    _load_jsonl,
    _write_json,
    _write_jsonl,
)
from .classification_metrics import classification_metrics
from .retrieval_metrics import (
    any_gold_passage_recall_at_k,
    complete_evidence_recall_at_k,
    reciprocal_rank,
)

METHODS = ("hybrid", "hybrid_reranked")


def _retrieval_measurements(
    retrieved_ids: list[str],
    candidate_ids: list[str],
    gold_sets: list[set[str]],
    final_k: int,
    candidate_count: int,
) -> dict[str, Any]:
    gold_ids = set().union(*gold_sets)
    first_gold_rank, rr = reciprocal_rank(retrieved_ids, gold_ids)
    returned_count = len(retrieved_ids)
    return {
        "complete_evidence_recall_at_final_k": complete_evidence_recall_at_k(
            retrieved_ids, gold_sets, final_k
        ),
        "any_gold_passage_recall_at_final_k": any_gold_passage_recall_at_k(
            retrieved_ids, gold_ids, final_k
        ),
        "evidence_precision_at_final_k": (
            sum(passage_id in gold_ids for passage_id in retrieved_ids)
            / returned_count
            if returned_count
            else 0.0
        ),
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": rr,
        "candidate_pool_complete_evidence_recall": complete_evidence_recall_at_k(
            candidate_ids, gold_sets, candidate_count
        ),
    }


def _aggregate_retrieval(
    predictions: list[dict[str, Any]],
    candidate_count: int,
    final_k: int,
) -> dict[str, Any]:
    count = len(predictions)
    measurements = [prediction["retrieval_metrics"] for prediction in predictions]
    latencies = [prediction["latency_ms"]["total"] for prediction in predictions]
    return {
        "evaluated_claims": count,
        "complete_evidence_recall_at_k": {
            str(final_k): sum(
                item["complete_evidence_recall_at_final_k"]
                for item in measurements
            )
            / count
        },
        "any_gold_passage_recall_at_k": {
            str(final_k): sum(
                item["any_gold_passage_recall_at_final_k"]
                for item in measurements
            )
            / count
        },
        "evidence_precision_at_k": {
            str(final_k): mean(
                item["evidence_precision_at_final_k"] for item in measurements
            )
        },
        "mrr": mean(item["reciprocal_rank"] for item in measurements),
        "candidate_pool_complete_evidence_recall_at_k": {
            str(candidate_count): sum(
                item["candidate_pool_complete_evidence_recall"]
                for item in measurements
            )
            / count
        },
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
    }


def _verify(
    verifier: EvidenceVerifier,
    claim_text: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = [
        {"passage_id": result["passage_id"], "text": result["text"]}
        for result in results
    ]
    started = time.perf_counter()
    output = verifier.verify(claim_text, evidence)
    measured_latency = (time.perf_counter() - started) * 1000
    if not isinstance(output, dict) or not isinstance(output.get("verdict"), str):
        raise ValueError("Verifier output must contain a string verdict.")
    latency = output.get("latency_ms", measured_latency)
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or latency < 0
    ):
        raise ValueError("Verifier latency must be a finite non-negative number.")
    return {"verdict": output["verdict"], "latency_ms": float(latency)}


def _ranking_change(
    claim_id: str,
    hybrid_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    gold_ids: set[str],
) -> dict[str, Any]:
    hybrid_ids = [item["passage_id"] for item in hybrid_results]
    reranked_ids = [item["passage_id"] for item in reranked_results]
    movements = [
        {
            "passage_id": item["passage_id"],
            "before": item["pre_rerank_rank"],
            "after": item["rank"],
        }
        for item in reranked_results
    ]
    largest = (
        sorted(
            movements,
            key=lambda item: (
                -abs(item["before"] - item["after"]),
                item["passage_id"],
            ),
        )[0]
        if movements
        else None
    )
    return {
        "claim_id": claim_id,
        "hybrid_top_5": hybrid_ids,
        "reranked_top_5": reranked_ids,
        "gold_passages_promoted_into_top_5": sorted(
            (set(reranked_ids) - set(hybrid_ids)) & gold_ids
        ),
        "gold_passages_removed_from_top_5": sorted(
            (set(hybrid_ids) - set(reranked_ids)) & gold_ids
        ),
        "largest_rank_change": largest,
    }


def _write_comparison_csv(
    retrieval_metrics: dict[str, Any],
    classification: dict[str, Any],
    final_k: int,
    candidate_count: int,
    path: Path,
) -> None:
    fields = [
        "method",
        f"complete_evidence_recall_at_{final_k}",
        f"any_gold_passage_recall_at_{final_k}",
        f"evidence_precision_at_{final_k}",
        f"candidate_pool_recall_at_{candidate_count}",
        "mrr",
        "mean_latency_ms",
        "median_latency_ms",
        "accuracy",
        "macro_f1",
    ]
    try:
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            for method in METHODS:
                retrieval = retrieval_metrics["methods"][method]
                method_classification = classification.get("methods", {}).get(
                    method, {}
                )
                writer.writerow(
                    {
                        "method": method,
                        fields[1]: retrieval["complete_evidence_recall_at_k"][
                            str(final_k)
                        ],
                        fields[2]: retrieval["any_gold_passage_recall_at_k"][
                            str(final_k)
                        ],
                        fields[3]: retrieval["evidence_precision_at_k"][str(final_k)],
                        fields[4]: retrieval[
                            "candidate_pool_complete_evidence_recall_at_k"
                        ][str(candidate_count)],
                        "mrr": retrieval["mrr"],
                        "mean_latency_ms": retrieval["mean_latency_ms"],
                        "median_latency_ms": retrieval["median_latency_ms"],
                        "accuracy": method_classification.get("accuracy"),
                        "macro_f1": method_classification.get("macro_f1"),
                    }
                )
    except (OSError, ValueError) as exc:
        raise EvaluationError(
            f"RERANKER_OUTPUT_WRITE_FAILED: Could not write {path}: {exc}."
        ) from exc


def compare_reranking(
    claims_path: Path,
    gold_evidence_path: Path,
    corpus_dir: Path,
    bm25_index_dir: Path,
    dense_index_dir: Path,
    split: str,
    output_dir: Path,
    reranker: EvidenceReranker,
    reranking_configuration: RerankingConfiguration | None = None,
    retrieval_settings: RetrievalSettings | None = None,
    verifier: EvidenceVerifier | None = None,
    max_claims: int | None = None,
    device: str = "cpu",
    *,
    dense_embedder: Any | None = None,
) -> dict[str, Any]:
    """Compare hybrid Top-K with the same candidate pool after reranking."""
    if output_dir.exists():
        raise EvaluationError(
            "RERANKING_EXPERIMENT_EXISTS: "
            f"Experiment output {output_dir.name} already exists."
        )
    if not isinstance(split, str) or not split.strip():
        raise EvaluationError("EVALUATION_INVALID_SPLIT: Split must be non-empty.")
    if max_claims is not None and (
        not isinstance(max_claims, int)
        or isinstance(max_claims, bool)
        or max_claims < 1
    ):
        raise EvaluationError(
            "RERANKER_INVALID_CONFIGURATION: max_claims must be positive."
        )
    configuration = reranking_configuration or RerankingConfiguration()
    if not configuration.enabled:
        raise EvaluationError(
            "RERANKER_INVALID_CONFIGURATION: Ablation requires enabled reranking."
        )
    if (
        reranker.model_id != configuration.model_id
        or reranker.model_revision != configuration.model_revision
        or reranker.batch_size != configuration.batch_size
        or reranker.maximum_input_length != configuration.maximum_input_length
    ):
        raise EvaluationError(
            "RERANKER_INVALID_CONFIGURATION: Reranker and configuration disagree."
        )
    hybrid_settings = retrieval_settings or RetrievalSettings()
    if configuration.candidate_count > (
        hybrid_settings.sparse_top_k + hybrid_settings.dense_top_k
    ):
        raise EvaluationError(
            "RERANKER_INVALID_CONFIGURATION: Candidate count exceeds hybrid limits."
        )
    try:
        sparse = BM25Retriever(bm25_index_dir, corpus_dir)
        dense = DenseRetriever(
            dense_index_dir, corpus_dir, device=device, embedder=dense_embedder
        )
        hybrid = HybridRetriever(
            sparse,
            dense,
            sparse_top_k=hybrid_settings.sparse_top_k,
            dense_top_k=hybrid_settings.dense_top_k,
            fusion_top_k=max(
                hybrid_settings.fusion_top_k, configuration.candidate_count
            ),
            rrf_k=hybrid_settings.rrf_k,
        )
    except (BM25Error, DenseError, HybridError) as exc:
        raise EvaluationError(str(exc)) from exc

    claims = _load_jsonl(claims_path, "claims")
    gold_records = _load_jsonl(gold_evidence_path, "gold evidence")
    claims_by_id = _index_unique_records(claims, "claims")
    gold_by_id = _index_unique_records(gold_records, "gold evidence")
    unknown = sorted(set(gold_by_id) - set(claims_by_id))
    if unknown:
        raise EvaluationError(
            f"EVALUATION_UNKNOWN_CLAIM: Gold evidence references {unknown[0]}."
        )
    normalized_split = split.strip()
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
    corpus_ids = set(sparse.passages_by_id)
    for claim in split_claims:
        claim_text = claim.get("claim_text")
        if not isinstance(claim_text, str) or not claim_text.strip():
            exclusions["missing_claim_text"] += 1
            continue
        if claim.get("unified_label") is None:
            exclusions["unlabeled"] += 1
            continue
        sets, reason = _gold_sets_for_claim(gold_by_id.get(claim["claim_id"]), corpus_ids)
        if reason is not None:
            exclusions[reason] += 1
            continue
        eligible.append((claim, sets))
    if max_claims is not None:
        eligible = eligible[:max_claims]
    if not eligible:
        raise EvaluationError(
            "EVALUATION_NO_ELIGIBLE_CLAIMS: No claims have usable gold evidence."
        )

    predictions = {method: [] for method in METHODS}
    changes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    classification_records = {method: [] for method in METHODS}
    for claim, ordered_sets in eligible:
        hybrid_result = hybrid.search(
            claim["claim_text"], top_k=configuration.candidate_count
        )
        candidates = hybrid_result["results"]
        hybrid_final = [
            dict(item) for item in candidates[: configuration.final_evidence_k]
        ]
        rerank_started = time.perf_counter()
        try:
            reranked_final = reranker.rerank(
                claim["claim_text"],
                candidates,
                min(configuration.final_evidence_k, len(candidates)),
            ) if candidates else []
        except RerankerError as exc:
            raise EvaluationError(str(exc)) from exc
        rerank_latency = (time.perf_counter() - rerank_started) * 1000
        hybrid_latency = hybrid_result["latency_ms"]["total"]
        method_results = {
            "hybrid": (hybrid_final, hybrid_latency, 0.0),
            "hybrid_reranked": (
                reranked_final,
                hybrid_latency + rerank_latency,
                rerank_latency,
            ),
        }
        gold_sets = [set(values) for values in ordered_sets]
        gold_ids = set().union(*gold_sets)
        candidate_ids = [item["passage_id"] for item in candidates]
        for method, (results, total_latency, stage_latency) in method_results.items():
            metrics = _retrieval_measurements(
                [item["passage_id"] for item in results],
                candidate_ids,
                gold_sets,
                configuration.final_evidence_k,
                configuration.candidate_count,
            )
            prediction = {
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"].strip(),
                "split": normalized_split,
                "method": method,
                "gold_evidence_sets": ordered_sets,
                "candidate_pool_passage_ids": candidate_ids,
                "retrieved": results,
                "retrieval_metrics": metrics,
                "latency_ms": {
                    "hybrid_retrieval": hybrid_latency,
                    "reranking": stage_latency,
                    "total": total_latency,
                },
                "verification": None,
            }
            if verifier is not None:
                try:
                    verification = _verify(
                        verifier, claim["claim_text"].strip(), results
                    )
                    prediction["verification"] = verification
                    classification_records[method].append(
                        {
                            "gold": claim["unified_label"],
                            "predicted": verification["verdict"],
                            "verification_latency_ms": verification["latency_ms"],
                            "total_latency_ms": total_latency
                            + verification["latency_ms"],
                        }
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "claim_id": claim["claim_id"],
                            "method": method,
                            "error_type": "VERIFICATION_FAILED",
                            "reason": str(exc),
                        }
                    )
            predictions[method].append(prediction)
        changes.append(
            _ranking_change(
                claim["claim_id"], hybrid_final, reranked_final, gold_ids
            )
        )

    retrieval_metrics = {
        "dataset": "scifact",
        "split": normalized_split,
        "candidate_count": configuration.candidate_count,
        "final_evidence_k": configuration.final_evidence_k,
        "methods": {
            method: _aggregate_retrieval(
                predictions[method],
                configuration.candidate_count,
                configuration.final_evidence_k,
            )
            for method in METHODS
        },
    }
    if verifier is None:
        classification = {
            "status": "not_available",
            "reason": "No verification pipeline is configured in this repository.",
            "methods": {},
        }
    else:
        classification = {"status": "available", "methods": {}}
        for method in METHODS:
            records = classification_records[method]
            metrics = classification_metrics(
                [item["gold"] for item in records],
                [item["predicted"] for item in records],
            )
            if records:
                metrics["mean_verification_latency_ms"] = mean(
                    item["verification_latency_ms"] for item in records
                )
                metrics["mean_total_latency_ms"] = mean(
                    item["total_latency_ms"] for item in records
                )
            classification["methods"][method] = metrics

    manifest = {
        "artifact_type": "reranking_ablation",
        "experiment_id": output_dir.name,
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "dataset": "scifact",
        "split": normalized_split,
        "corpus_version": sparse.corpus_manifest["corpus_version"],
        "corpus_content_hash": sparse.corpus_manifest["content_hash"],
        "bm25_index_version": sparse.index_manifest["index_version"],
        "dense_index_version": dense.index_manifest["index_version"],
        "rrf_configuration": {
            "sparse_top_k": hybrid_settings.sparse_top_k,
            "dense_top_k": hybrid_settings.dense_top_k,
            "rrf_k": hybrid_settings.rrf_k,
        },
        "reranker": {
            "model_id": configuration.model_id,
            "model_revision": configuration.model_revision,
            "device": reranker.device,
            "batch_size": configuration.batch_size,
            "maximum_input_length": configuration.maximum_input_length,
            "candidate_count": configuration.candidate_count,
            "final_evidence_k": configuration.final_evidence_k,
        },
        "verifier_model": getattr(verifier, "model_id", None),
        "verifier_prompt_version": getattr(verifier, "prompt_version", None),
        "random_seed": None,
        "max_claims": max_claims,
        "sample_experiment": max_claims is not None,
        "evaluated_claims": len(eligible),
        "excluded_claims": sum(exclusions.values()),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "outputs": {
            "hybrid_predictions": "hybrid_predictions.jsonl",
            "reranked_predictions": "reranked_predictions.jsonl",
            "retrieval_metrics": "retrieval_metrics.json",
            "classification_metrics": "classification_metrics.json",
            "comparison": "comparison.csv",
            "reranking_changes": "reranking_changes.jsonl",
            "errors": "errors.jsonl",
        },
    }

    temporary_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
        )
        _write_jsonl(predictions["hybrid"], temporary_dir / "hybrid_predictions.jsonl")
        _write_jsonl(
            predictions["hybrid_reranked"],
            temporary_dir / "reranked_predictions.jsonl",
        )
        _write_json(retrieval_metrics, temporary_dir / "retrieval_metrics.json")
        _write_json(classification, temporary_dir / "classification_metrics.json")
        _write_comparison_csv(
            retrieval_metrics,
            classification,
            configuration.final_evidence_k,
            configuration.candidate_count,
            temporary_dir / "comparison.csv",
        )
        _write_jsonl(changes, temporary_dir / "reranking_changes.jsonl")
        _write_jsonl(errors, temporary_dir / "errors.jsonl")
        _write_json(manifest, temporary_dir / "manifest.json")
        os.rename(temporary_dir, output_dir)
    except FileExistsError as exc:
        raise EvaluationError(
            "RERANKING_EXPERIMENT_EXISTS: "
            f"Experiment output {output_dir.name} already exists."
        ) from exc
    except EvaluationError as exc:
        if "OUTPUT_WRITE_FAILED" in str(exc):
            raise EvaluationError(
                f"RERANKER_OUTPUT_WRITE_FAILED: {exc}."
            ) from exc
        raise
    except OSError as exc:
        raise EvaluationError(
            f"RERANKER_OUTPUT_WRITE_FAILED: Could not create output: {exc}."
        ) from exc
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return {
        "retrieval_metrics": retrieval_metrics,
        "classification_metrics": classification,
    }
