"""Evaluate a project BM25 index against SciFact gold evidence mappings."""

from __future__ import annotations

import json
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

from .retrieval_metrics import (
    any_gold_passage_recall_at_k,
    complete_evidence_recall_at_k,
    reciprocal_rank,
)


class EvaluationError(Exception):
    """Raised when retrieval evaluation cannot be completed reliably."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric value {value}")


def _load_jsonl(path: Path, kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    reason = getattr(exc, "msg", str(exc))
                    raise EvaluationError(
                        "EVALUATION_INVALID_JSON: "
                        f"Could not parse {kind} file {path} at line "
                        f"{line_number}: {reason}."
                    ) from exc
                if not isinstance(record, dict):
                    raise EvaluationError(
                        "EVALUATION_INVALID_JSON: "
                        f"Expected an object in {kind} file {path} at line "
                        f"{line_number}."
                    )
                records.append(record)
    except FileNotFoundError as exc:
        raise EvaluationError(
            f"EVALUATION_INPUT_NOT_FOUND: Required {kind} file does not exist: {path}."
        ) from exc
    except OSError as exc:
        raise EvaluationError(
            f"EVALUATION_INPUT_READ_FAILED: Could not read {path}: {exc}."
        ) from exc
    return records


def _validate_ks(ks: list[int]) -> list[int]:
    if not isinstance(ks, list) or not ks:
        raise EvaluationError("EVALUATION_INVALID_K: Supply at least one K value.")
    if any(
        not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100
        for k in ks
    ):
        raise EvaluationError(
            "EVALUATION_INVALID_K: Every K must be an integer between 1 and 100."
        )
    if len(set(ks)) != len(ks):
        raise EvaluationError("EVALUATION_INVALID_K: K values must be unique.")
    return sorted(ks)


def _index_unique_records(
    records: list[dict[str, Any]], kind: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        claim_id = record.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id:
            raise EvaluationError(
                f"EVALUATION_INVALID_{kind.upper()}: A record has no valid claim_id."
            )
        if claim_id in indexed:
            raise EvaluationError(
                f"EVALUATION_DUPLICATE_CLAIM_ID: Claim ID {claim_id} appears "
                f"more than once in the {kind} file."
            )
        indexed[claim_id] = record
    return indexed


def _gold_sets_for_claim(
    gold_record: dict[str, Any] | None,
    corpus_passage_ids: set[str],
) -> tuple[list[list[str]], str | None]:
    if gold_record is None:
        return [], "no_gold_evidence"
    evidence_sets = gold_record.get("evidence_sets")
    if not isinstance(evidence_sets, list):
        raise EvaluationError(
            "EVALUATION_INVALID_GOLD_EVIDENCE: evidence_sets must be a list."
        )

    usable_sets: list[list[str]] = []
    saw_nonempty_set = False
    saw_unresolved_set = False
    for evidence_set in evidence_sets:
        if not isinstance(evidence_set, dict):
            raise EvaluationError(
                "EVALUATION_INVALID_GOLD_EVIDENCE: Evidence sets must be objects."
            )
        passage_ids = evidence_set.get("passage_ids")
        if not isinstance(passage_ids, list) or any(
            not isinstance(passage_id, str) or not passage_id
            for passage_id in passage_ids
        ):
            raise EvaluationError(
                "EVALUATION_INVALID_GOLD_EVIDENCE: passage_ids must be a list "
                "of non-empty strings."
            )
        ordered_unique_ids = list(dict.fromkeys(passage_ids))
        if not ordered_unique_ids:
            continue
        saw_nonempty_set = True
        if all(passage_id in corpus_passage_ids for passage_id in ordered_unique_ids):
            usable_sets.append(ordered_unique_ids)
        else:
            saw_unresolved_set = True

    if usable_sets:
        return usable_sets, None
    if saw_nonempty_set and saw_unresolved_set:
        return [], "unresolved_gold_evidence"
    return [], "no_gold_evidence"


def _write_json(value: Any, path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, allow_nan=False, indent=2)
            output_file.write("\n")
    except (OSError, ValueError) as exc:
        raise EvaluationError(
            f"EVALUATION_OUTPUT_WRITE_FAILED: Could not write {path}: {exc}."
        ) from exc


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    try:
        with path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    except (OSError, ValueError) as exc:
        raise EvaluationError(
            f"EVALUATION_OUTPUT_WRITE_FAILED: Could not write {path}: {exc}."
        ) from exc


def evaluate_bm25(
    claims_path: Path,
    gold_evidence_path: Path,
    corpus_dir: Path,
    index_dir: Path,
    split: str,
    output_dir: Path,
    ks: list[int],
) -> dict[str, Any]:
    """Run one immutable, offline BM25 retrieval evaluation."""
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
        retriever = BM25Retriever(index_dir, corpus_dir)
    except BM25Error as exc:
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
                    "bm25_score": item["bm25_score"],
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
        "retrieval_mode": "bm25",
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

    index_configuration = retriever.index_manifest["configuration"]
    manifest = {
        "artifact_type": "retrieval_evaluation",
        "experiment_id": output_dir.name,
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "dataset": "scifact",
        "split": normalized_split,
        "retrieval_mode": "bm25",
        "ks": sorted_ks,
        "corpus_version": retriever.corpus_manifest["corpus_version"],
        "corpus_content_hash": retriever.corpus_manifest["content_hash"],
        "index_version": retriever.index_manifest["index_version"],
        "index_configuration": {
            "tokenizer": index_configuration["tokenizer"],
            "k1": index_configuration["k1"],
            "b": index_configuration["b"],
            "epsilon": index_configuration["epsilon"],
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
