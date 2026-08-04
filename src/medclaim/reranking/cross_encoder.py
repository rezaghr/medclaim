"""Batch cross-encoder scoring with deterministic evidence ordering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import DEFAULT_RERANKER_MODEL

REQUIRED_CANDIDATE_FIELDS = (
    "rank",
    "passage_id",
    "document_id",
    "dataset",
    "text",
    "bm25_rank",
    "bm25_score",
    "dense_rank",
    "dense_score",
    "rrf_score",
)


class RerankerError(Exception):
    """Raised for controlled reranker model, input, or score failures."""


def _resolve_device(device: str) -> str:
    if device not in {"cpu", "cuda", "auto"}:
        raise RerankerError(
            "RERANKER_INVALID_CONFIGURATION: device must be cpu, cuda, or auto."
        )
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_top_k(top_k: int, candidate_count: int) -> None:
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
        or top_k > candidate_count
    ):
        raise RerankerError(
            "RERANKER_INVALID_CONFIGURATION: top_k must be between 1 and "
            "the configured candidate count."
        )


def _validate_candidates(candidates: list[dict[str, Any]]) -> None:
    if len(candidates) > 100:
        raise RerankerError(
            "RERANKER_INVALID_CONFIGURATION: No more than 100 candidates are allowed."
        )
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or any(
            field not in candidate for field in REQUIRED_CANDIDATE_FIELDS
        ):
            raise RerankerError(
                "RERANKER_INVALID_CANDIDATE: Candidate is missing required fields."
            )
        passage_id = candidate["passage_id"]
        if not isinstance(passage_id, str) or not passage_id or passage_id in seen_ids:
            raise RerankerError(
                "RERANKER_INVALID_CANDIDATE: Passage IDs must be non-empty and unique."
            )
        seen_ids.add(passage_id)
        if any(
            not isinstance(candidate[field], str)
            for field in ("document_id", "dataset", "text")
        ) or not candidate["text"].strip():
            raise RerankerError(
                f"RERANKER_INVALID_CANDIDATE: Passage {passage_id} has invalid text or provenance."
            )
        if (
            not isinstance(candidate["rank"], int)
            or isinstance(candidate["rank"], bool)
            or candidate["rank"] < 1
        ):
            raise RerankerError(
                f"RERANKER_INVALID_CANDIDATE: Passage {passage_id} has an invalid rank."
            )
        for rank_field in ("bm25_rank", "dense_rank"):
            value = candidate[rank_field]
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise RerankerError(
                    f"RERANKER_INVALID_CANDIDATE: Passage {passage_id} has an "
                    f"invalid {rank_field}."
                )
        for score_field in ("bm25_score", "dense_score"):
            value = candidate[score_field]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RerankerError(
                    f"RERANKER_INVALID_CANDIDATE: Passage {passage_id} has an "
                    f"invalid {score_field}."
                )
        rrf_score = candidate["rrf_score"]
        if (
            isinstance(rrf_score, bool)
            or not isinstance(rrf_score, (int, float))
            or not math.isfinite(float(rrf_score))
        ):
            raise RerankerError(
                f"RERANKER_INVALID_CANDIDATE: Passage {passage_id} has an invalid rrf_score."
            )


class CrossEncoderReranker:
    """Load one SentenceTransformers CrossEncoder and reuse it across claims."""

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        model_revision: str | None = None,
        maximum_input_length: int = 512,
        *,
        model: Any | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise RerankerError(
                "RERANKER_INVALID_CONFIGURATION: model_id must be non-empty."
            )
        if model_revision is not None and (
            not isinstance(model_revision, str) or not model_revision.strip()
        ):
            raise RerankerError(
                "RERANKER_INVALID_CONFIGURATION: model_revision is invalid."
            )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
            or not isinstance(maximum_input_length, int)
            or isinstance(maximum_input_length, bool)
            or maximum_input_length < 1
        ):
            raise RerankerError(
                "RERANKER_INVALID_CONFIGURATION: batch size and maximum input "
                "length must be positive integers."
            )
        resolved_device = _resolve_device(device)
        self.model_id = model_id.strip()
        self.model_revision = model_revision.strip() if model_revision else None
        self.device = resolved_device
        self.batch_size = batch_size
        self.maximum_input_length = maximum_input_length
        if model is not None:
            self.model = model
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "RERANKER_MODEL_LOAD_FAILED: Install sentence-transformers."
            ) from exc
        arguments: dict[str, Any] = {
            "device": self.device,
            "max_length": self.maximum_input_length,
        }
        if self.model_revision is not None:
            arguments["automodel_args"] = {"revision": self.model_revision}
            arguments["tokenizer_args"] = {"revision": self.model_revision}
        try:
            self.model = CrossEncoder(self.model_id, **arguments)
        except Exception as exc:
            raise RerankerError(
                f"RERANKER_MODEL_LOAD_FAILED: Could not load {self.model_id}: {exc}."
            ) from exc

    def rerank(
        self,
        claim: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not isinstance(candidates, list):
            raise RerankerError(
                "RERANKER_INVALID_CANDIDATE: Candidates must be a list."
            )
        if not candidates:
            return []
        if not isinstance(claim, str) or not claim.strip():
            raise RerankerError(
                "RERANKER_EMPTY_CLAIM: Claim must be a non-empty string."
            )
        _validate_top_k(top_k, len(candidates))
        _validate_candidates(candidates)
        pairs = [(claim.strip(), candidate["text"]) for candidate in candidates]
        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RerankerError(
                f"RERANKER_FAILED: Cross-encoder scoring failed: {exc}."
            ) from exc
        try:
            scores = np.asarray(raw_scores).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise RerankerError(
                f"RERANKER_INVALID_SCORE: Could not read model scores: {exc}."
            ) from exc
        if len(scores) != len(candidates):
            raise RerankerError(
                "RERANKER_SCORE_COUNT_MISMATCH: Model score count does not match candidates."
            )
        scored: list[dict[str, Any]] = []
        for candidate, raw_score in zip(candidates, scores, strict=True):
            try:
                score = float(raw_score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RerankerError(
                    "RERANKER_INVALID_SCORE: Model returned a non-numeric score."
                ) from exc
            if not math.isfinite(score):
                raise RerankerError(
                    "RERANKER_INVALID_SCORE: Model returned NaN or infinity."
                )
            output = dict(candidate)
            output["pre_rerank_rank"] = candidate["rank"]
            output["reranker_score"] = score
            scored.append(output)
        ranked = sorted(
            scored,
            key=lambda item: (
                -item["reranker_score"],
                item["pre_rerank_rank"],
                item["passage_id"],
            ),
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate["reranker_rank"] = rank
        self.last_scored_candidates = [dict(candidate) for candidate in ranked]
        selected = [dict(candidate) for candidate in ranked[:top_k]]
        for candidate in selected:
            candidate["rank"] = candidate["reranker_rank"]
        return selected
