"""Verification result models, aggregation, and component pipeline."""

from .aggregation import AggregationError, aggregate_component_results
from .models import AtomicClaimResult, VerificationResult
from .pipeline import ComponentVerificationError, VerificationPipeline

__all__ = [
    "AggregationError",
    "AtomicClaimResult",
    "ComponentVerificationError",
    "VerificationPipeline",
    "VerificationResult",
    "aggregate_component_results",
]
