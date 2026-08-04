"""Structured, bounded claim decomposition."""

from .decomposer import (
    ClaimDecomposer,
    DecompositionError,
    DecompositionOutcome,
    is_potentially_compound,
)
from .models import AtomicClaim, ClaimDecomposition

__all__ = [
    "AtomicClaim",
    "ClaimDecomposition",
    "ClaimDecomposer",
    "DecompositionError",
    "DecompositionOutcome",
    "is_potentially_compound",
]
