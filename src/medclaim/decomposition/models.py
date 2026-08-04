"""Validated decomposition record types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AtomicClaim:
    index: int
    text: str
    source_span: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimDecomposition:
    is_compound: bool
    atomic_claims: list[AtomicClaim]
    explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compound": self.is_compound,
            "atomic_claims": [claim.to_dict() for claim in self.atomic_claims],
            "explanation": self.explanation,
        }
