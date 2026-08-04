"""Dataset-specific normalization adapters."""

from .constants import SUPPORTED_DATASETS
from .scifact import (
    SciFactPreparationError,
    claim_id_for,
    document_id_for,
    map_scifact_label,
    normalize_scifact_claim,
    normalize_scifact_document,
    prepare_scifact,
)
from .unified import UnifiedDatasetError, build_unified_dataset

__all__ = [
    "SUPPORTED_DATASETS",
    "SciFactPreparationError",
    "claim_id_for",
    "document_id_for",
    "map_scifact_label",
    "normalize_scifact_claim",
    "normalize_scifact_document",
    "prepare_scifact",
    "UnifiedDatasetError",
    "build_unified_dataset",
]
