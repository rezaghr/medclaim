"""Authoritative dataset identifiers shared across MedClaim."""

SUPPORTED_DATASETS = {
    "scifact",
    "healthver",
    "pubhealth",
}

DATASET_ORDER = ("scifact", "healthver", "pubhealth")
DATASET_RANK = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}

assert set(DATASET_ORDER) == SUPPORTED_DATASETS
