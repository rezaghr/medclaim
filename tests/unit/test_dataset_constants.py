from medclaim.datasets import SUPPORTED_DATASETS
from medclaim.datasets.constants import DATASET_ORDER


def test_supported_datasets_are_authoritative():
    assert SUPPORTED_DATASETS == {
        "scifact",
        "healthver",
        "pubhealth",
    }
    assert set(DATASET_ORDER) == SUPPORTED_DATASETS
