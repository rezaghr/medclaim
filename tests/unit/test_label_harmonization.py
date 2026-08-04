import pytest

from medclaim.datasets.label_schema import UNIFIED_LABELS, extract_source_mapping


def test_four_label_schema_includes_mixed():
    assert UNIFIED_LABELS == ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "MIXED")


def test_adapter_mapping_shape_is_loaded_and_validated():
    assert extract_source_mapping(
        {"dataset": "pubhealth", "mappings": {"mixture": "MIXED"}},
        "pubhealth",
    ) == {"mixture": "MIXED"}
    with pytest.raises(ValueError, match="unknown label"):
        extract_source_mapping({"mappings": {"mixture": "PARTIAL"}}, "pubhealth")
