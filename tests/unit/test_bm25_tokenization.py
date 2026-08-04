import pytest

from medclaim.retrieval.tokenization import tokenize_bm25


def test_lowercase_tokenization():
    assert tokenize_bm25("Vitamin D") == ["vitamin", "d"]


def test_punctuation_tokenization():
    assert tokenize_bm25("Aspirin, ibuprofen, and acetaminophen.") == [
        "aspirin",
        "ibuprofen",
        "and",
        "acetaminophen",
    ]


def test_hyphenated_text():
    assert tokenize_bm25("COVID-19-related outcomes") == [
        "covid",
        "19",
        "related",
        "outcomes",
    ]


def test_unicode_nfkc_normalization():
    assert tokenize_bm25("Ｖｉｔａｍｉｎ Ｄ") == tokenize_bm25("Vitamin D")


@pytest.mark.parametrize("text", ["", "...", "— !!!"])
def test_empty_tokenization(text):
    assert tokenize_bm25(text) == []
