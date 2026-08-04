from medclaim.ui import collect_used_evidence


def test_collect_used_evidence_resolves_text_in_citation_order():
    verification = {
        "evidence_used": ["p2", "p1"],
        "component_results": [
            {
                "retrieved_candidates": [
                    {
                        "rank": 1,
                        "passage_id": "p1",
                        "document_id": "d1",
                        "dataset": "pubhealth",
                        "text": "First complete passage.",
                        "bm25_score": 8.0,
                    },
                    {
                        "rank": 2,
                        "passage_id": "p2",
                        "document_id": "d2",
                        "dataset": "healthver",
                        "text": "Second complete passage.",
                        "bm25_score": 7.0,
                    },
                ],
                "model_input_evidence": [
                    {"passage_id": "p1", "text": "First complete passage."},
                    {"passage_id": "p2", "text": "Second complete passage."},
                ],
            }
        ],
    }

    result = collect_used_evidence(verification)
    assert [row["passage_id"] for row in result] == ["p2", "p1"]
    assert result[0]["text"] == "Second complete passage."
    assert result[0]["document_id"] == "d2"
    assert result[1]["bm25_score"] == 8.0


def test_collect_used_evidence_preserves_ids_for_old_api_responses():
    assert collect_used_evidence({"evidence_used": ["p1"]}) == [
        {"passage_id": "p1"}
    ]


def test_collect_used_evidence_ignores_invalid_and_duplicate_ids():
    verification = {"evidence_used": ["p1", "p1", None, 3]}
    assert collect_used_evidence(verification) == [{"passage_id": "p1"}]
