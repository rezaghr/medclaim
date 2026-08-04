from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medclaim.datasets.unified import normalized_dataset_content_hash


MAPPINGS = {
    "scifact": {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"},
    "healthver": {
        "SUPPORT": "SUPPORTS",
        "REFUTE": "REFUTES",
        "NEUTRAL": "NOT_ENOUGH_INFO",
    },
    "pubhealth": {
        "true": "SUPPORTS",
        "false": "REFUTES",
        "unproven": "NOT_ENOUGH_INFO",
        "mixture": "MIXED",
    },
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def create_normalized_fixture(root: Path, dataset: str) -> Path:
    directory = root / dataset
    directory.mkdir(parents=True)
    if dataset == "scifact":
        documents = [{
            "document_id": "scifact:document:10",
            "dataset": "scifact",
            "source_document_id": "10",
            "title": "Immunity study",
            "source_type": "scientific_abstract",
            "abstract_sentences": ["Vitamin D supports immune function."],
            "text": "Vitamin D supports immune function.",
            "metadata": {"topic": "immunity"},
        }]
        claims = [claim("scifact", "1", "Vitamin D supports immunity.", "SUPPORT", "SUPPORTS", "train", "10", [0])]
    elif dataset == "healthver":
        documents = [{
            "document_id": "healthver:document:20",
            "dataset": "healthver",
            "source_document_id": "20",
            "title": None,
            "source_type": "health_evidence",
            "source_url": "https://example.invalid/health",
            "publication_year": None,
            "text": "Smoking increases respiratory disease risk.",
            "sentences": [],
            "metadata": {"topic": "respiratory"},
        }]
        claims = [claim("healthver", "2", "Smoking is harmless.", "REFUTE", "REFUTES", "dev", "20", [], "Smoking is associated with disease.")]
    elif dataset == "pubhealth":
        documents = [{
            "document_id": "pubhealth:document:30",
            "dataset": "pubhealth",
            "source_document_id": "30",
            "title": "Public health review",
            "source_type": "public_health_article",
            "source_url": None,
            "publication_year": 2024,
            "text": "Vaccines reduce severe disease. No intervention removes all risk.",
            "sentences": ["Vaccines reduce severe disease.", "No intervention removes all risk."],
            "metadata": {"topic": "vaccines"},
        }]
        claims = [
            claim("pubhealth", "3", "Vaccines have mixed outcomes.", "mixture", "MIXED", "test", "30", [], "Benefits and limitations coexist."),
            claim("pubhealth", "4", "An unresolved public claim.", "unproven", "NOT_ENOUGH_INFO", "train", None, [], "Insufficient published evidence."),
        ]
    else:
        raise ValueError(dataset)
    write_jsonl(directory / "claims.jsonl", claims)
    write_jsonl(directory / "documents.jsonl", documents)
    write_json(directory / "label_mapping.json", {"dataset": dataset, "schema_version": "1.0.0", "mappings": MAPPINGS[dataset]})
    write_json(directory / "quality_report.json", {"dataset": dataset, "status": "success"})
    write_json(directory / "manifest.json", {
        "artifact_type": "normalized_dataset",
        "dataset": dataset,
        "adapter_version": "1.0.0",
        "claim_count": len(claims),
        "document_count": len(documents),
        "content_hash": normalized_dataset_content_hash(claims, documents),
        "outputs": {
            "claims": "claims.jsonl",
            "documents": "documents.jsonl",
            "label_mapping": "label_mapping.json",
            "quality_report": "quality_report.json",
        },
    })
    return directory


def claim(
    dataset: str,
    source_id: str,
    text: str,
    original_label: str,
    unified_label: str,
    split: str,
    document_source_id: str | None,
    indices: list[int],
    explanation: str | None = None,
) -> dict[str, Any]:
    claim_id = f"{dataset}:claim:{source_id}"
    evidence = []
    if document_source_id is not None:
        evidence = [{
            "evidence_set_id": f"{claim_id}:evidence:0",
            "relationship": unified_label,
            "document_id": f"{dataset}:document:{document_source_id}",
            "source_sentence_indices": indices,
        }]
    return {
        "claim_id": claim_id,
        "dataset": dataset,
        "source_claim_id": source_id,
        "claim_text": text,
        "original_split": split,
        "original_label": original_label,
        "unified_label": unified_label,
        "language": "en",
        "evidence_sets": evidence,
        "gold_explanation": explanation,
        "metadata": {"source": "synthetic"},
    }


def create_all_normalized_fixtures(root: Path) -> dict[str, Path]:
    return {dataset: create_normalized_fixture(root, dataset) for dataset in MAPPINGS}
