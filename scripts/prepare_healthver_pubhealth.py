#!/usr/bin/env python3
"""Normalize the official HealthVer and PUBHEALTH releases for MedClaim."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medclaim.datasets.unified import normalized_dataset_content_hash  # noqa: E402


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def finish(output: Path, dataset: str, mapping: dict[str, str], claims, documents) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "claims.jsonl", claims)
    write_jsonl(output / "documents.jsonl", documents)
    write_json(output / "label_mapping.json", {"dataset": dataset, "source_to_unified": mapping})
    write_json(
        output / "quality_report.json",
        {
            "status": "success",
            "dataset": dataset,
            "claim_count": len(claims),
            "document_count": len(documents),
        },
    )
    write_json(
        output / "manifest.json",
        {
            "artifact_type": "normalized_dataset",
            "dataset": dataset,
            "adapter_version": "1.0.0",
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "claim_count": len(claims),
            "document_count": len(documents),
            "content_hash": normalized_dataset_content_hash(claims, documents),
            "outputs": {
                "claims": "claims.jsonl",
                "documents": "documents.jsonl",
                "label_mapping": "label_mapping.json",
                "quality_report": "quality_report.json",
            },
        },
    )


def prepare_healthver(source: Path, output: Path) -> None:
    mapping = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES", "NEI": "NOT_ENOUGH_INFO"}
    documents = []
    for line in (source / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sentences = [text.strip() for text in row["abstract"] if text.strip()]
        documents.append(
            {
                "document_id": f"healthver:document:{row['doc_id']}",
                "dataset": "healthver",
                "source_document_id": str(row["doc_id"]),
                "title": row.get("title"),
                "source_type": "scientific_abstract",
                "source_url": None,
                "publication_year": None,
                "text": " ".join(sentences),
                "sentences": sentences,
                "metadata": {},
            }
        )
    claims = []
    for split in ("train", "dev", "test"):
        path = source / f"claims_{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            evidence_sets = []
            labels = []
            for document_id, groups in row.get("evidence", {}).items():
                for group in groups:
                    label = group["label"]
                    labels.append(label)
                    evidence_sets.append(
                        {
                            "evidence_set_id": f"healthver:claim:{row['id']}:evidence:{len(evidence_sets)}",
                            "relationship": mapping[label],
                            "document_id": f"healthver:document:{document_id}",
                            "sentence_indices": group["sentences"],
                        }
                    )
            source_label = labels[0] if labels and len(set(labels)) == 1 else "NEI"
            claims.append(
                {
                    "claim_id": f"healthver:claim:{row['id']}",
                    "dataset": "healthver",
                    "source_claim_id": str(row["id"]),
                    "claim_text": row["claim"].strip(),
                    "original_split": split,
                    "original_label": source_label,
                    "unified_label": mapping[source_label],
                    "language": "en",
                    "evidence_sets": evidence_sets,
                    "gold_explanation": None,
                    "metadata": {"cited_document_ids": [str(value) for value in row.get("doc_ids", [])]},
                }
            )
    finish(output, "healthver", mapping, claims, documents)


def prepare_pubhealth(source: Path, output: Path) -> None:
    mapping = {"false": "REFUTES", "mixture": "MIXED", "true": "SUPPORTS", "unproven": "NOT_ENOUGH_INFO"}
    claims, documents = [], []
    seen_claims: set[str] = set()
    for split, filename in (("train", "train.tsv"), ("dev", "dev.tsv"), ("test", "test.tsv")):
        with (source / filename).open(encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle, delimiter="\t")
            next(rows, None)
            for index, row in enumerate(rows):
                if split == "test":
                    row = row[1:] if len(row) == 10 else row
                if len(row) != 9:
                    continue
                source_id, claim, date, explanation, checkers, main_text, sources, label, subjects = row
                if label not in mapping or not claim.strip() or not main_text.strip():
                    continue
                source_id = source_id.strip() or f"{split}-{index}"
                if source_id in seen_claims:
                    source_id = f"{split}-{source_id}"
                seen_claims.add(source_id)
                claim_id = f"pubhealth:claim:{source_id}"
                document_id = f"pubhealth:document:{source_id}"
                text = main_text.strip()
                documents.append(
                    {
                        "document_id": document_id,
                        "dataset": "pubhealth",
                        "source_document_id": source_id,
                        "title": None,
                        "source_type": "fact_check_article",
                        "source_url": None,
                        "publication_year": None,
                        "text": text,
                        "sentences": [text],
                        "metadata": {"date_published": date, "sources": sources, "subjects": subjects},
                    }
                )
                claims.append(
                    {
                        "claim_id": claim_id,
                        "dataset": "pubhealth",
                        "source_claim_id": source_id,
                        "claim_text": claim.strip(),
                        "original_split": split,
                        "original_label": label,
                        "unified_label": mapping[label],
                        "language": "en",
                        "evidence_sets": [{
                            "evidence_set_id": f"{claim_id}:evidence:0",
                            "relationship": mapping[label],
                            "document_id": document_id,
                            "sentence_indices": [0],
                        }],
                        "gold_explanation": explanation.strip() or None,
                        "metadata": {"fact_checkers": checkers},
                    }
                )
    finish(output, "pubhealth", mapping, claims, documents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthver-source", type=Path, required=True)
    parser.add_argument("--pubhealth-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    prepare_healthver(args.healthver_source, args.output_root / "healthver")
    prepare_pubhealth(args.pubhealth_source, args.output_root / "pubhealth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
