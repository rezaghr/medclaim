#!/usr/bin/env python3
"""Calibrate an evidence-sufficiency gate from development predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.evidence_gate.calibration import calibrate_evidence_gate  # noqa: E402
from medclaim.evidence_gate.gate import EvidenceGateError  # noqa: E402


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--gold-evidence", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="dev", choices=("dev", "validation", "test"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/evidence-gates"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--minimum-relevant-passages", type=positive_integer, default=1)
    parser.add_argument("--minimum-unique-documents", type=positive_integer, default=1)
    parser.add_argument("--score-field", default="reranker_score")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = calibrate_evidence_gate(
            args.claims,
            args.gold_evidence,
            args.split_manifest,
            args.predictions,
            args.split,
            args.output_root,
            args.version,
            args.minimum_relevant_passages,
            args.minimum_unique_documents,
            args.score_field,
        )
    except EvidenceGateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Evidence gate calibrated successfully.\n")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
