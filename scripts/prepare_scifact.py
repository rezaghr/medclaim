#!/usr/bin/env python3
"""Normalize local SciFact data into MedClaim artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.datasets.scifact import (  # noqa: E402
    SciFactPreparationError,
    prepare_scifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/scifact"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/scifact"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the generated SciFact output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = prepare_scifact(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except SciFactPreparationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("SciFact dataset prepared successfully.\n")
    print(f"Output: {args.output_dir}")
    print(f"Documents: {report['document_count']}")
    print(f"Claims: {report['claim_count']}")
    print(f"Evidence sets: {report['evidence_set_count']}")
    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
