#!/usr/bin/env python3
"""Export a deterministic stratified explanation-review CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.explanation.evaluation import export_explanation_review  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-by", nargs="+", choices=("dataset", "label", "correctness"),
        default=("dataset", "label", "correctness"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = export_explanation_review(
            args.predictions,
            args.output,
            args.sample_size,
            args.seed,
            tuple(args.stratify_by),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Review sample: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
