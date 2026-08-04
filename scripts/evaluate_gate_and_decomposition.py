#!/usr/bin/env python3
"""Evaluate precomputed gate and decomposition pipeline predictions offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.evaluation.gate_decomposition import (  # noqa: E402
    GateDecompositionEvaluationError,
    evaluate_gate_and_decomposition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = evaluate_gate_and_decomposition(args.predictions, args.output_dir)
    except GateDecompositionEvaluationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Gate and decomposition evaluation completed successfully.\n")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
