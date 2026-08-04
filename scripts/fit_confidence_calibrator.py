#!/usr/bin/env python3
"""Fit a development-only probability-of-correctness calibrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.calibration.calibrator import CalibrationError, fit_confidence_calibrator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--method", choices=("none", "logistic", "isotonic"), default="logistic")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/calibration"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = fit_confidence_calibrator(
            args.predictions, args.split_manifest, args.method, args.output_root,
            args.version, args.bins, args.seed,
        )
    except CalibrationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Confidence calibrator: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
