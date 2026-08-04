#!/usr/bin/env python3
"""Build an immutable unified medical dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.datasets.unified import UnifiedDatasetError, build_unified_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scifact-dir", type=Path, default=Path("data/processed/scifact"))
    parser.add_argument("--healthver-dir", type=Path, default=Path("data/processed/healthver"))
    parser.add_argument("--pubhealth-dir", type=Path, default=Path("data/processed/pubhealth"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = build_unified_dataset(
            args.scifact_dir,
            args.healthver_dir,
            args.pubhealth_dir,
            args.output_root,
            args.version,
        )
    except UnifiedDatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Unified medical dataset built successfully.\n")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
