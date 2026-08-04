#!/usr/bin/env python3
"""Build an immutable combined medical evidence corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.corpus.combined import build_combined_corpus  # noqa: E402
from medclaim.corpus.scifact_corpus import CorpusBuildError  # noqa: E402


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/corpora"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--max-passage-words", type=positive_integer, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output = build_combined_corpus(
            args.dataset_dir,
            args.output_root,
            args.version,
            args.max_passage_words,
        )
    except CorpusBuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Combined medical evidence corpus built successfully.\n")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
