#!/usr/bin/env python3
"""Build an immutable, versioned SciFact evidence corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.corpus.scifact_corpus import (  # noqa: E402
    CorpusBuildError,
    build_scifact_corpus,
)


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/processed/scifact")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/corpora")
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--merge-short-sentences", action="store_true")
    parser.add_argument(
        "--short-sentence-word-threshold",
        type=positive_integer,
        default=5,
    )
    parser.add_argument(
        "--max-passage-words", type=positive_integer, default=120
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_dir = build_scifact_corpus(
            input_dir=args.input_dir,
            output_root=args.output_root,
            version=args.version,
            merge_short_sentences=args.merge_short_sentences,
            short_sentence_word_threshold=args.short_sentence_word_threshold,
            max_passage_words=args.max_passage_words,
        )
    except CorpusBuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("SciFact evidence corpus built successfully.\n")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
