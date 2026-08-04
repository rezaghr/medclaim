#!/usr/bin/env python3
"""Run validated MedClaim experiment configurations sequentially."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medclaim.experiments.configuration import (  # noqa: E402
    ExperimentConfigurationError,
    load_experiment_configurations,
)
from medclaim.experiments.runner import ExperimentRunner, ExperimentRunnerError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config-dir", type=Path)
    selection.add_argument("--config", type=Path, action="append")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--qualitative-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = (
        sorted(args.config_dir.glob("*.json"))
        if args.config_dir is not None
        else list(args.config)
    )
    try:
        configurations = load_experiment_configurations(paths)
        output, reused = ExperimentRunner().run(
            configurations,
            args.output_root,
            args.run_group,
            continue_on_error=args.continue_on_error,
            qualitative_seed=args.qualitative_seed,
        )
    except (ExperimentConfigurationError, ExperimentRunnerError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    print(f"Experiment run group: {output}")
    print(f"Status: {manifest['status']}")
    print(f"Reused: {str(reused).lower()}")
    return 0 if manifest["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
