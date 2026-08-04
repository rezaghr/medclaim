#!/usr/bin/env python3
"""Measure a warm, deterministic demo pipeline without claiming production performance."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medclaim.safety import route_scope  # noqa: E402
from medclaim.security import build_verifier_prompt  # noqa: E402

TARGETS_MS = {
    "validation": 100.0,
    "bm25": 300.0,
    "dense_retrieval": 500.0,
    "fusion_metadata": 100.0,
    "reranking": 4000.0,
    "llm_verification": 10000.0,
    "total": 15000.0,
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _measure(operation: Callable[[], object]) -> float:
    started = time.perf_counter()
    operation()
    return (time.perf_counter() - started) * 1000


def profile(runs: int, warmups: int) -> dict:
    claim = "Vitamin D supplementation reduces respiratory infection incidence."
    passages = [
        {"passage_id": f"sample:p{i}", "text": "Synthetic evidence text for timing."}
        for i in range(20)
    ]
    operations: dict[str, Callable[[], object]] = {
        "validation": lambda: route_scope(claim),
        "bm25": lambda: sorted(
            passages, key=lambda item: claim.casefold().count(item["text"].casefold())
        ),
        "dense_retrieval": lambda: [sum(ord(char) for char in item["text"]) for item in passages],
        "fusion_metadata": lambda: list(dict.fromkeys(item["passage_id"] for item in passages)),
        "reranking": lambda: sorted(passages, key=lambda item: len(item["text"]), reverse=True)[:5],
        "llm_verification": lambda: build_verifier_prompt(claim, passages[:5]),
    }
    for _ in range(warmups):
        for operation in operations.values():
            operation()
    timings = {stage: [] for stage in (*operations, "total")}
    failures = 0
    for _ in range(runs):
        total_started = time.perf_counter()
        try:
            for stage, operation in operations.items():
                timings[stage].append(_measure(operation))
        except Exception:
            failures += 1
        timings["total"].append((time.perf_counter() - total_started) * 1000)
    stages = {}
    for stage, values in timings.items():
        stages[stage] = {
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "p99_ms": _percentile(values, 0.99),
            "mean_ms": statistics.fmean(values),
            "planning_target_ms": TARGETS_MS[stage],
            "synthetic_timing_under_target": _percentile(values, 0.50) < TARGETS_MS[stage],
            "target_demonstrated": False,
        }
    return {
        "profile_version": "demo-profile-v1",
        "measurement_mode": "synthetic_fake_components",
        "measurement_warning": "These timings exercise fake in-process components, not external models or production artifacts.",
        "warm_up_policy": {"discarded_runs": warmups, "measured_runs": runs},
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or "not-reported",
            "logical_cpu_count": os.cpu_count(),
        },
        "provider": "fake",
        "model": "fake-verifier-v1",
        "corpus_size": len(passages),
        "candidate_counts": {"retrieved": 20, "reranked": 5},
        "stages": stages,
        "token_usage": {"input": 0, "output": 0, "available": False},
        "approximate_cost": {"currency": "USD", "amount": 0.0, "available": False},
        "failed_requests": failures,
        "timed_out_requests": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "performance")
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups must be non-negative")
    report = profile(args.runs, args.warmups)
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "demo-profile-v1.json"
    csv_path = args.output_root / "demo-profile-v1.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "mean_ms",
                "planning_target_ms",
                "synthetic_timing_under_target",
                "target_demonstrated",
            ],
        )
        writer.writeheader()
        for stage, values in report["stages"].items():
            writer.writerow({"stage": stage, **values})
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
