"""Small Prometheus-text registry with bounded labels only."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock

_METRIC_TYPES = {
    "verification_requests_total": "counter",
    "verification_results_total": "counter",
    "scope_limited_requests_total": "counter",
    "request_duration_seconds": "summary",
    "retrieval_duration_seconds": "summary",
    "reranking_duration_seconds": "summary",
    "verification_duration_seconds": "summary",
    "provider_errors_total": "counter",
    "provider_timeouts_total": "counter",
    "schema_validation_failures_total": "counter",
    "evidence_abstentions_total": "counter",
    "retrieved_candidates_count": "summary",
    "reranked_candidates_count": "summary",
    "verifier_input_tokens_total": "counter",
    "verifier_output_tokens_total": "counter",
    "estimated_provider_cost_total": "counter",
}
_ALLOWED_LABELS = {
    "verification_results_total": {"verdict"},
    "scope_limited_requests_total": {"category"},
    "retrieval_duration_seconds": {"mode"},
    "verification_duration_seconds": {"verifier"},
    "provider_errors_total": {"code"},
    "evidence_abstentions_total": {"reason"},
}


class MetricsRegistry:
    def __init__(self) -> None:
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._counts: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
        self._lock = Lock()

    def observe(self, name: str, value: float = 1.0, **labels: str) -> None:
        if name not in _METRIC_TYPES:
            raise ValueError(f"Unknown metric {name!r}.")
        if set(labels) != _ALLOWED_LABELS.get(name, set()):
            raise ValueError(f"Unsafe or invalid labels for metric {name!r}.")
        key = (name, tuple(sorted((key, str(item)) for key, item in labels.items())))
        with self._lock:
            self._values[key] += float(value)
            if _METRIC_TYPES[name] == "summary":
                self._counts[key] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            keys = set(self._values)
            for name in _METRIC_TYPES:
                matching = sorted(key for key in keys if key[0] == name)
                if not matching and name not in _ALLOWED_LABELS:
                    matching = [(name, ())]
                lines.append(f"# TYPE {name} {_METRIC_TYPES[name]}")
                for key in matching:
                    _, labels = key
                    suffix = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
                    value = self._values.get(key, 0.0)
                    if _METRIC_TYPES[name] == "summary":
                        lines.append(f"{name}_count{suffix} {self._counts.get(key, 0)}")
                        lines.append(f"{name}_sum{suffix} {value}")
                    else:
                        lines.append(f"{name}{suffix} {value}")
        return "\n".join(lines) + "\n"
