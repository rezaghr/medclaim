import io
import json
import logging

import pytest

from medclaim.observability.logging import JsonFormatter, RequestTracer, safe_claim_hash
from medclaim.observability.metrics import MetricsRegistry


def test_trace_is_correlated_and_excludes_raw_claim():
    stream = io.StringIO()
    logger = logging.Logger("test")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    claim = "My private medical claim"
    tracer = RequestTracer(
        logger, {"request_id": "req-1", "safe_claim_hash": safe_claim_hash(claim)}
    )
    tracer.emit("request_received", claim_length=len(claim))
    value = json.loads(stream.getvalue())
    assert value["request_id"] == "req-1"
    assert claim not in stream.getvalue()
    with pytest.raises(ValueError, match="TRACE_PRIVACY_VIOLATION"):
        tracer.emit("bad", claim_text=claim)


def test_metrics_have_bounded_labels_and_no_user_identifiers():
    metrics = MetricsRegistry()
    metrics.observe("verification_requests_total")
    metrics.observe("verification_results_total", verdict="SUPPORTS")
    metrics.observe("request_duration_seconds", 0.2)
    rendered = metrics.render()
    assert 'verification_results_total{verdict="SUPPORTS"}' in rendered
    assert "request_id" not in rendered
    with pytest.raises(ValueError, match="Unsafe"):
        metrics.observe("verification_results_total", request_id="user-value")
