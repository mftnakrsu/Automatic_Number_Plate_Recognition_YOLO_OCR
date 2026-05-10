"""Prometheus metrics + OpenTelemetry tracer hook.

Metrics are exposed at `/metrics` by the FastAPI app. OpenTelemetry is opt-in:
when `OTEL_EXPORTER_OTLP_ENDPOINT` is set the app installs the FastAPI
instrumentation + an OTLP gRPC span exporter on startup. If the OTel packages
are not installed, OTel setup is skipped silently — this keeps the dependency
surface minimal in the default install.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


REQUESTS = Counter(
    "anpr_requests_total",
    "Total HTTP requests handled.",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "anpr_request_seconds",
    "HTTP request latency seconds.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INFERENCE_LATENCY = Histogram(
    "anpr_inference_seconds",
    "End-to-end inference time per call.",
    ["route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

DETECTIONS = Counter(
    "anpr_detections_total",
    "Total plate detections produced.",
    ["route", "parsed"],
)

CONFIRMED_PLATES = Counter(
    "anpr_confirmed_plates_total",
    "Total temporally-confirmed plate reads.",
    ["province_code"],
)

ACTIVE_STREAMS = Gauge(
    "anpr_active_websocket_streams",
    "Currently active WebSocket streaming sessions.",
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def configure_otel(app: FastAPI) -> bool:
    """Install OpenTelemetry FastAPI instrumentation if configured.

    Returns True when instrumentation was set up; False otherwise (env not
    set, or the optional packages not installed).
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace  # pyright: ignore[reportMissingImports]
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # pyright: ignore[reportMissingImports]
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import (  # pyright: ignore[reportMissingImports]
            FastAPIInstrumentor,
        )
        from opentelemetry.sdk.resources import Resource  # pyright: ignore[reportMissingImports]
        from opentelemetry.sdk.trace import TracerProvider  # pyright: ignore[reportMissingImports]
        from opentelemetry.sdk.trace.export import (  # pyright: ignore[reportMissingImports]
            BatchSpanProcessor,
        )
    except ImportError:
        return False

    resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "anpr")})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    return True
