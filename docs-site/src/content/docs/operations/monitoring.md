---
title: Monitoring
description: Prometheus metrics, structured logs and OpenTelemetry tracing for the anpr service.
---

## Prometheus metrics

`GET /metrics` returns metrics in Prometheus text format. Point a scrape job at it:

```yaml
scrape_configs:
  - job_name: anpr
    static_configs:
      - targets: ["anpr:8000"]
```

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `anpr_requests_total` | counter | method, path, status | HTTP requests handled. |
| `anpr_request_seconds` | histogram | method, path | Time to answer an HTTP request. |
| `anpr_inference_seconds` | histogram | route | Time spent in detection and OCR on `POST /api/v1/infer`. |
| `anpr_detections_total` | counter | route, parsed | Plates read; `parsed` is `valid` or `invalid`. |
| `anpr_confirmed_plates_total` | counter | province_code | Plates confirmed on the stream, by province. |
| `anpr_active_websocket_streams` | gauge | | Open `/ws/stream` connections. |

Two numbers are worth a dashboard panel: the share of `invalid` in `anpr_detections_total`, which rises
when the camera angle or focus is off, and the 95th percentile of `anpr_inference_seconds`.

## Logs

Logs come from structlog. As text by default; with `ANPR_LOG_JSON=true` every line is a JSON
object, ready for Loki, Elasticsearch or CloudWatch. Every log line written while handling a request
carries its `request_id`, `method` and `path`, and the response returns the same ID in the
`X-Request-ID` header, so a failed call can be found in the logs.

Events worth alerting on:

| Event | Meaning |
|---|---|
| `infer.image_rejected`, `stream.frame_rejected` | An image or frame was over the size limits. Many of them in a short time can mean someone is probing the service. |
| `stream.error` | A stream ended with an unexpected error. |
| `storage.retention.error` | The retention task couldn't delete old rows. |

## OpenTelemetry

Install the extra and set an OTLP endpoint; the service then traces every request:

```bash
pip install "anpr-pipeline[otel]"
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_SERVICE_NAME=anpr   # the default
anpr serve
```

Without the endpoint, or without the extra, tracing stays off and costs nothing.
