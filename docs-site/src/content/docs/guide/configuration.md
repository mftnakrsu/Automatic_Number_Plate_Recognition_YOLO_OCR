---
title: Configuration
description: Every ANPR_ environment variable, its default and what it changes.
---

All settings are environment variables starting with `ANPR_`. They can also go in a `.env` file in
the directory you start the service from; `.env.example` in the repository lists them all. Settings
are read once at start-up, so restart the service after changing one. `anpr dump-config` shows the
values in effect.

## Storage and privacy

| Variable | Default | Meaning |
|---|---|---|
| `ANPR_PLATE_HMAC_PEPPER` | random, with a warning | The secret plates are hashed with. At least 32 characters; `anpr generate-pepper` makes one. **Set it in production**: a random pepper changes on every restart, and then the same plate gets a different hash. |
| `ANPR_DATABASE_URL` | `sqlite+aiosqlite:///data/anpr.db` | Where detections are stored. For PostgreSQL, install `asyncpg` and use `postgresql+asyncpg://user:pass@host/db`. |
| `ANPR_RETENTION_HOURS` | `720` (30 days) | Rows older than this are deleted. The check runs at start-up and then every hour. |

## Pipeline

| Variable | Default | Meaning |
|---|---|---|
| `ANPR_DETECTOR_MODEL` | `yolo-v9-t-384-license-plate-end2end` | fast-alpr detector model name, or a path to your own ONNX file. |
| `ANPR_DETECTOR_CONF_THRESHOLD` | `0.4` | Boxes the detector is less sure about than this are dropped. |
| `ANPR_OCR_MODEL` | `cct-s-v2-global-model` | fast-plate-ocr model name. |
| `ANPR_DEVICE` | `auto` | Where the OCR model runs: `auto`, `cpu`, `cuda` or `openvino`. |
| `ANPR_DETECT_EVERY_N_FRAMES` | `3` | On the WebSocket stream, process only every Nth frame. |
| `ANPR_MIN_TRACK_DWELL` | `5` | Processed frames a plate must be read on before it is confirmed. |

`ANPR_SR_MIN_PLATE_WIDTH` and `ANPR_OCR_FALLBACK_THRESHOLD` are read, but nothing uses them yet;
they are reserved for super-resolution and a second OCR engine. See the [Roadmap](/roadmap/).

## API

| Variable | Default | Meaning |
|---|---|---|
| `ANPR_API_HOST` | `0.0.0.0` | Address `anpr serve` listens on. |
| `ANPR_API_PORT` | `8000` | Port `anpr serve` listens on. |
| `ANPR_MAX_UPLOAD_MB` | `10` | Largest file `POST /api/v1/infer` accepts. |
| `ANPR_MAX_IMAGE_PIXELS` | `40000000` | Largest decoded image, in pixels, on both the upload and the stream. 40 MP is above 8K. |
| `ANPR_MAX_WS_FRAME_BYTES` | `4194304` (4 MB) | Largest frame on `/ws/stream`. |
| `ANPR_CORS_ORIGINS` | `*` | Comma-separated origins allowed to call the API from a browser. Narrow it in production. |

## Logging and tracing

| Variable | Default | Meaning |
|---|---|---|
| `ANPR_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `ANPR_LOG_JSON` | `false` | JSON log lines, for Loki, Elasticsearch and the like. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | not set | Turns on OpenTelemetry tracing. Needs the `otel` extra. |
| `OTEL_SERVICE_NAME` | `anpr` | Service name on the traces. |
