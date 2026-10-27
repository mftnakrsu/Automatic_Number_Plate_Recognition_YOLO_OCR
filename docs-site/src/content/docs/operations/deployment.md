---
title: Deployment
description: Run anpr-pipeline in production with Docker, PostgreSQL, a reverse proxy and a GPU.
---

## Docker

The image is built in two stages on `python:3.13-slim` and runs as an unprivileged user. Its health
check calls `/health`. `docker-compose.yml` is a ready single-host setup:

```bash
export ANPR_PLATE_HMAC_PEPPER="…"   # from `anpr generate-pepper`, kept in your secret store
docker compose up -d --build
```

Compose refuses to start without the pepper. The SQLite database lives in the `anpr-data` volume.
Tagged releases also publish an image to `ghcr.io/mftnakrsu/anpr-pipeline`.

## Checklist

- **Set the pepper** from a secret store, and keep it the same across restarts and machines.
- **Put a reverse proxy in front.** The API has no authentication. Let nginx, Caddy, Traefik or your
  ingress handle TLS and access, and don't expose port 8000 directly.
- **Narrow CORS.** Set `ANPR_CORS_ORIGINS` to the origins that really call the API.
- **Pick the retention** your purpose needs with `ANPR_RETENTION_HOURS`.
- **Log as JSON** (`ANPR_LOG_JSON=true`, the default in Compose) if a log collector reads them.

## PostgreSQL

SQLite is fine for one machine. For more, use PostgreSQL. The driver isn't a dependency, so install
it next to the package:

```bash
pip install anpr-pipeline asyncpg
export ANPR_DATABASE_URL="postgresql+asyncpg://anpr:secret@db:5432/anpr"
```

The table is created on start-up if it doesn't exist.

## GPU

The OCR model can run on an NVIDIA GPU with ONNX Runtime's CUDA build:

```bash
pip install "anpr-pipeline[gpu]"
export ANPR_DEVICE=cuda
```

`ANPR_DEVICE=openvino` uses Intel's OpenVINO, if that runtime is installed. `auto` picks what is
available.

## Limits worth knowing

- Uploads are capped at `ANPR_MAX_UPLOAD_MB`, stream frames at `ANPR_MAX_WS_FRAME_BYTES`, and both at
  `ANPR_MAX_IMAGE_PIXELS` decoded pixels. Lower them if your cameras are small, to bound memory
  further.
- One service process holds one copy of each model. Inference runs in a thread pool, so a single
  process serves several streams, but CPU is the limit; scale out with more containers.

## Reporting security problems

Please don't open a public issue. Use a
[private security advisory](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/security/advisories/new)
on GitHub, as described in
[SECURITY.md](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/blob/main/SECURITY.md).
