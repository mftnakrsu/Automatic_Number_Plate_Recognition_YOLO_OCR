---
title: Install
description: Install anpr-pipeline from PyPI, run it with Docker, or set it up from source with uv.
---

anpr-pipeline needs Python 3.11 to 3.14. The models download on first use (about 30 MB) and are
cached after that.

## From PyPI

```bash
pip install anpr-pipeline
```

This installs the `anpr` command. Check it:

```bash
anpr version
```

### Optional extras

| Extra | What it adds |
|---|---|
| `anpr-pipeline[gpu]` | `onnxruntime-gpu`, for running the models on an NVIDIA GPU. |
| `anpr-pipeline[otel]` | OpenTelemetry tracing for the API. See [Monitoring](/operations/monitoring/). |

## With Docker

The repository has a `Dockerfile` and a `docker-compose.yml`. The service needs a secret to hash
plates with, called the pepper. Generate one and start the service:

```bash
git clone https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR.git
cd Automatic_Number_Plate_Recognition_YOLO_OCR
export ANPR_PLATE_HMAC_PEPPER="$(python -c 'import secrets; print(secrets.token_hex(32))')"
docker compose up --build
```

Open <http://localhost:8000>. The database lives in the `anpr-data` volume, so it survives restarts.
Keep the pepper: with a different one, the same plate gets a different hash. See
[Privacy](/operations/privacy/) for why.

## From source

You need [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR.git
cd Automatic_Number_Plate_Recognition_YOLO_OCR
uv sync --all-extras --group dev
cp .env.example .env
uv run anpr generate-pepper   # paste the output into .env as ANPR_PLATE_HMAC_PEPPER
uv run anpr serve
```

Next: [read your first plate](/start/first-plate/).
