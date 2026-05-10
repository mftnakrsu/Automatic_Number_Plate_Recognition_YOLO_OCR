# anpr-pipeline — Automatic Number Plate Recognition

[![PyPI](https://img.shields.io/pypi/v/anpr-pipeline.svg)](https://pypi.org/project/anpr-pipeline/)
[![CI](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/actions/workflows/ci.yml/badge.svg)](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

End-to-end license plate recognition: **plate detection → tracking → OCR → Turkish format parsing → temporal voting → privacy-aware persistence**, served behind a FastAPI HTTP + WebSocket API.

> Originally written in 2022 with a YOLOv5 fork + EasyOCR + Flask 1.1. Rewritten in 2026 on top of [`fast-alpr`](https://github.com/ankandrew/fast-alpr) + [`fast-plate-ocr`](https://github.com/ankandrew/fast-plate-ocr) + FastAPI + uv. The original code is preserved verbatim under [`legacy/`](legacy/) for reference.

---

## Quickstart

### Via PyPI

```bash
pip install anpr-pipeline
export ANPR_PLATE_HMAC_PEPPER="$(python -c 'import secrets; print(secrets.token_hex(32))')"
anpr serve                          # FastAPI on http://localhost:8000
anpr infer path/to/plate.jpg        # one-shot CLI
```

### Docker

```bash
export ANPR_PLATE_HMAC_PEPPER="$(python -c 'import secrets; print(secrets.token_hex(32))')"
docker compose up --build
```

Open <http://localhost:8000>.

### From source, via `uv`

```bash
git clone https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR.git
cd Automatic_Number_Plate_Recognition_YOLO_OCR
uv sync --all-extras --group dev
cp .env.example .env
# Generate a pepper and paste it into .env
uv run anpr generate-pepper
uv run anpr serve            # FastAPI on http://localhost:8000
# or: single-image inference
uv run anpr infer tests/fixtures/sample_plate.jpg
```

---

## What's in the box

| Layer | Implementation |
|---|---|
| **Detector** | [`fast-alpr`](https://github.com/ankandrew/fast-alpr)'s bundled YOLOv9-t 384 ONNX license-plate model (MIT, 65+ countries) |
| **OCR** | [`fast-plate-ocr`](https://github.com/ankandrew/fast-plate-ocr) `cct-s-v2-global-model` (sub-millisecond GPU, per-character confidences) |
| **Tracker** | Dependency-free greedy IoU tracker (swap to ByteTrack via the `Tracker` Protocol) |
| **Turkish parser** | All 81 province codes, strict regex on `NN [A]{1-3} [N]{2-4}` |
| **Confusion fix** | Position-aware OCR fixups (0/O, 1/I/L, 5/S, 8/B, 2/Z, 6/G, 0/D/Q) |
| **Temporal voting** | Per-character-position majority vote over top-K confidence-weighted reads per track |
| **Web** | FastAPI 0.115 + uvicorn, REST + WebSocket, dark-theme HTMX UI |
| **Storage** | SQLModel + aiosqlite default; HMAC-SHA256 plate hashing (no raw plate text ever stored) |
| **Observability** | structlog + Prometheus `/metrics` + opt-in OpenTelemetry (`otel` extras) |
| **Packaging** | `uv` + `pyproject.toml`, `ruff` (lint + format), `pyright`, `pytest` (60+ tests) |

---

## API surface

| Endpoint | Method | What |
|---|---|---|
| `/` | GET | HTMX UI (upload + live webcam) |
| `/health` | GET | k8s probe |
| `/version` | GET | running version |
| `/metrics` | GET | Prometheus text-format metrics |
| `/api/v1/infer` | POST (multipart) | single-image inference |
| `/api/v1/detections` | GET | recent persisted reads (HMAC-hashed) |
| `/ws/stream` | WS | binary JPEG frames in → JSON read events out |

Full OpenAPI at <http://localhost:8000/docs>.

---

## Architecture

```
┌──────────────────────┐       ┌─────────────────────┐
│  Webcam / Upload     │──────►│  FastAPI            │
└──────────────────────┘       │   /infer  /ws/stream │
                               └──────────┬──────────┘
                                          ▼
┌──────────────────────────────────────────────────────┐
│  Pipeline (async, frame-skip + drop-on-full queue)   │
│                                                       │
│   Detector ──► IoUTracker ──► dewarp ──► PlateReader  │
│    fast-alpr      greedy        cv2       fast-plate  │
│      ONNX                                    ONNX     │
│                       │                               │
│                       ▼                               │
│              parse_turkish_plate                      │
│              correct_confusions                       │
│              TemporalVoter (majority)                 │
└──────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────┐       ┌─────────────────────┐
│  SQLite / Postgres   │◄──────│  HMAC-SHA256(plate)  │
│  (only hashes)       │       │  KVKK + GDPR safe    │
└──────────────────────┘       └─────────────────────┘
```

---

## Configuration

All settings live under the `ANPR_` env prefix and can be overridden via `.env`. See [`.env.example`](.env.example) for the full reference.

| Env | Default | Notes |
|---|---|---|
| `ANPR_PLATE_HMAC_PEPPER` | _(generated; warns)_ | **Required in prod.** Generate with `anpr generate-pepper`. Must be ≥ 32 chars. |
| `ANPR_DATABASE_URL` | `sqlite+aiosqlite:///data/anpr.db` | Switch to `postgresql+asyncpg://...` for production. |
| `ANPR_RETENTION_HOURS` | `720` (30d) | Background worker purges rows older than this. |
| `ANPR_DETECT_EVERY_N_FRAMES` | `3` | Detect on every Nth frame; track between. |
| `ANPR_MIN_TRACK_DWELL` | `5` | Frames a track must persist before the voter emits confirmed. |
| `ANPR_DETECTOR_MODEL` | `yolo-v9-t-384-license-plate-end2end` | Pass a path to use a fine-tuned ONNX. |
| `ANPR_OCR_MODEL` | `cct-s-v2-global-model` | fast-plate-ocr model name. |
| `ANPR_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `openvino`. |
| `ANPR_LOG_JSON` | `false` | Set to `true` for structured logs to OTLP / Loki. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | _(unset)_ | When set, installs OTel FastAPI instrumentation (requires `otel` extras). |

---

## Privacy — KVKK & GDPR

License plates are **personal data** under both Türkiye's KVKK (Law 6698, art. 3(d)) and the EU GDPR (Reg. 2016/679 art. 4(1)). This project is privacy-aware by default:

- Plates are **never stored as plaintext**. The database holds only HMAC-SHA256 of the normalized plate string, salted with a per-deployment pepper. Hashes are deterministic for matching but not reversible without the pepper.
- A retention worker purges rows older than `ANPR_RETENTION_HOURS` automatically.
- The web UI (`templates/index.html`) shows the human-readable plate live in the browser but does **not** persist the raw text.

If you deploy this for an organization, you'll likely need to publish a KVKK Aydınlatma Metni / GDPR Article 13 notice. A starter template is in [`KVKK_AYDINLATMA_METNI.md`](KVKK_AYDINLATMA_METNI.md) — **adapt it to your actual processing context, this is not legal advice.**

---

## Development

```bash
uv sync --all-extras --group dev
make lint       # ruff check + format check
make type       # pyright
make test       # pytest (60+ tests)
make serve      # uvicorn --reload
```

Run the end-to-end test against real bundled models (downloads ~30MB on first run):

```bash
ANPR_RUN_REAL=1 uv run pytest tests/integration/test_pipeline_real.py
```

---

## Project structure

```
src/anpr/
├── api/                 FastAPI app, routes, templates, static
│   ├── app.py
│   ├── deps.py          lifespan-loaded singletons
│   ├── middleware.py    request_id + Prometheus middleware
│   └── routes/          health, infer, stream (WS), detections
├── detector/            Detector Protocol + fast-alpr adapter
├── ocr/                 PlateReader Protocol + fast-plate-ocr adapter
├── postprocess/         turkish, confusion, temporal, dewarp
├── storage/             SQLModel Detection + HMAC + repository + retention
├── tracker/             IoU greedy tracker
├── observability.py     Prometheus metrics + optional OTel
├── config.py            pydantic-settings
├── logging.py           structlog
├── cli.py               typer
└── pipeline.py          sync infer_image + async Pipeline
tests/
├── unit/                turkish, confusion, temporal, iou_tracker, hashing (+ hypothesis properties)
└── integration/         pipeline_smoke, api_smoke, metrics, storage, infer_response_shape, pipeline_real
legacy/                  2022 codebase preserved verbatim
```

---

## Migration from the 2022 pipeline

The 2022 code is **not** removed — it sits intact under [`legacy/`](legacy/). What changed:

| 2022 | 2026 |
|---|---|
| Vendored YOLOv5 fork (~7900 LOC in `utils/` + `models/`) | `fast-alpr` 0.4 → YOLOv9-t-384 ONNX, MIT, ~150 LOC of glue |
| EasyOCR 1.4 + PaddleOCR + Tesseract (mixed/inconsistent) | `fast-plate-ocr` 1.1 cct-s-v2 (single source of truth) |
| Flask 1.1.2 (CVE) | FastAPI 0.115 + uvicorn |
| `python main.py` (webcam) and `python app.py` (Flask, parallel + no OCR) | `anpr infer image.jpg`, `anpr serve` |
| No tracking | IoU greedy + temporal majority voting |
| No plate validation | Turkish format parser + confusion correction |
| `requirements.txt` (broken: `tensorrt==0.0.1.dev5`, etc.) | `pyproject.toml` + `uv.lock` |
| No tests, no CI | 60+ tests, GitHub Actions matrix |
| `print()` | structlog + Prometheus + opt-in OTel |
| CSV append, raw plate text | SQLModel + HMAC-SHA256 + retention worker |

If you depended on the old import paths or the `model/best.pt` checkpoint, pin to the pre-merge tag and follow [`legacy/README.md`](legacy/) (yet to be written) for context.

---

## Acknowledgments

This rewrite stands on the work of:

- [`ankandrew/fast-alpr`](https://github.com/ankandrew/fast-alpr) — turn-key MIT-licensed ANPR pipeline
- [`ankandrew/fast-plate-ocr`](https://github.com/ankandrew/fast-plate-ocr) — fast plate-specialized OCR with the `cct-s-v2-global-model`
- [Ultralytics](https://github.com/ultralytics/ultralytics) — original YOLOv5 the 2022 pipeline built on
- [Astral](https://astral.sh) — `uv` and `ruff`

---

## License

[MIT](LICENSE)
