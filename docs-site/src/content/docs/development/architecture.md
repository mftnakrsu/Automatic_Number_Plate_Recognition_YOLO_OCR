---
title: Architecture
description: How a frame becomes a stored hash, and which module does each step.
---

```text
One image (POST /api/v1/infer, anpr infer)

  decode ─▶ detect ─▶ crop ─▶ OCR ─▶ format check + look-alike fixes ─▶ JSON

A stream (/ws/stream)

  decode ─▶ every Nth frame ─▶ detect ─▶ track ─▶ crop + resize ─▶ OCR
         ─▶ format check + look-alike fixes ─▶ vote per track ─▶ WebSocket event
                                                     │
                                                     └─ first confirmation ─▶ HMAC ─▶ database
```

A single image goes straight from detection to the format check. A stream adds frame skipping,
tracking and the vote. It and `anpr watch --store` are the only paths that write to the database.

## Modules (`src/anpr/`)

| Path | Responsibility |
|---|---|
| `detector/` | The `Detector` protocol and its fast-alpr implementation. Returns boxes with confidences. |
| `ocr/` | The `PlateReader` protocol and its fast-plate-ocr implementation. Returns text and per-character confidences. |
| `tracker/iou.py` | Greedy IoU tracker. Stable track IDs, expiry after 30 unseen frames. |
| `postprocess/turkish.py` | The 81 province codes and the plate pattern. Turns `34ABC1234` into province, letters and digits. |
| `postprocess/formats.py` | One `PlateFormat` per country (`ANPR_PLATE_FORMATS`) and `parse_plate`, which tries exact reads in every format before corrected ones. |
| `postprocess/confusion.py` | When a reading doesn't fit, tries the three possible letter-block lengths and swaps look-alike characters to fit each. |
| `postprocess/temporal.py` | Confidence-weighted, per-character majority vote over a track's best reads. |
| `postprocess/dewarp.py` | On the stream, scales the crop to a fixed height. A placeholder for a real perspective correction. |
| `pipeline.py` | `infer_image` for one image, and the async `Pipeline` for streams. |
| `imaging.py` | Decodes uploads and frames, and rejects them by declared size before decoding. |
| `storage/` | The `Detection` table (SQLModel), HMAC hashing, the repository and the retention task. |
| `api/` | The FastAPI app, its routes, the web page and the middleware for request IDs and metrics. |
| `observability.py` | Prometheus metrics and optional OpenTelemetry. |
| `config.py` | Settings from `ANPR_` environment variables, with pydantic-settings. |
| `cli.py` | The `anpr` command, with Typer. |

## Design choices

**Protocols at the model boundary.** The pipeline only knows `Detector` and `PlateReader`. Another
detector or OCR engine can be added by writing one class, and the tests use fakes so that most of
them run in a fraction of a second without any model.

**Format rules after OCR, not inside it.** The OCR model is general. Knowing that a Turkish plate is
`NN L{1-3} N{2-4}` lets the parser fix the characters a general model gets wrong, such as an `O` in
a digit position. Other countries would need their own parser; see the [Roadmap](/roadmap/).

**Hash at the edge of storage.** The plate text exists in memory and in the response, and is hashed
right where a row is written, so no code path can store it by mistake.

**Bounded everything on the stream.** A queue of four frames, frames dropped when it is full, and
size checks before decoding: a slow model or a hostile client can't make the service use unbounded
memory.

## The 2022 version

The first version, built on a YOLOv5 fork, EasyOCR and Flask, is kept unchanged in `legacy/` and
tagged `v0.1.0-legacy`. It isn't maintained.
