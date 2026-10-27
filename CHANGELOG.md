# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added

- `anpr watch SOURCE` reads plates from a video file, a network stream (RTSP, HTTP)
  or a camera index, with the same frame skipping, tracking and temporal vote as
  `/ws/stream`. It prints each track's plate once when confirmed (`--json` for JSON
  lines), can store hashed reads with `--store`, and reconnects live sources after a
  drop (`--reconnect`, default 5 attempts).
- Watchlist with webhooks. `anpr watchlist add|import|remove|list` and
  `GET/POST/DELETE /api/v1/watchlist` manage the plates to look for; each is
  stored only as its HMAC, in a new `watchlist` table. When `/ws/stream` or
  `anpr watch` confirms a listed plate, the JSON event (hash, label, province,
  track, confidence; never the plate text) is POSTed to `ANPR_WEBHOOK_URL` in
  the background, signed with `ANPR_WEBHOOK_SECRET` as
  `X-ANPR-Signature: sha256=HMAC("<timestamp>.<body>")`. Network errors, 429
  and 5xx are retried twice; the same plate is announced at most once per
  `ANPR_WEBHOOK_COOLDOWN_SECONDS`. Adding plates is refused while the pepper is
  unset, because a generated pepper changes the hash on every restart. New
  metrics: `anpr_watchlist_matches_total`, `anpr_webhook_deliveries_total`.

### Changed

- Plate hashes ignore every character that isn't a letter or digit, so
  `34-ABC-1234` and `34 ABC 1234` hash the same. Stored hashes don't change:
  the plates that were hashed so far only ever had spaces.

## [0.2.5] — 2026-10-27

### Added

- Optional API key. With `ANPR_API_KEY` set, `/api/v1/*` needs it as `X-API-Key` or
  `Authorization: Bearer`, and `/ws/stream` as a header or as the `anpr.key.<key>`
  WebSocket subprotocol (browsers can't set WebSocket headers, and a query string
  would land in access logs). Keys are compared in constant time. The web page asks
  for the key when the server needs one. New command: `anpr generate-api-key`.
- Documentation site at <https://anpr.mefby.com> in English and Turkish, built with
  Astro Starlight from `docs-site/` and published to GitHub Pages.
- Python 3.14 support (`requires-python` now `<3.15`, CI matrix includes 3.14).
  The optional `paddle` extra is skipped on 3.14 because `paddlepaddle` has no
  cp314 wheels yet; nothing in `src/` imports it.

### Changed

- Dev dependency `httpx` → `httpx2`: Starlette 1.x deprecates plain `httpx`
  for `TestClient`. On 3.13 it was only arriving transitively via the paddle
  extra, which hid the warning.
- CI: GitHub Actions bumped to their current majors (checkout v7, setup-uv v10.3.2 — it no longer publishes a floating major tag —
  codecov v7, build-push v7, setup-buildx v4, login v4, metadata v6, CodeQL v4,
  upload-artifact v7, download-artifact v8). Most of these are the Node 20 →
  Node 24 runtime move; workflow inputs are unchanged.
- Dependabot now groups GitHub Actions updates into a single PR and no longer
  produces `chore(deps)(deps)` titles.

## [0.2.4] — 2026-07-31

### Security

- **Unbounded decoded-image size on the unauthenticated inference APIs**
  (GHSA-r5v2-w3w2-4c8w, CWE-400, CVSS 3.1 7.5 High). `POST /api/v1/infer` and
  `WS /ws/stream` decoded attacker-supplied images before bounding the decoded
  raster; only the *compressed* payload was capped, via `max_upload_mb`. A
  highly-compressible PNG under 1 MB decoded to hundreds of megabytes, and a
  single unauthenticated WebSocket message was enough to OOM-kill a container
  running on a 512 MB budget. Both paths now check the declared dimensions
  from the image header — before any pixel data is materialised — against the
  new `ANPR_MAX_IMAGE_PIXELS` budget, and `/ws/stream` additionally caps the
  compressed frame at `ANPR_MAX_WS_FRAME_BYTES` both in the handler and at
  uvicorn's `ws_max_size`. Reported by [@EQSTLab](https://github.com/EQSTLab).

### Fixed

- `/api/v1/infer` no longer echoes the raw decoder exception back to the
  caller; the detail is logged server-side and the response is generic.
- A failure in the `/ws/stream` producer other than a client disconnect left
  the consumer awaiting a queue nobody would fill, leaking the connection and
  its `ACTIVE_STREAMS` slot. The sentinel is now pushed from a `finally`.

## [0.2.3] — 2026-05-10

### Fixed

- `anpr.__version__` was returning `0.0.0+unknown` for installed users — the
  package queried `importlib.metadata.version("anpr")` while the distribution
  name on PyPI is `anpr-pipeline`. Now correctly reports the installed
  version. ([#25](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/pull/25))

## [0.2.2] — 2026-05-10

### Fixed

- Source distribution size: hatchling's default file sweep was including the
  150MB `legacy/` tree, blowing past PyPI's 100MB project limit. An explicit
  `[tool.hatch.build.targets.sdist].include` allow-list now produces a 36 KB
  sdist. ([#24](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/pull/24))

## [0.2.1] — 2026-05-10 [YANKED]

First PyPI publish attempt. The sdist exceeded PyPI's size limit; only the
wheel uploaded before the sdist upload failed. Yanked — use 0.2.2+.

### Added

- Distribution renamed `anpr` → `anpr-pipeline` (the import name remains
  `anpr`).
- GitHub Actions publish workflow via OIDC trusted publishing.
- README PyPI badge + `pip install anpr-pipeline` quickstart. ([#19](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/pull/19))

## [0.2.0] — 2026-05-10

Initial release of the 2026 modernization. Full rewrite of the 2022 YOLOv5 +
EasyOCR + Flask 1.1 pipeline. The original code is preserved verbatim under
[`legacy/`](legacy/) and tagged [`v0.1.0-legacy`](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/releases/tag/v0.1.0-legacy).

### Added

- `fast-alpr` 0.4 / YOLOv9-t-384 ONNX plate detector (MIT, 65+ countries).
- `fast-plate-ocr` 1.1 / `cct-s-v2-global-model` OCR (MIT, sub-millisecond GPU
  with per-character confidences).
- Dependency-free IoU greedy tracker (`Tracker` Protocol allows swap to
  ByteTrack / BoT-SORT).
- Position-aware OCR confusion-pair correction (0/O, 1/I/L, 5/S, 8/B, 2/Z,
  6/G, 0/D/Q) keyed on the Turkish plate format.
- Per-track temporal majority voting (top-K confidence-weighted, per-character
  position).
- Turkish-plate parser covering all 81 province codes.
- FastAPI 0.115 + uvicorn HTTP/WebSocket API with async streaming pipeline,
  bounded queue + drop-on-full backpressure.
- SQLModel storage with HMAC-SHA256 plate hashing (KVKK + GDPR aware; raw
  plate text is never persisted). Background retention worker.
- `structlog` + Prometheus `/metrics` + opt-in OpenTelemetry instrumentation.
- Multi-stage Dockerfile + `docker-compose.yml`.
- GitHub Actions CI: matrix on Python 3.11 / 3.12 / 3.13 + ruff + pyright +
  pytest + Docker smoke.
- 60+ tests (unit + integration + Hypothesis property tests).
- `pydantic-settings`-driven config (`.env` + `ANPR_*` env vars).
- typer CLI: `anpr serve | infer | dump-config | generate-pepper | version`.
- Modernized README + KVKK Aydınlatma Metni şablonu.

### Removed

- Vendored YOLOv5 fork (~7900 LOC under `utils/` + `models/`).
- EasyOCR 1.4 + PaddleOCR + Tesseract triple-OCR layer.
- Flask 1.1.2 + Werkzeug stack (CVE-laden).
- `helper/params.py` ≡ `utils/params.py` byte-identical duplicates.
- Hardcoded `/home/mef/Documents/plate_detection_project/best.pt`.
- CSV plate-text persistence (replaced by hashed SQLite via SQLModel).

### Migration

- Entry points changed: `python main.py` → `anpr serve`; `python app.py`
  removed in favor of the FastAPI app.
- Pinning to the 2022 pipeline: `git checkout v0.1.0-legacy`.

### Merged in this release

- [#16](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/pull/16) — F0–F7 (scaffold + pipeline + API + storage + tests)
- [#17](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/pull/17) — F8 (Docker + CI matrix)
- [#18](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/pull/18) — F9 (README + KVKK template)

[Unreleased]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/releases/tag/v0.2.0
