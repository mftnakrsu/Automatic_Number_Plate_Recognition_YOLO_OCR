# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/releases/tag/v0.2.0
