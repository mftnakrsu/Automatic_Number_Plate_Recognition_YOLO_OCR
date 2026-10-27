---
title: HTTP API
description: The REST endpoints of the anpr service, with request and response examples.
---

`anpr serve` starts a FastAPI app. The examples below assume it runs on `localhost:8000`. FastAPI
also serves an interactive reference at `/docs` and the OpenAPI schema at `/openapi.json`.

## API key

Since 0.2.5 the service can ask for a key. Set `ANPR_API_KEY` (make one with
`anpr generate-api-key`) and every `/api/v1/` endpoint needs it, sent either way:

```bash
curl -H "X-API-Key: $ANPR_API_KEY" -F file=@car.jpg http://localhost:8000/api/v1/infer
curl -H "Authorization: Bearer $ANPR_API_KEY" http://localhost:8000/api/v1/detections
```

A missing or wrong key gets `401` with `WWW-Authenticate: Bearer`. `/health`, `/version`,
`/metrics` and the web page stay open. Without `ANPR_API_KEY` the API is open to anyone who can
reach it, so then keep it on a private network or behind a proxy that controls access; see
[Deployment](/operations/deployment/).

## `POST /api/v1/infer`

Reads the plates in one image. Send the image as a multipart form field named `file`. Any format
Pillow can open works: JPEG, PNG, WebP, BMP and so on.

```bash
curl -F file=@car.jpg http://localhost:8000/api/v1/infer
```

The response is a list with one entry per plate found, and an empty list if there were none:

```json
[
  {
    "bbox": { "x1": 412, "y1": 388, "x2": 618, "y2": 436 },
    "detection_confidence": 0.91,
    "raw_text": "34ABC1Z34",
    "normalized_text": "34ABC1234",
    "parsed": {
      "province_code": "34",
      "province_name": "İstanbul",
      "letters": "ABC",
      "digits": "1234",
      "canonical": "34 ABC 1234"
    }
  }
]
```

| Field | Meaning |
|---|---|
| `bbox` | The plate's box in pixels. |
| `detection_confidence` | How sure the detector is, from 0 to 1. |
| `raw_text` | What the OCR model read. `null` if it read nothing. |
| `normalized_text` | `raw_text` without spaces, in upper case, after fixing look-alike characters. |
| `parsed` | The reading split into province, letters and digits. `null` if it isn't a valid Turkish plate. |

This endpoint doesn't store anything. Only the [WebSocket stream](/guide/websocket/) stores
confirmed plates.

### Errors

| Status | When |
|---|---|
| `400` | The upload is empty (`empty upload`) or can't be decoded as an image (`cannot decode image`). |
| `413` | The file is bigger than `ANPR_MAX_UPLOAD_MB` (`upload too large`), or its decoded size is over `ANPR_MAX_IMAGE_PIXELS` (`image too large`). |
| `401` | `ANPR_API_KEY` is set and the request didn't send it. |
| `422` | The request has no `file` field. |

The pixel limit is checked from the image header before the image is decoded, so a small file that
claims to be enormous is turned away without using the memory.

## `GET /api/v1/detections`

Lists the latest stored plates, newest first.

```bash
curl "http://localhost:8000/api/v1/detections?limit=20&confirmed_only=true"
```

| Query parameter | Default | Meaning |
|---|---|---|
| `limit` | `100` | How many rows to return, from 1 to 1000. |
| `confirmed_only` | `false` | Only return confirmed reads. |

```json
[
  {
    "id": 17,
    "timestamp": "2026-10-27T09:41:12.503000",
    "track_id": 3,
    "plate_hmac": "9f2c…e41a",
    "province_code": "34",
    "confidence": 0.88,
    "bbox": [412, 388, 618, 436],
    "confirmed": true
  }
]
```

There is no plate text in the response, because the database doesn't have it. To check whether a
known plate was seen, compute its hash with the same pepper and compare it with `plate_hmac`; see
[Privacy](/operations/privacy/).

## Service endpoints

| Endpoint | Response |
|---|---|
| `GET /health` | `{"status": "ok"}` once the app is up. Use it as a liveness probe. |
| `GET /version` | `{"version": "0.2.5"}` |
| `GET /metrics` | Prometheus metrics in text format. See [Monitoring](/operations/monitoring/). |
| `GET /` | The web page with image upload and webcam streaming. |

Every response carries an `X-Request-ID` header. If the request sent one, it is passed through;
otherwise the service makes one up. The same ID appears in the logs.
