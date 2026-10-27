---
title: Privacy, KVKK and GDPR
description: Why anpr-pipeline stores a keyed hash instead of the plate, how matching still works, and what you still have to do as the operator.
---

A number plate identifies a vehicle, and through it a person. It is personal data under Türkiye's
KVKK (Law 6698, article 3(d)) and the EU's GDPR (Regulation 2016/679, article 4(1)). anpr-pipeline
is built so that the database never holds a plate in readable form.

## What is stored

When the [WebSocket stream](/guide/websocket/) confirms a plate, one row is written:

| Column | Example |
|---|---|
| `plate_hmac` | `9f2c…e41a`, the HMAC-SHA256 of the plate |
| `province_code` | `34` |
| `confidence` | `0.88` |
| bounding box | `412, 388, 618, 436` |
| `track_id`, `timestamp`, `confirmed` | |

The plate text, the image and the frame are not stored. `POST /api/v1/infer` and the CLI store
nothing at all.

## How the hash works

The plate is put in one form first (spaces removed, upper case), so `34 abc 1234` and `34ABC1234`
give the same hash. It is then hashed with HMAC-SHA256, keyed with `ANPR_PLATE_HMAC_PEPPER`.

- **Matching still works.** The same plate with the same pepper always gives the same hash, so you
  can count visits or check a plate you already know.
- **Guessing doesn't.** Turkish plates follow a known pattern, so a plain hash could be reversed by
  trying every plate. Without the pepper, an attacker can't compute the hashes to compare against.
- **Keep the pepper safe and stable.** Store it like a password, outside the database. If you
  change it, old and new hashes of the same plate no longer match.

To check a known plate against the stored rows:

```python
from anpr.storage.hashing import hash_plate

wanted = hash_plate("34 ABC 1234", pepper)
# compare `wanted` with plate_hmac from GET /api/v1/detections
```

## Retention

A background task deletes rows older than `ANPR_RETENTION_HOURS`, 720 hours (30 days) by default.
It runs when the service starts and then every hour. Set the period to what your purpose actually
needs.

## What is still up to you

The software keeps the stored data small. It doesn't make a deployment lawful on its own. As the
data controller you still need, among other things:

- a lawful basis and a defined purpose for recording plates,
- a privacy notice for the people being recorded. The repository has a starting point in Turkish,
  [`KVKK_AYDINLATMA_METNI.md`](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/blob/main/KVKK_AYDINLATMA_METNI.md).
  Adapt it to your case; it is not legal advice,
- access control on the service and the database: set `ANPR_API_KEY` and serve over HTTPS,
- a narrowed `ANPR_CORS_ORIGINS`.

The web page shows the readable plate in the browser while you watch, but doesn't send it anywhere
to be kept.
