---
title: Watchlist and webhooks
description: Give anpr the plates to look for, and get an HTTP call when one of them is confirmed.
---

The watchlist is the set of plates you want to hear about. When the
[WebSocket stream](/guide/websocket/) or [`anpr watch`](/guide/cli/#anpr-watch-source) confirms a
plate on it, anpr POSTs a JSON message to a URL of yours. Since 0.3.0.

The plates are hashed the moment they come in, with the same pepper as the stored detections, so
the watchlist holds no readable plate either. See [Privacy](/operations/privacy/).

## Set it up

1. Fix the pepper. A watchlist entry is a hash, and a random pepper changes the hash on every
   restart. anpr refuses to add plates while `ANPR_PLATE_HMAC_PEPPER` is unset.

   ```bash
   export ANPR_PLATE_HMAC_PEPPER=$(anpr generate-pepper)
   ```

2. Tell it where to send matches, and give it a secret to sign them with:

   ```bash
   export ANPR_WEBHOOK_URL=https://example.com/hooks/anpr
   export ANPR_WEBHOOK_SECRET=$(anpr generate-api-key)
   ```

3. Add plates:

   ```bash
   anpr watchlist add "34 ABC 1234" --label "delivery van"
   anpr watchlist import plates.txt
   ```

Spaces, hyphens and case don't matter: `34-abc-1234` and `34 ABC 1234` are the same entry. A
running server reloads the watchlist from the database every 30 seconds, so plates added from the
command line reach it without a restart.

The label is sent with each alert so the receiver knows what it is looking at. Keep it to a role or
a vehicle ("delivery van", "gate 2 contractor"); a person's name there would be personal data in
clear text.

## Commands

| Command | What it does |
|---|---|
| `anpr watchlist add PLATE [--label TEXT]` | Adds a plate, or changes the label of one already listed. Adding it again without `--label` keeps the old label. |
| `anpr watchlist import FILE` | Adds every plate in a text file: one per line, optionally `plate,label`. Blank lines and lines starting with `#` are skipped. |
| `anpr watchlist remove PLATE` | Removes a plate. Takes the plate text or its 64-character hash. Exits with code 1 if it wasn't listed. |
| `anpr watchlist list [--json]` | Shows the hashes, labels and when each was added. |

```text
# plates.txt
34 ABC 1234, delivery van
06 AA 42
```

## HTTP endpoints

The same list over HTTP. These endpoints only work when `ANPR_API_KEY` is set, and need the
[key](/guide/http-api/#api-key) on every call; without it they answer `403`. Adding a plate and
listing the watchlist shows its hash, and on an open API anyone could use that to check
`/api/v1/detections` for a plate of their choosing.

```bash
curl -X POST http://localhost:8000/api/v1/watchlist \
  -H "Content-Type: application/json" \
  -d '{"plate": "34 ABC 1234", "label": "delivery van"}'
```

```json
{
  "plate_hmac": "9f2c…e41a",
  "label": "delivery van",
  "created_at": "2026-10-27T14:02:51.120000"
}
```

| Endpoint | Answer |
|---|---|
| `GET /api/v1/watchlist` | Every entry, oldest first. |
| `POST /api/v1/watchlist` | `201` and the new entry. Body: `plate` (required, up to 32 characters) and `label` (optional, up to 200). Posting a listed plate again updates its label, or keeps it if `label` is left out. `409` while the pepper is unset, `422` if `plate` has no letters or digits. |
| `DELETE /api/v1/watchlist/{plate_hmac}` | `204`, or `404` if the hash isn't listed. |

## The webhook call

On a match anpr sends one `POST` with a JSON body:

```json
{
  "event": "watchlist.match",
  "timestamp": "2026-10-27T14:05:09.731204+00:00",
  "source": "stream",
  "plate_hmac": "9f2c…e41a",
  "label": "delivery van",
  "province_code": "34",
  "track_id": 7,
  "frame_idx": 118,
  "confidence": 0.88
}
```

`source` is `stream` for `/ws/stream` and `watch` for `anpr watch`. The plate text isn't in the
body; the label and the hash tell you which entry matched.

The call runs in the background, so a slow receiver doesn't hold up the video. anpr waits up to
`ANPR_WEBHOOK_TIMEOUT_SECONDS` for an answer and treats any `2xx` as delivered. A network error, a
`429` or a `5xx` is tried again after one and then two seconds, three attempts in all; redirects
(`3xx`) and other `4xx` answers count as failures and are not retried. Redirects aren't followed,
so point `ANPR_WEBHOOK_URL` at the final address, `https://` included. If every attempt fails, the
cooldown is lifted and the next sighting of the plate tries again. The same plate is announced at most once every
`ANPR_WEBHOOK_COOLDOWN_SECONDS` (60 by default), because a car that stops and moves on often comes
back as a new track.

## Checking the signature

With `ANPR_WEBHOOK_SECRET` set, every call carries two headers:

```text
X-ANPR-Timestamp: 1793109909
X-ANPR-Signature: sha256=5d0c…
```

The signature is the hex HMAC-SHA256 of the timestamp, a dot and the raw body, keyed with the
secret. Check it before trusting the message, and reject old timestamps so a recorded call can't be
replayed:

```python
import hashlib
import hmac
import time


def verify(secret: str, headers: dict[str, str], body: bytes, max_age: int = 300) -> bool:
    timestamp = headers["X-ANPR-Timestamp"]
    if abs(time.time() - int(timestamp)) > max_age:
        return False
    mac = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
    return hmac.compare_digest(f"sha256={mac.hexdigest()}", headers["X-ANPR-Signature"])
```

Use the body exactly as it arrived; parsing and re-encoding the JSON changes the bytes.

## Where it applies

- **`/ws/stream`** checks every confirmed plate.
- **`anpr watch`** checks the watchlist when `ANPR_WEBHOOK_URL` is set or `--store` is on, since
  those are the cases where it opens the database. Matches are marked `watchlist` in the output,
  and the JSON lines get a `watchlist` field.
- **`POST /api/v1/infer`** doesn't: a single image has no confirmed reads.

The `anpr_watchlist_matches_total` and `anpr_webhook_deliveries_total` metrics count matches and
delivery outcomes; see [Monitoring](/operations/monitoring/).
