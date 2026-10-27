---
title: WebSocket stream
description: Send video frames to /ws/stream and receive plate reads, with tracking and confirmation across frames.
---

`/ws/stream` is for video: a webcam, a camera feed or frames from a file. You send frames, and the
service sends back what it read, following each plate from frame to frame and confirming it once it
has been read often enough.

## Protocol

1. Open a WebSocket to `ws://localhost:8000/ws/stream`.
2. Send each frame as one **binary** message holding an encoded image, usually JPEG.
3. Receive **text** messages, each a JSON object for one plate on one processed frame.

Each connection has its own tracker and vote, so plates from two streams never mix.

## What comes back

```json
{
  "frame_idx": 42,
  "track_id": 3,
  "bbox": [412, 388, 618, 436],
  "detection_confidence": 0.91,
  "raw_text": "34ABC1234",
  "normalized_text": "34ABC1234",
  "parsed": {
    "canonical": "34 ABC 1234",
    "province_code": "34",
    "province_name": "İstanbul"
  },
  "confirmed": "34 ABC 1234"
}
```

| Field | Meaning |
|---|---|
| `frame_idx` | Number of the frame on this connection, counting from 0. |
| `track_id` | Stays the same while the same plate stays in view. |
| `confirmed` | `null` until the vote has enough reads for this track, then the agreed plate. |

The other fields are the same as in [`POST /api/v1/infer`](/guide/http-api/).

## How a plate gets confirmed

- **Frame skipping.** Only every Nth frame is processed (`ANPR_DETECT_EVERY_N_FRAMES`, default 3).
  The other frames are decoded but never reach the models, which keeps a live feed cheap.
- **Tracking.** On each processed frame, every box is matched to the track whose last box overlaps
  it most, if they overlap by at least 30% (IoU 0.3). A box that matches nothing starts a new
  track, and a track that goes unseen for 30 processed frames is dropped.
- **Voting.** Every valid reading of a track goes into a history of the last 8. Once the track has
  been read on `ANPR_MIN_TRACK_DWELL` processed frames (default 5), the three most confident reads
  vote character by character, each vote weighted by its confidence. The winner is the confirmed
  plate.
- **Storage.** The first time a track is confirmed, one row is written: the hash of the plate, its
  province code, the confidence and the box. Later messages for the same track aren't stored again.

## Back-pressure and limits

The service keeps at most four frames waiting. If you send faster than it can read, new frames are
dropped rather than queued, so the delay stays short.

A frame larger than `ANPR_MAX_WS_FRAME_BYTES` (4 MB by default) or with more than
`ANPR_MAX_IMAGE_PIXELS` pixels closes the connection with code `1009`. A frame that can't be
decoded is skipped and the stream carries on.

## Example client

A Python client that streams a video file, using `opencv-python` and `websockets`:

```python
import asyncio, json
import cv2, websockets

async def main() -> None:
    cap = cv2.VideoCapture("traffic.mp4")
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:

        async def receive() -> None:
            async for message in ws:
                event = json.loads(message)
                if event["confirmed"]:
                    print(event["track_id"], event["confirmed"])

        listener = asyncio.create_task(receive())
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, jpeg = cv2.imencode(".jpg", frame)
            await ws.send(jpeg.tobytes())
            await asyncio.sleep(1 / 25)  # roughly the video's frame rate
        await asyncio.sleep(2)  # let the last reads arrive
        listener.cancel()

asyncio.run(main())
```

The web page at `/` does the same with your webcam in the browser.
