---
title: Roadmap
description: What anpr-pipeline can't do yet, and what is planned.
---

## Known limits

These are honest gaps in the current version, 0.2.x.

- **One shared API key.** `ANPR_API_KEY` is a single key for every client; there are no users,
  roles or per-client keys.
- **Turkish plates only.** The detector finds plates from many countries, but the reading is only
  checked and corrected against the Turkish format. Other plates come back with `parsed: null`.
- **No video or camera input in the CLI.** `anpr infer` reads one image. For video, use the
  [WebSocket stream](/guide/websocket/) with a small client.
- **No perspective correction.** Plates seen at a steep angle aren't straightened before OCR; the
  stream only scales the crop to a fixed height.
- **Two settings do nothing yet.** `ANPR_SR_MIN_PLATE_WIDTH` (super-resolution for small plates) and
  `ANPR_OCR_FALLBACK_THRESHOLD` (a second OCR engine for unsure reads) are placeholders.

## Planned

In rough order:

1. **Video files and RTSP cameras** from the command line, as in `anpr watch rtsp://camera/stream`,
   with the same tracking, voting and storage as the WebSocket stream.
2. **A watchlist with webhooks.** Give it the plates to look for; it keeps only their hashes, and
   calls a URL when one of them is confirmed.
3. **Plate formats for other countries**, starting with Germany and the rest of the EU, chosen with a
   setting.

Ideas and pull requests are welcome on
[GitHub](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/issues).
