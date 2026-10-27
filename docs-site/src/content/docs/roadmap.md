---
title: Roadmap
description: What anpr-pipeline can't do yet, and what is planned.
---

## Known limits

These are honest gaps in the current version, 0.3.x.

- **One shared API key.** `ANPR_API_KEY` is a single key for every client; there are no users,
  roles or per-client keys.
- **Seven countries.** Readings are checked against Türkiye, Germany, France, Italy, Spain,
  Belgium and Portugal ([Plate formats](/guide/plate-formats/)). Other plates come back with
  `parsed: null`, and German plates aren't split into district and the rest.
- **No perspective correction.** Plates seen at a steep angle aren't straightened before OCR; the
  stream only scales the crop to a fixed height.
- **Two settings do nothing yet.** `ANPR_SR_MIN_PLATE_WIDTH` (super-resolution for small plates) and
  `ANPR_OCR_FALLBACK_THRESHOLD` (a second OCR engine for unsure reads) are placeholders.

## Planned

In rough order:

1. **More countries**: the Netherlands, Poland and Austria next, and the German district list so
   `MAB 1234` can be shown as `M-AB 1234`.

Ideas and pull requests are welcome on
[GitHub](https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/issues).
