---
title: Plate formats
description: Which countries' plates anpr checks and corrects, and how to choose them.
---

The detector finds plates from many countries, but a reading is only accepted when it fits a plate
layout you have turned on. `ANPR_PLATE_FORMATS` lists them, in order of preference. The default is
`tr`, Türkiye only. Since 0.3.0.

```bash
export ANPR_PLATE_FORMATS=de,fr,tr
```

## Countries

| Code | Country | Layout | Canonical form | Region |
|---|---|---|---|---|
| `tr` | Türkiye | 2-digit province, 1–3 letters, 2–4 digits | `34 ABC 1234` | province code and name |
| `de` | Germany | 2–5 letters, 1–4 digits without a leading zero, at most 8 together, optional `E` or `H` | `MAB 1234` | none |
| `fr` | France (SIV, 2009 on) | 2 letters, 3 digits, 2 letters; no I, O, U; no `SS`, no `000` | `AB-123-CD` | none |
| `it` | Italy (1994 on) | 2 letters, 3 digits, 2 letters; no I, O, Q, U | `AB 123 CD` | none |
| `es` | Spain (2000 on) | 4 digits, 3 consonants; no Ñ, no Q | `1234 BCD` | none |
| `be` | Belgium (2010 on) | 1 digit, 3 letters, 3 digits | `1-ABC-123` | none |
| `pt` | Portugal | `AA-00-AA`, `00-AA-00`, `00-00-AA` or `AA-00-00` | `AA-00-AA` | none |

German plates come out as letters and digits, `MAB 1234`, not `M-AB 1234`. The district code is one
to three letters, so the split between the district and the rest can't be told from the characters
alone; that needs the list of district codes, which isn't built in yet. Matching isn't affected:
hashes ignore spaces and hyphens, so `M-AB 1234` added to the [watchlist](/guide/watchlist/) still
matches.

Only Turkish plates have a region. For the others, `province_code` and `province_name` are `null`
in every response.

## How a reading is checked

1. The OCR text is stripped of spaces and put in upper case.
2. Each format in `ANPR_PLATE_FORMATS` tries it as it stands, in order. The first that accepts it
   wins.
3. If none does, each format tries again with look-alike characters swapped for what its layout
   expects in each position: `0` and `O`, `1` and `I`, `8` and `B`, `5` and `S`, `2` and `Z`, `6`
   and `G`. A Spanish `I2B4BCD` becomes `1284BCD`.
4. If that fails too, `parsed` is `null` and the reading isn't confirmed on the stream.

An exact reading in any format beats a corrected one. When two formats accept the same text as
it stands, the one listed first wins: `AB123CD` is valid in both France and Italy, so with
`ANPR_PLATE_FORMATS=it,fr` it is read as Italian.

Turn on only the countries you expect. Every extra format is another way for a misread to look
valid, and the short Portuguese layouts in particular accept many six-character strings.

## In the responses

Every `parsed` object has a `country` field with the code from the table:

```json
"parsed": {
  "country": "de",
  "province_code": null,
  "province_name": null,
  "letters": "MAB",
  "digits": "1234",
  "canonical": "MAB 1234"
}
```

`anpr infer` and `anpr watch` show the province for Turkish plates and the country code for the
others. Stored detections have a `country` column, which `GET /api/v1/detections` returns; the
service adds it to a database made by an older version on start-up, and rows from before then
keep it `null`. The `anpr_confirmed_plates_total` metric has a `country` label.
