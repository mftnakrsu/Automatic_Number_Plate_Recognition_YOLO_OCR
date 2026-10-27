---
title: Command line
description: Every anpr command and option.
---

The `anpr` command comes with the package. `anpr --help` lists the commands, and
`anpr <command> --help` shows the options of one.

## Global options

These go before the command, as in `anpr --log-json serve`.

| Option | Environment variable | Default | Meaning |
|---|---|---|---|
| `--log-level` | `ANPR_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `--log-json` | `ANPR_LOG_JSON` | off | Write logs as JSON lines instead of coloured text. |

## `anpr infer IMAGE`

Finds and reads the plates in one image file. Prints a table, or JSON with `--json`. See
[Your first plate](/start/first-plate/) for what the columns mean.

| Option | Meaning |
|---|---|
| `--json` | Print the result as a JSON array instead of a table. |

It exits with code 2 if the file doesn't exist. Nothing is stored.

## `anpr watch SOURCE`

Reads plates from video: a file, a network stream or a camera. It runs the same frame skipping,
tracking and vote as the [WebSocket stream](/guide/websocket/) and prints each plate once, when it
is confirmed. Since 0.3.0.

```bash
anpr watch traffic.mp4
anpr watch rtsp://user:pass@192.168.1.20:554/stream1
anpr watch 0                       # the first local camera
anpr watch traffic.mp4 --json      # one JSON object per line
```

```text
frame     42  track    3  34 ABC 1234  İstanbul  det=0.91
frame    118  track    7  06 AA 42  Ankara  det=0.88
240 frames read, 2 plates confirmed.
```

| Option | Meaning |
|---|---|
| `--json` | Print one JSON object per confirmed plate: `frame_idx`, `track_id`, `plate`, `province_code`, `province_name`, `confidence`, `bbox`. |
| `--store` | Also save each confirmed plate to `ANPR_DATABASE_URL`, hashed like the stream does. Set `ANPR_PLATE_HMAC_PEPPER` first. |
| `--max-frames N` | Stop after N frames. |
| `--reconnect N` | For streams and cameras, how many times to reopen the source after it drops (default 5, two seconds apart). A file just ends. |

`ANPR_DETECT_EVERY_N_FRAMES` and `ANPR_MIN_TRACK_DWELL` apply as on the stream. The summary line
goes to standard error, so `--json` output can be piped as it is. `Ctrl+C` stops it. The command
exits with code 2 if the source can't be opened.

## `anpr serve`

Starts the HTTP and WebSocket service with uvicorn.

| Option | Environment variable | Default | Meaning |
|---|---|---|---|
| `--host` | `ANPR_API_HOST` | `0.0.0.0` | Address to listen on. |
| `--port` | `ANPR_API_PORT` | `8000` | Port to listen on. |
| `--reload` | | off | Restart when the code changes. For development only. |

## `anpr generate-pepper`

Prints a new random 64-character hex string to use as `ANPR_PLATE_HMAC_PEPPER`.

## `anpr generate-api-key`

Prints a new random key to use as `ANPR_API_KEY`. Since 0.2.5.

## `anpr dump-config`

Prints the settings the service would run with, after reading the environment and `.env`, as
JSON. The pepper and the API key are shown as `***`.

## `anpr version`

Prints the installed version.
