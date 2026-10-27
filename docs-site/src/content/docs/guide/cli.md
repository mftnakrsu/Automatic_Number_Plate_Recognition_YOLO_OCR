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

## `anpr serve`

Starts the HTTP and WebSocket service with uvicorn.

| Option | Environment variable | Default | Meaning |
|---|---|---|---|
| `--host` | `ANPR_API_HOST` | `0.0.0.0` | Address to listen on. |
| `--port` | `ANPR_API_PORT` | `8000` | Port to listen on. |
| `--reload` | | off | Restart when the code changes. For development only. |

## `anpr generate-pepper`

Prints a new random 64-character hex string to use as `ANPR_PLATE_HMAC_PEPPER`.

## `anpr dump-config`

Prints the settings the service would run with, after reading the environment and `.env`, as
JSON. The pepper is shown as `***`.

## `anpr version`

Prints the installed version.
