---
title: Contributing
description: Set up the project, run the same checks as CI, and open a pull request.
---

Bug reports, fixes, features and documentation are all welcome. For anything bigger than a small
fix, open an issue first so the scope is agreed before you write the code.

## Set up

You need [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer.

```bash
git clone https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR.git
cd Automatic_Number_Plate_Recognition_YOLO_OCR
uv sync --all-extras --group dev
uv run pre-commit install   # optional: run the checks on every commit
```

## The checks

CI runs these on Python 3.11, 3.12, 3.13 and 3.14, then builds the Docker image and calls its
`/health` endpoint. Run them before you push:

```bash
make lint   # ruff check and ruff format --check
make type   # pyright
make test   # pytest
```

The default test suite uses fake models and needs no network. The end-to-end test with the real
models downloads them on its first run (about 30 MB):

```bash
ANPR_RUN_REAL=1 uv run pytest tests/integration/test_pipeline_real.py
```

## Conventions

- Branch names start with `feat/`, `fix/`, `docs/`, `chore/`, `ci/`, `test/` or `refactor/`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`,
  `fix:`, `docs:` and so on.
- One logical change per pull request.
- Code is formatted with `ruff format` at 100 columns and type-checked with pyright. Public functions
  have full type hints.
- Never store plate text. Anything that persists a plate goes through `storage.hashing.hash_plate`.

## The documentation

This site lives in `docs-site/` and is built with [Astro Starlight](https://starlight.astro.build).
Pages are Markdown files in `docs-site/src/content/docs/`, with Turkish translations under `tr/`.

```bash
cd docs-site
npm install
npm run dev   # http://localhost:4321
```

Every push to `main` that touches `docs-site/` publishes the site.
