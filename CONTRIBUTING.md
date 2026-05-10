# Contributing

Contributions are welcome — bug reports, fixes, features, docs improvements.

## Quick start

```bash
git clone https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR.git
cd Automatic_Number_Plate_Recognition_YOLO_OCR
uv sync --all-extras --group dev
make test
```

## Workflow

1. Open an issue first for non-trivial changes so we agree on scope.
2. Fork the repo, then branch off `main`. Branch naming:
   - `feat/...` for features
   - `fix/...` for bugs
   - `docs/...` for documentation
   - `chore/...` / `ci/...` / `test/...` / `refactor/...` as appropriate
3. Keep PRs focused — one logical change per PR.
4. Use [conventional-commits](https://www.conventionalcommits.org/) prefixes
   in your commits: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `ci:`,
   `refactor:`.
5. Make sure CI passes (see below).

## Pre-merge checks

CI runs `ruff`, `pyright`, `pytest` on Python 3.11 / 3.12 / 3.13 + a Docker
smoke test on every PR. To run the same checks locally:

```bash
make lint       # ruff check + ruff format --check
make type       # pyright
make test       # pytest (60+ tests)
```

`pre-commit` hooks are configured — opt in with:

```bash
uv run pre-commit install
```

## Testing

- Unit tests live in `tests/unit/`.
- Integration tests in `tests/integration/`.
- Tests using the real `fast-alpr` / `fast-plate-ocr` models are gated behind
  `ANPR_RUN_REAL=1` so the default suite is fast and offline.

## Code style

- Python 3.11+ (3.13 preferred for local dev).
- Formatted by `ruff format` (Black-compatible, 100-col line length).
- Type-checked by `pyright` (`standard` mode).
- Public functions need full type hints.
- New modules should use `from __future__ import annotations`.

## Privacy by design (KVKK / GDPR)

This is an ANPR project — license plates are personal data under KVKK
(Law 6698) and the EU GDPR. New features must respect the existing model:

- Don't add code paths that persist **raw plate text** to durable storage.
- New API responses returning plate text should be auth-gated or documented
  as in-memory only.
- Persistence operations must continue to use `hash_plate(plate, pepper)`
  from `src/anpr/storage/hashing.py`.

## Reporting security issues

**Don't** open public issues for security vulnerabilities — see
[`SECURITY.md`](SECURITY.md) for the disclosure process.

## Releasing

(Maintainers only.)

1. Bump `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
2. Open a "release: vX.Y.Z" PR.
3. After merge, tag from `main`:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

4. The `Publish` workflow auto-builds + pushes to PyPI (via OIDC trusted
   publishing) and the Docker image to GHCR.
5. Create a GitHub Release pointing at the new tag with summarized notes.

## Code of Conduct

This project follows the [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Be kind. Personal attacks, harassment, and discrimination aren't welcome.
