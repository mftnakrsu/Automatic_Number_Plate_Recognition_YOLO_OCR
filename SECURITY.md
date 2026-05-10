# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.2.x | ✅ |
| < 0.2.0 (legacy 2022 stack) | ❌ |

The `v0.1.0-legacy` tag is preserved verbatim under [`legacy/`](legacy/) for
historical reference. It is not maintained; CVEs in those packages will not
receive patches here. Use the modernized pipeline (`pip install anpr-pipeline`).

## Reporting a Vulnerability

**Please do not open a public issue for security problems.**

### Preferred — GitHub Private Security Advisory

<https://github.com/mftnakrsu/Automatic_Number_Plate_Recognition_YOLO_OCR/security/advisories/new>

### Alternative — Email

<meftunakrsu@gmail.com> — include `[ANPR-SECURITY]` in the subject. PGP key
available on request.

## Response Timeline

- **Acknowledgement**: within 7 days
- **Triage + fix plan**: within 14 days
- **Patched release on PyPI**: as fast as the severity warrants

## Disclosure

After a patched release is on PyPI we will publish a GitHub Security Advisory
and credit the reporter (unless anonymity is requested).

## Out of Scope

- The archived `legacy/` codebase — historical reference only, not maintained.
- Issues that require the attacker to already control the deployment
  environment (e.g., a leaked `ANPR_PLATE_HMAC_PEPPER`, root on the host).
- Vulnerabilities in transitive dependencies — please report directly to the
  upstream maintainer (`fast-alpr`, `fast-plate-ocr`, `fastapi`, etc.). We
  receive Dependabot alerts and will bump when patches land upstream.

## Hardening Notes for Operators

If you are deploying this in production:

- **Set a strong `ANPR_PLATE_HMAC_PEPPER`** (≥ 32 hex chars from `anpr
  generate-pepper`). Don't reuse across deployments.
- Restrict `ANPR_CORS_ORIGINS` to your actual frontend origins; never leave
  it as `*` in production.
- Front the FastAPI app with a TLS-terminating reverse proxy (Caddy, nginx,
  Cloudflare).
- Tune `ANPR_RETENTION_HOURS` to your retention policy. The default (720h /
  30d) is a hobby default, not a legal recommendation.
- Run behind authentication for the `/api/v1/detections` endpoint if it
  matters to your threat model — the bundled app exposes it unauthenticated.
