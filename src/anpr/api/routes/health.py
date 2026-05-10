from __future__ import annotations

from fastapi import APIRouter

from anpr import __version__

router = APIRouter(tags=["meta"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/version")
async def version() -> dict[str, str]:
    return {"version": __version__}
