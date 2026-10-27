"""/api/v1/watchlist — the plates that raise an alert when confirmed.

A plate is sent once, as text, and stored as its HMAC; listing and deleting
work on the hash.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from pydantic import BaseModel, Field

from anpr.api.auth import require_api_key
from anpr.api.deps import app_settings, get_watchlist
from anpr.config import Settings
from anpr.storage.hashing import hash_plate
from anpr.storage.models import WatchlistEntry
from anpr.watchlist import Watchlist

router = APIRouter(prefix="/api/v1", tags=["watchlist"], dependencies=[Depends(require_api_key)])


class WatchlistIn(BaseModel):
    plate: str = Field(min_length=1, max_length=32, examples=["34 ABC 1234"])
    label: str | None = Field(default=None, max_length=200, examples=["delivery van"])


class WatchlistOut(BaseModel):
    plate_hmac: str
    label: str | None
    created_at: datetime


def _out(entry: WatchlistEntry) -> WatchlistOut:
    return WatchlistOut(plate_hmac=entry.plate_hmac, label=entry.label, created_at=entry.created_at)


@router.get("/watchlist", response_model=list[WatchlistOut])
async def list_watchlist(
    watchlist: Annotated[Watchlist, Depends(get_watchlist)],
) -> list[WatchlistOut]:
    return [_out(e) for e in watchlist.entries()]


@router.post("/watchlist", response_model=WatchlistOut, status_code=status.HTTP_201_CREATED)
async def add_to_watchlist(
    body: WatchlistIn,
    settings: Annotated[Settings, Depends(app_settings)],
    watchlist: Annotated[Watchlist, Depends(get_watchlist)],
) -> WatchlistOut:
    if settings.pepper_is_generated:
        # The hash would change with the next pepper, and the entry would never match again.
        raise HTTPException(status_code=409, detail="set ANPR_PLATE_HMAC_PEPPER first")
    if not any(c.isalnum() for c in body.plate):
        raise HTTPException(status_code=422, detail="plate has no letters or digits")
    plate_hmac = hash_plate(body.plate, settings.plate_hmac_pepper.get_secret_value())
    return _out(await watchlist.add(plate_hmac, body.label))


@router.delete("/watchlist/{plate_hmac}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_from_watchlist(
    plate_hmac: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")],
    watchlist: Annotated[Watchlist, Depends(get_watchlist)],
) -> Response:
    if not await watchlist.remove(plate_hmac):
        raise HTTPException(status_code=404, detail="not on the watchlist")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
