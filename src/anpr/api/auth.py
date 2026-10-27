"""Optional API key for the inference and detection endpoints.

Off unless `ANPR_API_KEY` is set. HTTP callers send the key as `X-API-Key` or
`Authorization: Bearer <key>`. Browsers can't set headers on a WebSocket, so
`/ws/stream` also accepts it as a subprotocol: the client offers
`["anpr", "anpr.key.<key>"]` and the server answers with `anpr`. That keeps the
key out of the URL, where uvicorn's access log would record it.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, WebSocket, status
from starlette.datastructures import Headers

from anpr.api.deps import app_settings
from anpr.config import Settings

WS_SUBPROTOCOL = "anpr"
_WS_KEY_PREFIX = "anpr.key."


def _matches(presented: str | None, settings: Settings) -> bool:
    if presented is None:
        return False
    expected = settings.api_key.get_secret_value()
    return secrets.compare_digest(presented.encode(), expected.encode())


def _key_from_headers(headers: Headers) -> str | None:
    key = headers.get("x-api-key")
    if key:
        return key
    auth = headers.get("authorization") or ""
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token.strip()
    return None


async def require_api_key(
    request: Request,
    settings: Annotated[Settings, Depends(app_settings)],
) -> None:
    """Router dependency: 401 unless the request carries the configured key."""
    if not settings.auth_required:
        return
    if not _matches(_key_from_headers(request.headers), settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def authorize_websocket(websocket: WebSocket, settings: Settings) -> tuple[bool, str | None]:
    """Check a WebSocket handshake before accepting it.

    Returns whether the connection is allowed and the subprotocol to accept
    with: `anpr` when the client offered it, so browsers that sent the key as a
    subprotocol get a matching answer, otherwise None.
    """
    offered: list[str] = list(websocket.scope.get("subprotocols") or [])
    subprotocol = WS_SUBPROTOCOL if WS_SUBPROTOCOL in offered else None
    if not settings.auth_required:
        return True, subprotocol

    presented = _key_from_headers(websocket.headers)
    if presented is None:
        presented = next(
            (p.removeprefix(_WS_KEY_PREFIX) for p in offered if p.startswith(_WS_KEY_PREFIX)),
            None,
        )
    return _matches(presented, settings), subprotocol
