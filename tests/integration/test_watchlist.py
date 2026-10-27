"""Watchlist storage, HTTP routes, the stream alert and the `anpr watchlist` commands."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from anpr.detector.base import BBox, Detection
from anpr.ocr.base import OcrResult
from anpr.storage.hashing import hash_plate
from anpr.storage.repository import WatchlistRepository, init_db
from anpr.watchlist import Watchlist

PEPPER = "w" * 64


class _MockDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(BBox(w // 4, h // 4, 3 * w // 4, 3 * h // 4), confidence=0.9)]


class _MockReader:
    def read(self, crop: np.ndarray) -> OcrResult | None:
        return OcrResult(text="34 ABC 1234", confidence=0.96)

    def read_batch(self, crops: list[np.ndarray]) -> list[OcrResult | None]:
        return [self.read(c) for c in crops]


class _MockRepo:
    def __init__(self) -> None:
        self.saved: list[Any] = []

    async def save(self, detection: Any) -> Any:
        self.saved.append(detection)
        return detection


class _FakeNotifier:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def notify(self, event: dict[str, Any]) -> bool:
        self.events.append(event)
        return True


# --- Storage ------------------------------------------------------------------


@pytest_asyncio.fixture
async def repo() -> AsyncIterator[WatchlistRepository]:
    engine = await init_db("sqlite+aiosqlite:///:memory:")
    yield WatchlistRepository(engine)
    await engine.dispose()


async def test_add_relabel_and_remove(repo: WatchlistRepository) -> None:
    watchlist = Watchlist(repo)
    await watchlist.add("a" * 64, "van")
    await watchlist.add("a" * 64, "truck")
    await watchlist.add("b" * 64)

    reloaded = Watchlist(repo)
    await reloaded.load()
    assert len(reloaded) == 2
    entry = reloaded.match("a" * 64)
    assert entry is not None
    assert entry.label == "truck"

    removed = await reloaded.remove("a" * 64)
    removed_again = await reloaded.remove("a" * 64)
    assert removed
    assert not removed_again
    assert reloaded.match("a" * 64) is None
    remaining = await repo.list_all()
    assert [e.plate_hmac for e in remaining] == ["b" * 64]


async def test_memory_only_watchlist() -> None:
    watchlist = Watchlist()
    await watchlist.add("c" * 64, None)
    assert watchlist.match("c" * 64) is not None
    removed = await watchlist.remove("c" * 64)
    assert removed
    assert len(watchlist) == 0


# --- HTTP and WebSocket -------------------------------------------------------


@pytest.fixture
def deps(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", PEPPER)
    monkeypatch.setenv("ANPR_DETECT_EVERY_N_FRAMES", "1")
    monkeypatch.setenv("ANPR_MIN_TRACK_DWELL", "3")
    monkeypatch.delenv("ANPR_API_KEY", raising=False)
    from anpr.api import deps as deps_mod

    deps_mod._settings_singleton.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setattr(deps_mod, "_DETECTOR", _MockDetector())
    monkeypatch.setattr(deps_mod, "_READER", _MockReader())
    monkeypatch.setattr(deps_mod, "_REPO", _MockRepo())
    monkeypatch.setattr(deps_mod, "_WATCHLIST", Watchlist())
    monkeypatch.setattr(deps_mod, "_NOTIFIER", _FakeNotifier())
    yield deps_mod
    deps_mod._settings_singleton.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def client(deps: Any) -> TestClient:
    from anpr.api.app import create_app

    return TestClient(create_app())


def test_routes_add_list_delete(client: TestClient) -> None:
    r = client.post("/api/v1/watchlist", json={"plate": "34-abc-1234", "label": "van"})
    assert r.status_code == 201, r.text
    plate_hmac = r.json()["plate_hmac"]
    assert plate_hmac == hash_plate("34 ABC 1234", PEPPER)

    listed = client.get("/api/v1/watchlist").json()
    assert [(e["plate_hmac"], e["label"]) for e in listed] == [(plate_hmac, "van")]
    assert set(listed[0]) == {"plate_hmac", "label", "created_at"}

    first = client.delete(f"/api/v1/watchlist/{plate_hmac}")
    second = client.delete(f"/api/v1/watchlist/{plate_hmac}")
    assert first.status_code == 204
    assert second.status_code == 404
    assert client.get("/api/v1/watchlist").json() == []


def test_route_rejects_bad_input(client: TestClient) -> None:
    responses = [
        client.post("/api/v1/watchlist", json={"plate": " - "}),
        client.post("/api/v1/watchlist", json={"plate": ""}),
        client.delete("/api/v1/watchlist/not-a-hash"),
    ]
    assert [r.status_code for r in responses] == [422, 422, 422]


def test_route_refuses_a_generated_pepper(deps: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from anpr.api.app import create_app

    monkeypatch.delenv("ANPR_PLATE_HMAC_PEPPER")
    deps._settings_singleton.cache_clear()
    with pytest.warns(UserWarning, match="PEPPER"):
        client = TestClient(create_app())
    r = client.post("/api/v1/watchlist", json={"plate": "34 ABC 1234"})
    assert r.status_code == 409


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_stream_alerts_on_a_watched_plate(client: TestClient, deps: Any) -> None:
    client.post("/api/v1/watchlist", json={"plate": "34 ABC 1234", "label": "van"})
    with client.websocket_connect("/ws/stream") as ws:
        events = []
        for _ in range(4):
            ws.send_bytes(_jpeg())
            events.append(ws.receive_json())
    assert events[2]["confirmed"] == "34 ABC 1234"

    notifier: _FakeNotifier = deps._NOTIFIER
    assert len(notifier.events) == 1
    event = notifier.events[0]
    assert event["source"] == "stream"
    assert event["label"] == "van"
    assert event["plate_hmac"] == hash_plate("34 ABC 1234", PEPPER)


def test_stream_ignores_unwatched_plates(client: TestClient, deps: Any) -> None:
    with client.websocket_connect("/ws/stream") as ws:
        for _ in range(4):
            ws.send_bytes(_jpeg())
            ws.receive_json()
    assert deps._NOTIFIER.events == []
    assert len(deps._REPO.saved) == 1


# --- CLI ----------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    import anpr.cli as cli

    db = tmp_path / "anpr.db"
    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", PEPPER)
    monkeypatch.setenv("ANPR_DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("ANPR_DETECT_EVERY_N_FRAMES", "1")
    monkeypatch.setenv("ANPR_MIN_TRACK_DWELL", "3")
    monkeypatch.delenv("ANPR_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(cli, "_build_models", lambda settings: (_MockDetector(), _MockReader()))
    return db


def _run(*args: str) -> Any:
    import anpr.cli as cli

    return CliRunner().invoke(cli.app, list(args))


def _listed() -> list[dict[str, Any]]:
    result = _run("watchlist", "list", "--json")
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_cli_add_list_remove(cli_env: Path) -> None:
    assert _run("watchlist", "add", "34 ABC 1234", "--label", "van").exit_code == 0
    listed = _listed()
    assert listed[0]["plate_hmac"] == hash_plate("34ABC1234", PEPPER)
    assert listed[0]["label"] == "van"

    assert _run("watchlist", "remove", "34 abc 1234").exit_code == 0
    assert _run("watchlist", "remove", "34 abc 1234").exit_code == 1
    assert _listed() == []


def test_cli_remove_by_hash(cli_env: Path) -> None:
    _run("watchlist", "add", "06 AA 42")
    assert _run("watchlist", "remove", hash_plate("06 AA 42", PEPPER)).exit_code == 0
    assert _listed() == []


def test_cli_import(cli_env: Path, tmp_path: Path) -> None:
    file = tmp_path / "plates.txt"
    file.write_text("# fleet\n34 ABC 1234, van\n\n06 AA 42\n", encoding="utf-8")
    result = _run("watchlist", "import", str(file))
    assert result.exit_code == 0, result.output
    labels = {e["plate_hmac"]: e["label"] for e in _listed()}
    assert labels == {hash_plate("34ABC1234", PEPPER): "van", hash_plate("06AA42", PEPPER): None}


def test_cli_add_needs_a_pepper(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANPR_PLATE_HMAC_PEPPER")
    result = _run("watchlist", "add", "34 ABC 1234")
    assert result.exit_code == 2
    assert not cli_env.exists()


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (320, 240))
    for _ in range(30):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


def test_watch_calls_the_webhook(
    cli_env: Path, video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anpr.webhook as webhook

    calls: list[dict[str, Any]] = []

    def receive(url: str, body: bytes, headers: dict[str, str], timeout: float) -> int:
        calls.append({"url": url, "body": json.loads(body), "headers": headers})
        return 200

    monkeypatch.setattr(webhook, "post_json", receive)
    monkeypatch.setenv("ANPR_WEBHOOK_URL", "http://hooks.test/anpr")
    monkeypatch.setenv("ANPR_WEBHOOK_SECRET", "s3cret")
    _run("watchlist", "add", "34 ABC 1234", "--label", "van")

    result = _run("watch", str(video), "--json")
    assert result.exit_code == 0, result.output
    line = json.loads(result.stdout.splitlines()[0])
    assert line["watchlist"] == {"label": "van"}

    assert len(calls) == 1
    assert calls[0]["url"] == "http://hooks.test/anpr"
    assert calls[0]["body"]["source"] == "watch"
    assert calls[0]["body"]["label"] == "van"
    assert calls[0]["headers"]["X-ANPR-Signature"].startswith("sha256=")


def test_watch_without_store_or_webhook_opens_no_database(cli_env: Path, video: Path) -> None:
    result = _run("watch", str(video), "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout.splitlines()[0])["watchlist"] is None
    assert not cli_env.exists()


def test_webhook_url_must_be_http(monkeypatch: pytest.MonkeyPatch) -> None:
    from anpr.config import Settings

    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", PEPPER)
    monkeypatch.setenv("ANPR_WEBHOOK_URL", "file:///etc/passwd")
    with pytest.raises(ValueError, match="http"):
        Settings()
