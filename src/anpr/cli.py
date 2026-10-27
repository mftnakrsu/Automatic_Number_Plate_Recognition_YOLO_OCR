"""Typer CLI entry point — `anpr ...`."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from anpr import __version__
from anpr.config import Settings, get_settings
from anpr.logging import configure_logging, get_logger

if TYPE_CHECKING:
    import numpy as np

    from anpr.detector.base import Detector
    from anpr.ocr.base import PlateReader

app = typer.Typer(
    name="anpr",
    help="Automatic Number Plate Recognition CLI.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)
log = get_logger(__name__)


def _build_models(settings: Settings) -> tuple[Detector, PlateReader]:
    """Load the detector and OCR model named in the settings (downloads on first use)."""
    from anpr.detector.fast_alpr import FastAlprDetector
    from anpr.ocr.fast_plate import FastPlateOcr

    detector = FastAlprDetector(
        model_name=settings.detector_model,
        conf_threshold=settings.detector_conf_threshold,
    )
    reader = FastPlateOcr(model_name=settings.ocr_model, device=settings.device)
    return detector, reader


@app.callback()
def _root(
    log_level: Annotated[str, typer.Option("--log-level", envvar="ANPR_LOG_LEVEL")] = "INFO",
    log_json: Annotated[bool, typer.Option("--log-json", envvar="ANPR_LOG_JSON")] = False,
) -> None:
    configure_logging(level=log_level, json=log_json)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"anpr {__version__}")


@app.command("dump-config")
def dump_config() -> None:
    """Print the effective settings (secrets redacted)."""
    settings = get_settings()
    data = settings.model_dump()
    if "plate_hmac_pepper" in data:
        data["plate_hmac_pepper"] = "***"
    data["api_key"] = "***" if settings.auth_required else ""
    console.print_json(json.dumps(data, default=str))


@app.command("generate-pepper")
def generate_pepper() -> None:
    """Generate a fresh HMAC pepper for ANPR_PLATE_HMAC_PEPPER."""
    console.print(secrets.token_hex(32))


@app.command("generate-api-key")
def generate_api_key() -> None:
    """Generate a random key for ANPR_API_KEY."""
    console.print(secrets.token_urlsafe(32))


@app.command()
def infer(
    image: Annotated[Path, typer.Argument(help="Path to image file.")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
) -> None:
    """Detect plates and OCR them in a single image."""
    if not image.exists():
        console.print(f"[red]File not found:[/red] {image}")
        raise typer.Exit(code=2)

    from anpr.pipeline import infer_image

    settings = get_settings()
    detector, reader = _build_models(settings)

    log.info("inference.start", image=str(image))
    reads = infer_image(image, detector, reader)
    log.info("inference.done", count=len(reads))

    if json_out:
        out = [
            {
                "bbox": r.detection.bbox.as_xyxy(),
                "detection_confidence": r.detection.confidence,
                "raw_text": r.raw_text,
                "normalized_text": r.normalized_text,
                "parsed": (
                    {
                        "province_code": r.parsed.province_code,
                        "province_name": r.parsed.province_name,
                        "letters": r.parsed.letters,
                        "digits": r.parsed.digits,
                        "canonical": r.parsed.canonical,
                    }
                    if r.parsed
                    else None
                ),
            }
            for r in reads
        ]
        console.print_json(json.dumps(out, ensure_ascii=False))
        return

    if not reads:
        console.print("[yellow]No plates detected.[/yellow]")
        return

    table = Table(title=f"Detections in {image.name}")
    table.add_column("BBox", style="cyan")
    table.add_column("Det conf")
    table.add_column("Raw OCR")
    table.add_column("Plate")
    table.add_column("Province")
    for r in reads:
        bb = r.detection.bbox
        table.add_row(
            f"({bb.x1},{bb.y1})-({bb.x2},{bb.y2})",
            f"{r.detection.confidence:.2f}",
            r.raw_text or "—",
            r.parsed.canonical if r.parsed else "[red](invalid)[/red]",
            r.parsed.province_name if r.parsed else "—",
        )
    console.print(table)


@app.command()
def watch(
    source: Annotated[
        str,
        typer.Argument(help="Video file, stream URL (rtsp://, http://, ...) or camera index (0)."),
    ],
    json_out: Annotated[
        bool, typer.Option("--json", help="One JSON object per confirmed plate.")
    ] = False,
    store: Annotated[
        bool,
        typer.Option("--store", help="Save confirmed plates, hashed, to ANPR_DATABASE_URL."),
    ] = False,
    max_frames: Annotated[
        int | None, typer.Option("--max-frames", min=1, help="Stop after this many frames.")
    ] = None,
    reconnect: Annotated[
        int,
        typer.Option("--reconnect", min=0, help="Reconnect attempts after a live source drops."),
    ] = 5,
) -> None:
    """Read plates from a video file, network stream or camera, with tracking and voting."""
    from anpr.video import SourceError

    settings = get_settings()
    detector, reader = _build_models(settings)
    try:
        frames, plates = asyncio.run(
            _watch(
                source,
                detector,
                reader,
                settings,
                json_out=json_out,
                store=store,
                max_frames=max_frames,
                reconnect=reconnect,
            )
        )
    except SourceError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e
    except KeyboardInterrupt:
        return
    err_console.print(f"{frames} frames read, {plates} plates confirmed.")


async def _watch(
    source: str,
    detector: Detector,
    reader: PlateReader,
    settings: Settings,
    *,
    json_out: bool,
    store: bool,
    max_frames: int | None,
    reconnect: int,
) -> tuple[int, int]:
    from anpr.storage.hashing import hash_plate
    from anpr.storage.models import Detection
    from anpr.storage.repository import DetectionRepository, init_db
    from anpr.video import parse_source, read_frames, watch

    counted = 0

    async def counting() -> AsyncIterator[np.ndarray]:
        nonlocal counted
        async for frame in read_frames(
            parse_source(source), max_frames=max_frames, reconnect_attempts=reconnect
        ):
            counted += 1
            yield frame

    engine = await init_db(settings.database_url) if store else None
    repo = DetectionRepository(engine) if engine is not None else None
    pepper = settings.plate_hmac_pepper.get_secret_value()
    confirmed = 0
    try:
        async for plate in watch(
            counting(),
            detector,
            reader,
            detect_every_n=settings.detect_every_n_frames,
            min_track_dwell=settings.min_track_dwell,
            sr_min_plate_width=settings.sr_min_plate_width,
        ):
            confirmed += 1
            if json_out:
                print(json.dumps(dataclasses.asdict(plate), ensure_ascii=False), flush=True)
            else:
                console.print(
                    f"frame {plate.frame_idx:>6}  track {plate.track_id:>4}  "
                    f"[bold]{plate.plate}[/bold]  {plate.province_name}  "
                    f"det={plate.confidence:.2f}",
                    highlight=False,
                )
            if repo is not None:
                x1, y1, x2, y2 = plate.bbox
                await repo.save(
                    Detection(
                        track_id=plate.track_id,
                        plate_hmac=hash_plate(plate.plate, pepper),
                        province_code=plate.province_code,
                        confidence=plate.confidence,
                        bbox_x1=x1,
                        bbox_y1=y1,
                        bbox_x2=x2,
                        bbox_y2=y2,
                        confirmed=True,
                    )
                )
    finally:
        if engine is not None:
            await engine.dispose()
    return counted, confirmed


@app.command()
def serve(
    host: Annotated[str, typer.Option(envvar="ANPR_API_HOST")] = "0.0.0.0",
    port: Annotated[int, typer.Option(envvar="ANPR_API_PORT")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on file changes.")] = False,
) -> None:
    """Start the FastAPI inference server."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "anpr.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        # Defence in depth: reject oversized frames at the protocol layer so
        # they never reach the decoder. uvicorn's own default is 16 MB.
        ws_max_size=settings.max_ws_frame_bytes,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
