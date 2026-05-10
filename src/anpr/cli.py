"""Typer CLI entry point — `anpr ...`."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from anpr import __version__
from anpr.config import get_settings
from anpr.logging import configure_logging, get_logger

app = typer.Typer(
    name="anpr",
    help="Automatic Number Plate Recognition CLI.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
log = get_logger(__name__)


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
    console.print_json(json.dumps(data, default=str))


@app.command("generate-pepper")
def generate_pepper() -> None:
    """Generate a fresh HMAC pepper for ANPR_PLATE_HMAC_PEPPER."""
    console.print(secrets.token_hex(32))


@app.command()
def infer(
    image: Annotated[Path, typer.Argument(help="Path to image file.")],
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
) -> None:
    """Detect plates and OCR them in a single image."""
    if not image.exists():
        console.print(f"[red]File not found:[/red] {image}")
        raise typer.Exit(code=2)

    from anpr.detector.fast_alpr import FastAlprDetector
    from anpr.ocr.fast_plate import FastPlateOcr
    from anpr.pipeline import infer_image

    settings = get_settings()
    detector = FastAlprDetector(
        model_name=settings.detector_model,
        conf_threshold=settings.detector_conf_threshold,
    )
    reader = FastPlateOcr(model_name=settings.ocr_model, device=settings.device)

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
def serve(
    host: Annotated[str, typer.Option(envvar="ANPR_API_HOST")] = "0.0.0.0",
    port: Annotated[int, typer.Option(envvar="ANPR_API_PORT")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on file changes.")] = False,
) -> None:
    """Start the FastAPI inference server."""
    import uvicorn

    uvicorn.run(
        "anpr.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
