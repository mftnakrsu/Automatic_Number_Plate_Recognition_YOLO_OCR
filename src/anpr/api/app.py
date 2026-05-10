"""FastAPI app factory."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from anpr import __version__
from anpr.api.deps import set_detector, set_reader, set_repo
from anpr.api.middleware import MetricsMiddleware, RequestContextMiddleware
from anpr.api.routes import detections, health, infer, stream
from anpr.config import get_settings
from anpr.logging import configure_logging, get_logger
from anpr.observability import configure_otel, render_metrics

_HERE = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_HERE / "templates"))
_STATIC_DIR = _HERE / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log = get_logger(__name__)
    log.info(
        "api.startup",
        detector=settings.detector_model,
        ocr=settings.ocr_model,
        database=settings.database_url.split("///")[-1],
    )

    from anpr.detector.fast_alpr import FastAlprDetector
    from anpr.ocr.fast_plate import FastPlateOcr
    from anpr.storage.repository import DetectionRepository, init_db, retention_worker

    set_detector(
        FastAlprDetector(
            model_name=settings.detector_model,
            conf_threshold=settings.detector_conf_threshold,
        )
    )
    set_reader(FastPlateOcr(model_name=settings.ocr_model, device=settings.device))

    engine = await init_db(settings.database_url)
    repo = DetectionRepository(engine)
    set_repo(repo)

    purge_task = asyncio.create_task(
        retention_worker(repo, retention_hours=settings.retention_hours)
    )

    log.info("api.ready")
    try:
        yield
    finally:
        purge_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await purge_task
        await engine.dispose()
        log.info("api.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, json=settings.log_json)

    app = FastAPI(
        title="anpr",
        version=__version__,
        description="Automatic Number Plate Recognition — 2026 modernized.",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(MetricsMiddleware)

    app.include_router(health.router)
    app.include_router(infer.router)
    app.include_router(stream.router)
    app.include_router(detections.router)

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(request, "index.html")

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        body, content_type = render_metrics()
        return Response(content=body, media_type=content_type)

    configure_otel(app)

    return app
