"""Application settings — env + .env driven via pydantic-settings.

All settings are namespaced with ANPR_ env prefix. See `.env.example` for the
canonical list. Settings are read once at app startup; no hot-reload.
"""

from __future__ import annotations

import re
import secrets
import warnings

from pydantic import Field, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MIN_API_KEY_LEN = 16
# The key travels in a WebSocket subprotocol token, which rules out most punctuation.
_API_KEY_CHARS = re.compile(r"[A-Za-z0-9_-]+")
# Peppers made up because none was configured. Hashes made with one are lost on restart.
_GENERATED_PEPPERS: set[str] = set()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ANPR_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "anpr"
    log_level: str = "INFO"
    log_json: bool = False

    detect_every_n_frames: PositiveInt = 3
    sr_min_plate_width: int = Field(default=80, ge=0)
    min_track_dwell: PositiveInt = 5
    ocr_fallback_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    detector_model: str = "yolo-v9-t-384-license-plate-end2end"
    detector_conf_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    ocr_model: str = "cct-s-v2-global-model"
    device: str = "auto"

    database_url: str = "sqlite+aiosqlite:///data/anpr.db"
    plate_hmac_pepper: SecretStr = SecretStr("")
    retention_hours: PositiveInt = 720

    api_host: str = "0.0.0.0"
    api_port: PositiveInt = 8000
    max_upload_mb: PositiveInt = 10
    # Decoded-raster ceiling for both /api/v1/infer and /ws/stream. The byte caps
    # above do not bound this: a small compressed file can declare an enormous
    # raster. 40 MP sits above 8K UHD (33.2 MP) and bounds one decode at ~120 MB
    # RGB (~240 MB peak, counting the BGR copy).
    max_image_pixels: PositiveInt = Field(default=40_000_000, ge=10_000)
    # Per-frame compressed cap for /ws/stream, also passed to uvicorn's
    # ws_max_size so the protocol layer rejects oversized frames first.
    max_ws_frame_bytes: PositiveInt = Field(default=4 * 1024 * 1024, ge=1024)
    cors_origins: str = "*"
    # Empty leaves /api/v1/* and /ws/stream open. When set, callers send it as
    # `X-API-Key`, `Authorization: Bearer`, or the `anpr.key.<key>` WebSocket
    # subprotocol, so the key stays URL-safe and out of query strings and logs.
    api_key: SecretStr = SecretStr("")

    # Watchlist matches are POSTed here. A secret: chat webhook URLs carry their token.
    webhook_url: SecretStr = SecretStr("")
    # Signs each call (X-ANPR-Signature); empty sends unsigned requests.
    webhook_secret: SecretStr = SecretStr("")
    webhook_timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    # The same plate is announced at most once in this many seconds.
    webhook_cooldown_seconds: int = Field(default=60, ge=0)

    @field_validator("plate_hmac_pepper")
    @classmethod
    def _ensure_pepper(cls, v: SecretStr) -> SecretStr:
        s = v.get_secret_value()
        if not s:
            generated = secrets.token_hex(32)
            _GENERATED_PEPPERS.add(generated)
            warnings.warn(
                "ANPR_PLATE_HMAC_PEPPER not set — generated a random one. "
                "In production, set this env var so plate hashes are stable across restarts.",
                stacklevel=2,
            )
            return SecretStr(generated)
        if len(s) < 32:
            raise ValueError("ANPR_PLATE_HMAC_PEPPER must be at least 32 hex chars (256 bits).")
        return v

    @field_validator("api_key")
    @classmethod
    def _check_api_key(cls, v: SecretStr) -> SecretStr:
        s = v.get_secret_value()
        if not s:
            return v
        if len(s) < _MIN_API_KEY_LEN:
            raise ValueError(f"ANPR_API_KEY must be at least {_MIN_API_KEY_LEN} characters.")
        if not _API_KEY_CHARS.fullmatch(s):
            raise ValueError(
                "ANPR_API_KEY may only contain letters, digits, '-' and '_' "
                "(generate one with `anpr generate-api-key`)."
            )
        return v

    @field_validator("webhook_url")
    @classmethod
    def _check_webhook_url(cls, v: SecretStr) -> SecretStr:
        s = v.get_secret_value()
        if s and not s.lower().startswith(("http://", "https://")):
            raise ValueError("ANPR_WEBHOOK_URL must start with http:// or https://")
        return v

    @property
    def pepper_is_generated(self) -> bool:
        """True when no pepper was configured, so hashes won't survive a restart."""
        return self.plate_hmac_pepper.get_secret_value() in _GENERATED_PEPPERS

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key.get_secret_value())

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    return Settings()
