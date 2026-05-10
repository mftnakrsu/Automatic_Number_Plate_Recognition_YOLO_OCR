"""Application settings — env + .env driven via pydantic-settings.

All settings are namespaced with ANPR_ env prefix. See `.env.example` for the
canonical list. Settings are read once at app startup; no hot-reload.
"""

from __future__ import annotations

import secrets
import warnings

from pydantic import Field, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    cors_origins: str = "*"

    @field_validator("plate_hmac_pepper")
    @classmethod
    def _ensure_pepper(cls, v: SecretStr) -> SecretStr:
        s = v.get_secret_value()
        if not s:
            generated = secrets.token_hex(32)
            warnings.warn(
                "ANPR_PLATE_HMAC_PEPPER not set — generated a random one. "
                "In production, set this env var so plate hashes are stable across restarts.",
                stacklevel=2,
            )
            return SecretStr(generated)
        if len(s) < 32:
            raise ValueError("ANPR_PLATE_HMAC_PEPPER must be at least 32 hex chars (256 bits).")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    return Settings()
