"""`anpr generate-api-key` prints a key that ANPR_API_KEY accepts."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from anpr.cli import app
from anpr.config import Settings


def test_generated_api_key_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    result = CliRunner().invoke(app, ["generate-api-key"])
    assert result.exit_code == 0, result.output
    key = result.output.strip()
    assert len(key) >= 32

    monkeypatch.setenv("ANPR_PLATE_HMAC_PEPPER", "x" * 64)
    monkeypatch.setenv("ANPR_API_KEY", key)
    assert Settings().auth_required
