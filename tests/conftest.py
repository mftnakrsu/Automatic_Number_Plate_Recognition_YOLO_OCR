from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_plate_path(fixture_dir: Path) -> Path:
    return fixture_dir / "sample_plate.jpg"
