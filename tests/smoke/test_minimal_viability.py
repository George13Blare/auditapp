"""Smoke tests for minimal application viability."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.mark.smoke
def test_import_smoke_core_package() -> None:
    """Базовый import пакета должен работать без побочных эффектов."""
    module = importlib.import_module("dcmmetatest")
    assert module is not None


@pytest.mark.smoke
def test_cli_help_smoke() -> None:
    """CLI --help должен завершаться успешно."""
    result = subprocess.run(
        [sys.executable, "-m", "src.dcmmetatest.cli", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage" in result.stdout.lower() or "использование" in result.stdout.lower()


@pytest.mark.smoke
def test_streamlit_module_load_smoke() -> None:
    """Streamlit entrypoint должен импортироваться."""
    module = importlib.import_module("app")
    assert module is not None
