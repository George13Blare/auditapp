"""CLI entrypoint to launch the Streamlit UI without manual PYTHONPATH hacks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    raise SystemExit(main())
