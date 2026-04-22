"""Smoke tests for public exports in ui module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcmmetatest import ui


def test_ui_module_exports_preprocessing_symbols():
    assert hasattr(ui, "AugmentationConfig")
    assert hasattr(ui, "PreprocessingPipelineConfig")
    assert callable(ui.run_preprocessing_pipeline)
