"""Smoke tests for public exports in ui module."""

import pytest


pytest.importorskip("streamlit")

from dcmmetatest import ui


def test_ui_module_exports_preprocessing_symbols():
    assert hasattr(ui, "AugmentationConfig")
    assert hasattr(ui, "PreprocessingPipelineConfig")
    assert callable(ui.run_preprocessing_pipeline)
