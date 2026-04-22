"""Import smoke tests to verify installable package layout."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "dcmmetatest",
        "dcmmetatest.cli",
        "dcmmetatest.normalizer",
        "dcmmetatest.split",
    ],
)
def test_core_module_imports(module_name: str):
    importlib.import_module(module_name)


def test_streamlit_related_imports():
    pytest.importorskip("streamlit")
    importlib.import_module("dcmmetatest.ui")
    importlib.import_module("dcmmetatest.ui_entrypoint")
