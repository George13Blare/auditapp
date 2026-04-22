"""Smoke-тест загрузки Streamlit UI без тяжёлых вычислений."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


class _SessionState(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


class _Widget:
    def progress(self, *_args, **_kwargs):
        return None

    def text(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def success(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class _Context(_Widget):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _StreamlitStub(ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.sidebar = self

    def set_page_config(self, **_kwargs):
        return None

    def title(self, *_args, **_kwargs):
        return None

    def markdown(self, *_args, **_kwargs):
        return None

    def caption(self, *_args, **_kwargs):
        return None

    def header(self, *_args, **_kwargs):
        return None

    def subheader(self, *_args, **_kwargs):
        return None

    def text_input(self, *_args, **kwargs):
        return kwargs.get("value", "")

    def selectbox(self, _label, options, index=0, **_kwargs):
        return options[index]

    def slider(self, _label, _min_value=None, _max_value=None, value=0, *_args, **_kwargs):
        return value

    def checkbox(self, _label, value=False, **_kwargs):
        return value

    def multiselect(self, _label, options=None, default=None, **_kwargs):
        return default or []

    def button(self, *_args, **_kwargs):
        return False

    def divider(self):
        return None

    def container(self):
        return _Context()

    def progress(self, *_args, **_kwargs):
        return _Widget()

    def empty(self):
        return _Widget()

    def columns(self, spec, **_kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_Context() for _ in range(count)]

    def tabs(self, labels):
        return [_Context() for _ in labels]

    def expander(self, *_args, **_kwargs):
        return _Context()

    def plotly_chart(self, *_args, **_kwargs):
        return None

    def metric(self, *_args, **_kwargs):
        return None

    def dataframe(self, *_args, **_kwargs):
        return None

    def write(self, *_args, **_kwargs):
        return None

    def code(self, *_args, **_kwargs):
        return None

    def json(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def success(self, *_args, **_kwargs):
        return None

    def stop(self):
        raise RuntimeError("st.stop() called")

    def spinner(self, *_args, **_kwargs):
        return _Context()

    def rerun(self):
        return None

    def radio(self, _label, options, index=0, **_kwargs):
        return options[index]

    def number_input(self, _label, value=0, **_kwargs):
        return value

    def file_uploader(self, *_args, **_kwargs):
        return None

    def download_button(self, *_args, **_kwargs):
        return None

    def color_picker(self, _label, value="#FF0000", **_kwargs):
        return value


def test_app_module_import_smoke(monkeypatch):
    streamlit_stub = _StreamlitStub()
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)

    runtime_mod = ModuleType("streamlit.runtime")
    caching_mod = ModuleType("streamlit.runtime.caching")

    def _cache_data(*_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    caching_mod.cache_data = _cache_data
    monkeypatch.setitem(sys.modules, "streamlit.runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "streamlit.runtime.caching", caching_mod)

    app_module = importlib.import_module("app")
    assert app_module is not None
