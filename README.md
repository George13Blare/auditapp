# DCM MetaTest Plus

[![CI](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/ci.yml)
[![Smoke](https://github.com/<OWNER>/<REPO>/actions/workflows/smoke.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/smoke.yml)

> Замените `<OWNER>/<REPO>` на фактический путь к вашему GitHub-репозиторию, чтобы бейджи отображали реальный статус.

Расширенный анализатор DICOM-датасетов с поддержкой группировки по UID, детекцией разметки, параллельной обработкой и **визуальным интерфейсом**.

## ✅ CI проверки

- **CI / Lint**: быстрый линтинг (`ruff`, `black --check`) на каждом `push` и `pull_request`.
- **CI / Unit tests**: запуск юнит/интеграционных тестов (`pytest tests -m "not smoke"`).
- **Smoke / Minimal viability**: отдельный smoke-набор минимальной жизнеспособности:
  - import core-пакета;
  - `CLI --help`;
  - загрузка Streamlit entrypoint (`app.py`).

## 🚀 Быстрый старт
Расширенный анализатор DICOM-датасетов с CLI и Streamlit UI.

## Установка

### Для использования

```bash
pip install .
```

### Для разработки

```bash
pip install -e ".[dev]"
```

## Запуск

### CLI

После установки пакет запускается без `PYTHONPATH`:

```bash
dcmmetatest-plus /path/to/dataset --format json --output report.json
```

Альтернатива через модуль:

```bash
python -m dcmmetatest.cli /path/to/dataset --format txt
```

### Streamlit UI

Запуск через entrypoint (без ручных path-хаков):

```bash
dcmmetatest-ui
```

Или напрямую:

```bash
streamlit run app.py
```

После запуска откройте `http://localhost:8501`.

## Smoke-check импортов

```bash
python -c "import dcmmetatest, dcmmetatest.cli, dcmmetatest.ui, dcmmetatest.normalizer, dcmmetatest.split"
```

## Структура пакета

```text
src/dcmmetatest/
  __init__.py
  cli.py
  ui.py
  ui_entrypoint.py
  normalizer.py
  split.py
  ...
```

## Тесты

```bash
pytest -q
```
