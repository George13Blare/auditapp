# DCM MetaTest Plus

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
