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
python -m src.dcmmetatest.cli /path/to/dataset --config config.yaml
```

---

## Registry Builder v1 (канонический manifest)

Для задач реестра данных доступен отдельный режим `registry builder v1`: intake сырых файлов → канонический `manifest`.

### Зафиксированные сущности v1

- `patient`
- `study`
- `series`
- `instance`
- `annotation`
- `split_assignment`
- `preprocessing_artifact`

### Schema v1 и versioning

- `manifest_version`: версия контракта манифеста (для v1 = `1.0`)
- `schema_revision`: ревизия внутри версии схемы (для v1 = `1`)

Правила:
1. Ломающие изменения структуры повышают `manifest_version`.
2. Обратно-совместимые доработки внутри v1 повышают `schema_revision`.
3. Экспортеры/валидаторы должны проверять оба поля до чтения сущностей.

### Пример конфига для builder

```yaml
# examples/registry_builder_v1.yaml
exclude_patterns:
  - "**/tmp/*"
  - "**/*.bak"
registry_format: json
```

### Пример запуска builder

```bash
python -m src.dcmmetatest.cli /path/to/dataset \
  --build-registry \
  --registry-format json \
  --registry-output manifest_v1.json \
  --summary-output manifest_summary.json
```

Поддерживаемые форматы экспорта manifest: `json`, `csv`, `parquet` (для parquet нужен `pyarrow`).

---

## Структура проекта

```
.
├── app.py                     # Точка входа Streamlit (веб-интерфейс)
├── dcmmetatest_plus.py        # Legacy скрипт (для обратной совместимости)
├── requirements.txt           # Зависимости
├── pyproject.toml            # Конфигурация проекта
├── README.md                 # Документация
├── src/
│   └── dcmmetatest/
│       ├── __init__.py       # Публичный API
│       ├── analyzer.py       # Логика анализа
│       ├── cli.py            # CLI интерфейс
│       ├── detectors.py      # Детекторы DICOM и разметки
│       ├── io.py             # Операции ввода-вывода
│       ├── models.py         # Модели данных
│       ├── ui.py             # Утилиты для веб-интерфейса
│       └── utils.py          # Вспомогательные функции
└── tests/
    └── test_dcmmetatest.py   # Юнит-тесты
```

---

## Разработка

### Установка для разработки

```bash
pip install -e ".[dev]"
```

### Запуск тестов

```bash
pytest tests/ -v
```

### Линтинг и форматирование

```bash
# Форматирование кода
black src/ app.py

# Проверка стиля
ruff check src/ app.py

# Проверка типов
mypy src/
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
