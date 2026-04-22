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

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск CLI

```bash
python -m src.dcmmetatest.cli /path/to/dataset --output report.txt
```

### Запуск веб-интерфейса (Streamlit)

```bash
streamlit run app.py
```

Или:

```bash
python -m streamlit run app.py
```

После запуска откройте браузер по адресу `http://localhost:8501`

---

## Возможности

### Анализ данных

- **Надёжное определение DICOM**: без строгой привязки к преамбуле "DICM", с fallback через pydicom
- **Детекция разметки**: поддержка RTSTRUCT, SEG, RTSEGANN, SOP Class UID, SeriesDescription, SegmentSequence
- **Группировка исследований**: по директориям или по StudyInstanceUID
- **Параллельная обработка**: выбор типа пула (process/thread) и числа воркеров
- **Контроль обхода**: глубина, следование симлинкам, исключение по шаблонам
- **Авто-детект структуры**: анализ структуры датасета с сохранением схемы (JSON/YAML)
- **Фильтры**: по modality, только размеченные/неанонимные исследования
- **Отчёты**: TXT, CSV, JSON форматы

### Визуальный интерфейс (Streamlit)

- **📊 Dashboard**: Карточки с ключевыми метриками (исследования, файлы, пациенты, процент разметки)
- **📈 Графики**: 
  - Круговая диаграмма распределения модальностей
  - Столбчатая диаграмма источников разметки
- **📋 Интерактивная таблица**: 
  - Полные данные по всем исследованиям
  - Поиск и фильтрация в реальном времени
  - Сортировка по любому полю
- **⚠️ Детектор проблем**: 
  - Список ошибок обработки
  - Пустые папки
  - Неанонимизированные данные
- **📄 Экспорт**: 
  - Скачивание отчётов в CSV, JSON, TXT прямо из интерфейса
  - Мгновенная генерация без повторного анализа

---

## Использование

### CLI (Командная строка)

#### Базовый запуск

```bash
python -m src.dcmmetatest.cli /path/to/dataset --output report.txt
```

#### Группировка по StudyInstanceUID

```bash
python -m src.dcmmetatest.cli /path/to/dataset --group-by study --executor thread --workers 4
```

#### Фильтрация по модальностям

```bash
python -m src.dcmmetatest.cli /path/to/dataset --modality-filter CT MR --only-labeled
```

#### Интерактивный режим

```bash
python -m src.dcmmetatest.cli /path/to/dataset --interactive
```

#### Авто-детект структуры

```bash
python -m src.dcmmetatest.cli /path/to/dataset --auto-detect-schema --schema-output schema.json
```

#### Все опции

```bash
python -m src.dcmmetatest.cli --help
```

### Веб-интерфейс

1. Запустите приложение:
   ```bash
   streamlit run app.py
   ```

2. Откройте браузер: `http://localhost:8501`

3. В боковой панели:
   - Укажите путь к датасету
   - Настройте параметры (группировка, потоки, фильтры)
   - Нажмите "🚀 Запустить анализ"

4. Просмотр результатов:
   - **Вкладка "Графики"**: Визуализация распределения модальностей и источников разметки
   - **Вкладка "Таблица данных"**: Детальная информация по каждому исследованию с поиском
   - **Вкладка "Проблемы"**: Список ошибок и предупреждений
   - **Вкладка "Экспорт"**: Скачивание отчётов в различных форматах

---

## Форматы отчётов

### TXT (по умолчанию)
Текстовый отчёт со сводной статистикой.

### CSV
- `report_modality.csv` — статистика по модальностям
- `report_nonanon.csv` — список неанонимных исследований
- Полный список исследований с метаданными

### JSON
Полный отчёт со всеми деталями в машиночитаемом формате.

---

## Конфигурация через YAML/JSON

Создайте файл конфигурации:

```yaml
# config.yaml
group_by: study
executor: thread
workers: 4
modality_filter:
  - CT
  - MR
only_labeled: true
exclude_patterns:
  - "**/temp/*"
  - "**/*.bak"
max_depth: 5
follow_symlinks: false
```

Запуск с конфигом:

```bash
python -m src.dcmmetatest.cli /path/to/dataset --config config.yaml
```

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
```

### Pre-commit хуки (опционально)

Создайте файл `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff
    rev: v0.1.0
    hooks:
      - id: ruff
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
```

Установите pre-commit:

```bash
pip install pre-commit
pre-commit install
```

---

## Требования

- Python 3.8+
- Зависимости указаны в `requirements.txt`

### Основные зависимости

- `pydicom` — работа с DICOM-файлами
- `pandas` — обработка табличных данных
- `tqdm` — индикаторы прогресса
- `typer` + `rich` — CLI интерфейс
- `streamlit` — веб-интерфейс
- `plotly` — интерактивные графики

---

## Примеры использования

### Сценарий 1: Быстрая проверка датасета

```bash
# CLI
python -m src.dcmmetatest.cli /data/dicom_dataset --only-labeled --output labeled_report.txt

# Или через веб-интерфейс
streamlit run app.py
# Ввести путь, нажать "Запустить анализ", просмотреть результаты
```

### Сценарий 2: Поиск проблемных файлов

```bash
python -m src.dcmmetatest.cli /data/dicom_dataset --output errors.json
# Открыть errors.json и найти записи с has_label=false
```

### Сценарий 3: Анализ большого архива

```bash
python -m src.dcmmetatest.cli /data/large_archive \
  --group-by study \
  --executor process \
  --workers 8 \
  --modality-filter CT MR \
  --output large_analysis.csv
```

---

## Лицензия

MIT
