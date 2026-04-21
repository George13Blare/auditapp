# DCM MetaTest Plus

Расширенный анализатор DICOM-датасетов с поддержкой группировки по UID, детекцией разметки и параллельной обработкой.

## Возможности

- **Надёжное определение DICOM**: без строгой привязки к преамбуле "DICM", с fallback через pydicom
- **Детекция разметки**: поддержка RTSTRUCT, SEG, RTSEGANN, SOP Class UID, SeriesDescription, SegmentSequence
- **Группировка исследований**: по директориям или по StudyInstanceUID
- **Параллельная обработка**: выбор типа пула (process/thread) и числа воркеров
- **Контроль обхода**: глубина, следование симлинкам, исключение по шаблонам
- **Авто-детект структуры**: анализ структуры датасета с сохранением схемы (JSON/YAML)
- **Фильтры**: по modality, только размеченные/неанонимные исследования
- **Отчёты**: TXT, CSV, JSON форматы
- **Интерактивный режим**: пошаговая настройка перед запуском
- **Логирование**: настраиваемый уровень, вывод в файл

## Установка

```bash
pip install -r requirements.txt
```

Или для разработки:

```bash
pip install -e ".[dev]"
```

## Использование

### Базовый запуск

```bash
python dcmmetatest_plus.py /path/to/dataset --output report.txt
```

### Группировка по StudyInstanceUID

```bash
python dcmmetatest_plus.py /path/to/dataset --group-by study --executor thread --workers 4
```

### Фильтрация по модальностям

```bash
python dcmmetatest_plus.py /path/to/dataset --modality-filter CT MR --only-labeled
```

### Интерактивный режим

```bash
python dcmmetatest_plus.py /path/to/dataset --interactive
```

### Авто-детект структуры

```bash
python dcmmetatest_plus.py /path/to/dataset --auto-detect-schema --schema-output schema.json
```

### Все опции

```bash
python dcmmetatest_plus.py --help
```

## Форматы отчётов

### TXT (по умолчанию)
Текстовый отчёт со сводной статистикой.

### CSV
- `report_modality.csv` — статистика по модальностям
- `report_nonanon.csv` — список неанонимных исследований

### JSON
Полный отчёт со всеми деталями в машиночитаемом формате.

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
python dcmmetatest_plus.py /path/to/dataset --config config.yaml
```

## Разработка

### Запуск тестов

```bash
pytest tests/ -v
```

### Линтинг

```bash
black dcmmetatest_plus.py
ruff check dcmmetatest_plus.py
mypy dcmmetatest_plus.py
```

## Структура проекта

```
.
├── dcmmetatest_plus.py    # Основной скрипт
├── requirements.txt       # Зависимости
├── pyproject.toml        # Конфигурация проекта
├── README.md             # Документация
├── src/                  # Исходный код (для будущей модуляризации)
└── tests/                # Тесты
```

## Лицензия

MIT
