"""Модели данных для анализатора DICOM."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StudyResult:
    """Результат анализа одного исследования."""

    study_key: str
    has_label: bool = False
    non_anon_patients: list[str] = field(default_factory=list)
    non_anon_files: list[str] = field(default_factory=list)  # Файлы с неанонимизированными данными
    modalities: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    study_path_rep: str = ""
    label_sources: set[str] = field(default_factory=set)
    series: dict[str, dict[str, str | int | list[str]]] = field(default_factory=dict)
    file_count: int = 0
    patient_ids: list[str] = field(default_factory=list)
    study_date: str = ""  # Дата исследования для аналитики


@dataclass
class WorkerConfig:
    """Конфигурация воркера для обработки исследований."""

    modality_filter: set[str] | None = None
    strict: bool = False
    exclude_patterns: tuple[str, ...] = ()
    detect_series_keywords: set[str] = field(default_factory=set)
    detect_label_file_patterns: tuple[str, ...] = ()
    detect_label_json: bool = True
    group_by: str = "dir"
    max_workers: int | None = None
    pool_type: str = "process"
    follow_symlinks: bool = False
    max_depth: int | None = None
    list_empty: bool = False
    only_labeled: bool = False
    only_non_anon: bool = False
    min_files: int = 0
    show_progress: bool = True


@dataclass
class AnalysisReport:
    """Сводный отчёт по анализу датасета."""

    total_studies: int = 0
    processed_studies: int = 0
    filtered_out_studies: int = 0
    labeled_studies: int = 0
    non_anon_studies: int = 0
    total_dicom_files: int = 0
    unique_patients: int = 0
    modality_stats: dict[str, int] = field(default_factory=dict)
    label_source_stats: dict[str, int] = field(default_factory=dict)
    series_max_files: list[tuple[str, int]] = field(default_factory=list)
    patient_study_counts: dict[str, int] = field(default_factory=dict)
    empty_folders: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    results: list[StudyResult] = field(default_factory=list)
    # Расширенная аналитика
    study_date_distribution: dict[str, int] = field(default_factory=dict)  # Распределение по датам
    age_distribution: dict[str, int] = field(default_factory=dict)  # Распределение по возрастным группам
    quality_issues: dict[str, int] = field(default_factory=dict)  # Проблемы качества (битые файлы и т.д.)
