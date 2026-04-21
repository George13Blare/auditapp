"""Модели данных для анализатора DICOM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass
class StudyResult:
    """Результат анализа одного исследования."""

    study_key: str
    has_label: bool = False
    non_anon_patients: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    study_path_rep: str = ""
    label_sources: Set[str] = field(default_factory=set)
    series: Dict[str, Dict[str, Union[str, int, List[str]]]] = field(default_factory=dict)
    file_count: int = 0
    patient_ids: List[str] = field(default_factory=list)


@dataclass
class WorkerConfig:
    """Конфигурация воркера для обработки исследований."""

    modality_filter: Optional[Set[str]] = None
    strict: bool = False
    exclude_patterns: Tuple[str, ...] = ()
    detect_series_keywords: Set[str] = field(default_factory=set)
    detect_label_file_patterns: Tuple[str, ...] = ()
    detect_label_json: bool = True
    group_by: str = "dir"
    max_workers: Optional[int] = None
    pool_type: str = "process"
    follow_symlinks: bool = False
    max_depth: Optional[int] = None
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
    modality_stats: Dict[str, int] = field(default_factory=dict)
    label_source_stats: Dict[str, int] = field(default_factory=dict)
    series_max_files: List[Tuple[str, int]] = field(default_factory=list)
    patient_study_counts: Dict[str, int] = field(default_factory=dict)
    empty_folders: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    results: List[StudyResult] = field(default_factory=list)
