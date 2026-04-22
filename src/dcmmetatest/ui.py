"""Модуль визуального интерфейса (Streamlit) для анализатора DICOM."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit.runtime.caching import cache_data

from .image_processor import (
    AugmentationConfig,
    PreprocessingPipelineConfig,
    preprocess_dicom_series_pipeline,
)
from .models import AnalysisReport, WorkerConfig

logger = logging.getLogger(__name__)

__all__ = [
    "AugmentationConfig",
    "PreprocessingPipelineConfig",
    "run_preprocessing_pipeline",
    "convert_report_to_dataframe",
    "create_modality_pie_chart",
    "create_study_timeline",
    "create_label_source_bar_chart",
    "create_quality_metrics_cards",
    "create_study_date_timeline",
    "create_age_distribution_chart",
    "cached_run_analysis",
    "validate_folder_path",
]


def run_preprocessing_pipeline(
    input_series_dir: str,
    output_dir: str,
    config: PreprocessingPipelineConfig,
) -> dict[str, Any]:
    """
    Запускает preprocessing pipeline для одной серии.
    """
    input_path = Path(input_series_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = preprocess_dicom_series_pipeline(input_path, output_path, config)
    return {
        "input_series_dir": str(input_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "export_format": config.export_format,
        "files_saved": stats.get("files_saved", 0),
        "errors": stats.get("errors", []),
    }


def convert_report_to_dataframe(report: AnalysisReport) -> pd.DataFrame:
    """
    Преобразует отчёт анализа в DataFrame для отображения в таблице.

    Args:
        report: Отчёт анализа

    Returns:
        DataFrame с данными исследований
    """
    rows = []
    for study in report.results:
        row = {
            "Study Key": study.study_key,
            "Patient ID": ", ".join(study.patient_ids) if study.patient_ids else "N/A",
            "File Count": study.file_count,
            "Has Label": "✅" if study.has_label else "❌",
            "Modalities": ", ".join(study.modalities) if study.modalities else "N/A",
            "Series Count": len(study.series),
            "Non-Anon Patient": "⚠️" if study.non_anon_patients else "✅",
            "Non-Anon Files": len(study.non_anon_files),
            "Label Sources": ", ".join(study.label_sources) if study.label_sources else "-",
            "Study Date": study.study_date if study.study_date else "N/A",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_modality_pie_chart(report: AnalysisReport) -> go.Figure:
    """
    Создаёт круговую диаграмму распределения модальностей.

    Args:
        report: Отчёт анализа

    Returns:
        Plotly Figure
    """
    if not report.modality_stats:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных о модальностях", showarrow=False)
        return fig

    labels = list(report.modality_stats.keys())
    values = list(report.modality_stats.values())

    fig = px.pie(
        values=values,
        names=labels,
        title="Распределение по модальностям (DICOM Modality)",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400)
    return fig


def create_study_timeline(report: AnalysisReport) -> go.Figure | None:
    """
    Создаёт временную шкалу исследований (если доступны даты).

    Примечание: В текущей версии модели не хранят даты, поэтому заглушка.
    """
    # TODO: Добавить извлечение дат из StudyDate при сборе статистики
    fig = go.Figure()
    fig.add_annotation(text="Временная шкала будет доступна после добавления поля StudyDate", showarrow=False)
    return fig


def create_label_source_bar_chart(report: AnalysisReport) -> go.Figure:
    """
    Создаёт столбчатую диаграмму источников разметки.

    Args:
        report: Отчёт анализа

    Returns:
        Plotly Figure
    """
    if not report.label_source_stats:
        fig = go.Figure()
        fig.add_annotation(text="Источники разметки не обнаружены", showarrow=False)
        return fig

    labels = list(report.label_source_stats.keys())
    values = list(report.label_source_stats.values())

    fig = px.bar(
        x=labels,
        y=values,
        title="Источники определения разметки",
        labels={"x": "Источник", "y": "Количество исследований"},
        color=values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_quality_metrics_cards(report: AnalysisReport) -> dict[str, Any]:
    """
    Создаёт метрики качества для отображения в карточках.

    Args:
        report: Отчёт анализа

    Returns:
        Словарь с метриками
    """
    total_studies = report.processed_studies or 1  # Избегаем деления на ноль

    return {
        "total_studies": report.processed_studies,
        "total_files": report.total_dicom_files,
        "unique_patients": report.unique_patients,
        "labeled_percent": round((report.labeled_studies / total_studies) * 100, 1),
        "non_anon_percent": round((report.non_anon_studies / total_studies) * 100, 1),
        "avg_files_per_study": round(report.total_dicom_files / total_studies, 1),
        "empty_folders": len(report.empty_folders),
        "errors_count": len(report.errors),
        "non_anon_files_total": sum(len(r.non_anon_files) for r in report.results),
        "quality_issues_total": sum(report.quality_issues.values()) if report.quality_issues else 0,
    }


def create_study_date_timeline(report: AnalysisReport) -> go.Figure | None:
    """
    Создаёт временную шкалу исследований.

    Args:
        report: Отчёт анализа

    Returns:
        Plotly Figure или None
    """
    if not report.study_date_distribution:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных о датах исследований", showarrow=False)
        return fig

    dates = sorted(report.study_date_distribution.keys())
    counts = [report.study_date_distribution[d] for d in dates]

    fig = px.line(
        x=dates,
        y=counts,
        title="Распределение исследований по датам",
        labels={"x": "Дата", "y": "Количество исследований"},
        markers=True,
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig


def create_age_distribution_chart(report: AnalysisReport) -> go.Figure | None:
    """
    Создаёт диаграмму распределения по возрастным группам.

    Args:
        report: Отчёт анализа

    Returns:
        Plotly Figure или None
    """
    if not report.age_distribution:
        fig = go.Figure()
        fig.add_annotation(text="Нет данных о возрасте пациентов", showarrow=False)
        return fig

    age_groups = list(report.age_distribution.keys())
    counts = list(report.age_distribution.values())

    fig = px.bar(
        x=age_groups,
        y=counts,
        title="Распределение пациентов по возрастным группам",
        labels={"x": "Возрастная группа", "y": "Количество пациентов"},
        color=counts,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


@cache_data(ttl=3600)
def cached_run_analysis(folder_path: str, config_dict: dict) -> AnalysisReport:
    """
    Кэшированная версия запуска анализа для ускорения повторных запросов.

    Args:
        folder_path: Путь к директории
        config_dict: Конфигурация в виде словаря

    Returns:
        Отчёт анализа
    """
    from .analyzer import run_analysis

    config = WorkerConfig(**config_dict)
    return run_analysis(folder_path, config, debug=False)


def validate_folder_path(path_str: str) -> tuple[bool, str]:
    """
    Проверяет существование и доступность пути.

    Args:
        path_str: Строка пути

    Returns:
        Кортеж (успех, сообщение)
    """
    if not path_str:
        return False, "Путь не указан"

    path = Path(path_str)

    if not path.exists():
        return False, f"Путь не существует: {path_str}"

    if not path.is_dir():
        return False, f"Это не директория: {path_str}"

    if not path.is_absolute():
        path = path.resolve()

    return True, str(path)
