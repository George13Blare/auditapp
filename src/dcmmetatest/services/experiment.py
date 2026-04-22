"""Сервисы оркестрации экспериментальных операций."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.dcmmetatest.image_processor import PreprocessingPipelineConfig, preprocess_dataset_pipeline
from src.dcmmetatest.normalizer import NormalizationConfig, analyze_segmentation_masks, normalize_dataset
from src.dcmmetatest.ui import cached_run_analysis, run_preprocessing_pipeline


@dataclass(frozen=True)
class AnalysisRequest:
    dataset_path: str
    config_dict: dict[str, Any]


@dataclass(frozen=True)
class PreprocessSeriesRequest:
    input_series_dir: str
    output_dir: str
    config: PreprocessingPipelineConfig


@dataclass(frozen=True)
class NormalizeRequest:
    source_dir: str
    output_dir: str
    config: NormalizationConfig


@dataclass(frozen=True)
class DatasetPreprocessRequest:
    input_root: str
    output_root: str
    config: PreprocessingPipelineConfig
    max_series: int | None = None


def run_analysis(request: AnalysisRequest):
    return cached_run_analysis(request.dataset_path, request.config_dict)


def run_preprocess_series(request: PreprocessSeriesRequest) -> dict[str, Any]:
    return run_preprocessing_pipeline(request.input_series_dir, request.output_dir, request.config)


def run_normalize(request: NormalizeRequest):
    return normalize_dataset(request.source_dir, request.output_dir, request.config)


def run_segmentation_analysis(source_dir: str) -> dict[str, Any]:
    return analyze_segmentation_masks(source_dir)


def run_dataset_preprocessing(request: DatasetPreprocessRequest) -> dict[str, Any]:
    return preprocess_dataset_pipeline(
        input_root=Path(request.input_root),
        output_root=Path(request.output_root),
        config=request.config,
        max_series=request.max_series,
    )
