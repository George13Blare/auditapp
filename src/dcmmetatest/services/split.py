"""Сервисы split-оркестрации для UI."""

from __future__ import annotations

from dataclasses import dataclass

from src.dcmmetatest.normalizer import SplitConfig, split_dataset


@dataclass(frozen=True)
class SplitRequest:
    source_dir: str
    output_dir: str
    config: SplitConfig


@dataclass(frozen=True)
class SplitArtifact:
    train_samples: int
    val_samples: int
    test_samples: int
    train_patients: int
    val_patients: int
    test_patients: int
    split_manifest: dict | None


def run_split(request: SplitRequest) -> SplitArtifact:
    stats = split_dataset(request.source_dir, request.output_dir, request.config)
    return SplitArtifact(
        train_samples=stats.train_samples,
        val_samples=stats.val_samples,
        test_samples=stats.test_samples,
        train_patients=stats.train_patients,
        val_patients=stats.val_patients,
        test_patients=stats.test_patients,
        split_manifest=stats.split_manifest,
    )
