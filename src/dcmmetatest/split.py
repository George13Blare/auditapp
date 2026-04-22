"""Utilities for splitting normalized datasets into train/val/test subsets."""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset split."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_by: str = "patient"
    ensure_balance: bool = True
    min_files_per_study: int = 1
    require_segmentation: bool = False
    modalities_to_include: list[str] = field(default_factory=list)
    output_dir: str = ""
    create_manifest: bool = True
    seed: int = 42


@dataclass
class SplitStats:
    """Split statistics."""

    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_patients: int = 0
    val_patients: int = 0
    test_patients: int = 0
    split_manifest: dict = field(default_factory=dict)


def split_dataset(normalized_path: str, output_path: str, config: SplitConfig) -> SplitStats:
    """Split normalized dataset into train/val/test directories."""
    stats = SplitStats()
    input_dir = Path(normalized_path)
    output_dir = Path(output_path)

    random.seed(config.seed)

    patients = [item for item in input_dir.iterdir() if item.is_dir() and item.name.startswith("patient_")]
    if not patients:
        logger.warning("Пациенты не найдены. Проверьте структуру датасета.")
        return stats

    random.shuffle(patients)

    n_total = len(patients)
    n_train = max(1, int(n_total * config.train_ratio))
    n_val = max(1, int(n_total * config.val_ratio))

    train_patients = patients[:n_train]
    val_patients = patients[n_train : n_train + n_val]
    test_patients = patients[n_train + n_val :]

    stats.train_patients = len(train_patients)
    stats.val_patients = len(val_patients)
    stats.test_patients = len(test_patients)

    def copy_split(source_patients: list[Path], split_name: str) -> int:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        sample_count = 0
        for patient_dir in source_patients:
            try:
                dst_patient_dir = split_dir / patient_dir.name
                shutil.copytree(patient_dir, dst_patient_dir)
                sample_count += 1
            except Exception as e:
                logger.error(f"Ошибка копирования {patient_dir}: {e}")

        return sample_count

    stats.train_samples = copy_split(train_patients, "train")
    stats.val_samples = copy_split(val_patients, "val")
    stats.test_samples = copy_split(test_patients, "test")

    if config.create_manifest:
        manifest = {
            "config": {
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
                "seed": config.seed,
            },
            "stats": {
                "train": {"patients": stats.train_patients, "samples": stats.train_samples},
                "val": {"patients": stats.val_patients, "samples": stats.val_samples},
                "test": {"patients": stats.test_patients, "samples": stats.test_samples},
            },
            "train_patients": [p.name for p in train_patients],
            "val_patients": [p.name for p in val_patients],
            "test_patients": [p.name for p in test_patients],
        }

        manifest_file = output_dir / "split_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        stats.split_manifest = manifest

    return stats
