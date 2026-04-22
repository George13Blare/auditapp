"""Конфигурируемый pipeline intake/manifest/normalization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PipelineStage(str, Enum):
    SCAN = "scan"
    LINK_LABELS = "link_labels"
    NORMALIZE_PATHS = "normalize_paths"
    EXTRACT_METADATA = "extract_metadata"
    SAVE_MANIFEST = "save_manifest"


class DatasetTaskType(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    SLICE_CLASSIFICATION = "slice_classification"


class PipelineMode(str, Enum):
    NATIVE_DICOM = "native_dicom"
    NIFTI_EXPORT = "nifti_export"
    SLICES_2D_EXPORT = "slices_2d_export"
    MANIFEST_ONLY = "manifest_only"


@dataclass(frozen=True)
class StageConfig:
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelDictionaryArtifact:
    """Явный конфиг-артефакт словаря лейблов."""

    name: str
    labels: dict[str, int]


@dataclass(frozen=True)
class ClassMappingArtifact:
    """Явный конфиг-артефакт mapping классов."""

    name: str
    mapping: dict[str, str]


@dataclass(frozen=True)
class PipelineConfig:
    task_type: DatasetTaskType
    mode: PipelineMode
    stages: dict[str, StageConfig] = field(default_factory=dict)
    manifest_path: str = "manifest_pipeline.json"
    label_dictionary_path: str | None = None
    class_mapping_path: str | None = None

    def stage_enabled(self, stage: PipelineStage) -> bool:
        return self.stages.get(stage.value, StageConfig()).enabled

    def stage_params(self, stage: PipelineStage) -> dict[str, Any]:
        return self.stages.get(stage.value, StageConfig()).params


@dataclass
class PipelineResult:
    manifest: dict[str, Any]
    executed_stages: list[str]


def save_label_dictionary(artifact: LabelDictionaryArtifact, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(asdict(artifact), ensure_ascii=False, indent=2), encoding="utf-8")


def save_class_mapping(artifact: ClassMappingArtifact, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(asdict(artifact), ensure_ascii=False, indent=2), encoding="utf-8")


def load_label_dictionary(path: str | Path) -> LabelDictionaryArtifact:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LabelDictionaryArtifact(name=data["name"], labels=dict(data.get("labels", {})))


def load_class_mapping(path: str | Path) -> ClassMappingArtifact:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClassMappingArtifact(name=data["name"], mapping=dict(data.get("mapping", {})))


def run_pipeline(dataset_root: str | Path, config: PipelineConfig) -> PipelineResult:
    root = Path(dataset_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    manifest: dict[str, Any] = {
        "dataset_root": str(root.resolve()),
        "task_type": config.task_type.value,
        "mode": config.mode.value,
        "instances": [],
        "label_links": [],
        "metadata": {},
        "config_artifacts": {},
    }
    executed: list[str] = []

    # 1) scan
    if config.stage_enabled(PipelineStage.SCAN):
        suffixes = tuple(config.stage_params(PipelineStage.SCAN).get("dicom_suffixes", [".dcm", ".dicom", ""]))
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in suffixes:
                manifest["instances"].append({"path": str(path), "name": path.name})
        executed.append(PipelineStage.SCAN.value)

    # 2) link_labels
    if config.stage_enabled(PipelineStage.LINK_LABELS):
        label_dict = _maybe_load_label_dict(config)
        class_mapping = _maybe_load_class_mapping(config)

        for item in manifest["instances"]:
            stem = Path(item["name"]).stem
            label_id = label_dict.labels.get(stem, label_dict.labels.get("default")) if label_dict else None
            mapped = class_mapping.mapping.get(str(label_id), str(label_id)) if class_mapping and label_id else label_id
            if label_id is not None:
                manifest["label_links"].append(
                    {
                        "instance_path": item["path"],
                        "label_id": label_id,
                        "mapped_class": mapped,
                    }
                )

        if label_dict:
            manifest["config_artifacts"]["label_dictionary"] = asdict(label_dict)
        if class_mapping:
            manifest["config_artifacts"]["class_mapping"] = asdict(class_mapping)
        executed.append(PipelineStage.LINK_LABELS.value)

    # 3) normalize_paths
    if config.stage_enabled(PipelineStage.NORMALIZE_PATHS):
        for item in manifest["instances"]:
            item["path"] = Path(item["path"]).resolve().as_posix()
        for item in manifest["label_links"]:
            item["instance_path"] = Path(item["instance_path"]).resolve().as_posix()
        executed.append(PipelineStage.NORMALIZE_PATHS.value)

    # 4) extract_metadata
    if config.stage_enabled(PipelineStage.EXTRACT_METADATA):
        manifest["metadata"] = {
            "instances_total": len(manifest["instances"]),
            "labels_linked": len(manifest["label_links"]),
            "task_type": config.task_type.value,
            "mode": config.mode.value,
        }
        executed.append(PipelineStage.EXTRACT_METADATA.value)

    # 5) save_manifest
    if config.stage_enabled(PipelineStage.SAVE_MANIFEST):
        manifest_path = root / config.manifest_path
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        executed.append(PipelineStage.SAVE_MANIFEST.value)

    return PipelineResult(manifest=manifest, executed_stages=executed)


def _maybe_load_label_dict(config: PipelineConfig) -> LabelDictionaryArtifact | None:
    if not config.label_dictionary_path:
        return None
    path = Path(config.label_dictionary_path)
    if not path.exists():
        return None
    return load_label_dictionary(path)


def _maybe_load_class_mapping(config: PipelineConfig) -> ClassMappingArtifact | None:
    if not config.class_mapping_path:
        return None
    path = Path(config.class_mapping_path)
    if not path.exists():
        return None
    return load_class_mapping(path)
