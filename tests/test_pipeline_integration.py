from __future__ import annotations

import json
from pathlib import Path

import pytest

from dcmmetatest.services.intake import DatasetPathConfig, validate_dataset_path
from dcmmetatest.services.pipeline import (
    ClassMappingArtifact,
    DatasetTaskType,
    LabelDictionaryArtifact,
    PipelineConfig,
    PipelineMode,
    PipelineStage,
    StageConfig,
    run_pipeline,
    save_class_mapping,
    save_label_dictionary,
)


def _create_fake_dataset(root: Path) -> None:
    (root / "study1").mkdir(parents=True, exist_ok=True)
    (root / "study1" / "case_a.dcm").write_bytes(b"fake-dicom")
    (root / "study1" / "case_b.dcm").write_bytes(b"fake-dicom")


@pytest.mark.parametrize(
    "task_type",
    [
        DatasetTaskType.CLASSIFICATION,
        DatasetTaskType.SEGMENTATION,
        DatasetTaskType.DETECTION,
        DatasetTaskType.SLICE_CLASSIFICATION,
    ],
)
def test_pipeline_intake_manifest_normalize_for_all_task_types(tmp_path: Path, task_type: DatasetTaskType):
    dataset = tmp_path / "dataset"
    _create_fake_dataset(dataset)

    intake = validate_dataset_path(DatasetPathConfig(raw_path=str(dataset)))
    assert intake.is_valid is True

    label_dict_path = tmp_path / "label_dictionary.json"
    class_map_path = tmp_path / "class_mapping.json"
    save_label_dictionary(
        LabelDictionaryArtifact(name="labels", labels={"case_a": 1, "default": 0}), label_dict_path
    )
    save_class_mapping(ClassMappingArtifact(name="map", mapping={"1": "lesion", "0": "background"}), class_map_path)

    config = PipelineConfig(
        task_type=task_type,
        mode=PipelineMode.NATIVE_DICOM,
        label_dictionary_path=str(label_dict_path),
        class_mapping_path=str(class_map_path),
        stages={
            PipelineStage.SCAN.value: StageConfig(enabled=True),
            PipelineStage.LINK_LABELS.value: StageConfig(enabled=True),
            PipelineStage.NORMALIZE_PATHS.value: StageConfig(enabled=True),
            PipelineStage.EXTRACT_METADATA.value: StageConfig(enabled=True),
            PipelineStage.SAVE_MANIFEST.value: StageConfig(enabled=True),
        },
        manifest_path="manifest.json",
    )

    result = run_pipeline(intake.resolved_path or str(dataset), config)

    assert result.executed_stages == [
        "scan",
        "link_labels",
        "normalize_paths",
        "extract_metadata",
        "save_manifest",
    ]
    assert result.manifest["task_type"] == task_type.value
    assert result.manifest["metadata"]["instances_total"] == 2
    assert all("/" in item["path"] for item in result.manifest["instances"])

    manifest_path = dataset / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["config_artifacts"]["label_dictionary"]["name"] == "labels"


@pytest.mark.parametrize(
    "mode",
    [
        PipelineMode.NATIVE_DICOM,
        PipelineMode.NIFTI_EXPORT,
        PipelineMode.SLICES_2D_EXPORT,
        PipelineMode.MANIFEST_ONLY,
    ],
)
def test_pipeline_modes_and_stage_toggles(tmp_path: Path, mode: PipelineMode):
    dataset = tmp_path / "dataset"
    _create_fake_dataset(dataset)

    config = PipelineConfig(
        task_type=DatasetTaskType.CLASSIFICATION,
        mode=mode,
        stages={
            PipelineStage.SCAN.value: StageConfig(enabled=True),
            PipelineStage.LINK_LABELS.value: StageConfig(enabled=False),
            PipelineStage.NORMALIZE_PATHS.value: StageConfig(enabled=True),
            PipelineStage.EXTRACT_METADATA.value: StageConfig(enabled=True),
            PipelineStage.SAVE_MANIFEST.value: StageConfig(enabled=False),
        },
    )

    result = run_pipeline(dataset, config)

    assert result.manifest["mode"] == mode.value
    assert "link_labels" not in result.executed_stages
    assert "save_manifest" not in result.executed_stages
    assert result.manifest["metadata"]["labels_linked"] == 0
    assert "manifest_path" not in result.manifest
