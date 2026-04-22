"""Builder канонического registry manifest (schema v1)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .detectors import LABEL_FILE_PATTERNS, detect_label_from_dataset, extract_modality, is_label_json
from .io import iter_all_files
from .utils import read_dicom_header, should_exclude

MANIFEST_VERSION = "1.0"
SCHEMA_REVISION = 1


@dataclass
class ManifestPatient:
    patient_id: str
    patient_name: str | None = None


@dataclass
class ManifestStudy:
    study_uid: str
    patient_id: str
    study_date: str | None = None


@dataclass
class ManifestSeries:
    series_uid: str
    study_uid: str
    modality: str
    description: str | None = None


@dataclass
class ManifestInstance:
    sop_instance_uid: str
    series_uid: str
    file_path: str
    modality: str


@dataclass
class ManifestAnnotation:
    annotation_id: str
    study_uid: str
    series_uid: str | None
    source: str
    file_path: str | None = None


@dataclass
class ManifestSplitAssignment:
    study_uid: str
    split: str
    strategy: str


@dataclass
class ManifestPreprocessingArtifact:
    artifact_id: str
    study_uid: str
    artifact_type: str
    file_path: str


@dataclass
class RegistryManifestV1:
    manifest_version: str = MANIFEST_VERSION
    schema_revision: int = SCHEMA_REVISION
    source_root: str = ""
    generated_at: str = ""
    patients: list[ManifestPatient] = field(default_factory=list)
    studies: list[ManifestStudy] = field(default_factory=list)
    series: list[ManifestSeries] = field(default_factory=list)
    instances: list[ManifestInstance] = field(default_factory=list)
    annotations: list[ManifestAnnotation] = field(default_factory=list)
    split_assignments: list[ManifestSplitAssignment] = field(default_factory=list)
    preprocessing_artifacts: list[ManifestPreprocessingArtifact] = field(default_factory=list)


class RegistryBuilderV1:
    """Собирает канонический manifest из сырых данных."""

    def __init__(self, source_root: str | Path, exclude_patterns: tuple[str, ...] = ()) -> None:
        self.source_root = Path(source_root)
        self.exclude_patterns = exclude_patterns

    def build(self) -> RegistryManifestV1:
        if not self.source_root.exists():
            raise FileNotFoundError(f"Путь не найден: {self.source_root}")

        from datetime import datetime, timezone

        manifest = RegistryManifestV1(
            source_root=str(self.source_root.resolve()),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        patients: dict[str, ManifestPatient] = {}
        studies: dict[str, ManifestStudy] = {}
        series_map: dict[str, ManifestSeries] = {}
        annotations_seen: set[tuple[str, str, str | None]] = set()

        for path in iter_all_files(self.source_root, exclude_patterns=self.exclude_patterns):
            if should_exclude(path, self.exclude_patterns):
                continue

            ds = read_dicom_header(str(path))
            if ds is None:
                self._maybe_collect_annotation(path, manifest, annotations_seen)
                continue

            patient_id = str(ds.get((0x0010, 0x0020), "UNKNOWN_PATIENT")).strip() or "UNKNOWN_PATIENT"
            patient_name_raw = str(ds.get((0x0010, 0x0010), "")).strip() or None
            study_uid = str(ds.get((0x0020, 0x000D), "UNKNOWN_STUDY")).strip() or "UNKNOWN_STUDY"
            series_uid = str(ds.get((0x0020, 0x000E), "UNKNOWN_SERIES")).strip() or "UNKNOWN_SERIES"
            sop_uid = str(ds.get((0x0008, 0x0018), path.as_posix())).strip() or path.as_posix()
            study_date = str(ds.get((0x0008, 0x0020), "")).strip() or None
            modality = extract_modality(ds)
            series_desc = str(ds.get((0x0008, 0x103E), "")).strip() or None

            patients.setdefault(patient_id, ManifestPatient(patient_id=patient_id, patient_name=patient_name_raw))
            studies.setdefault(
                study_uid,
                ManifestStudy(study_uid=study_uid, patient_id=patient_id, study_date=study_date),
            )
            if series_uid not in series_map:
                series_map[series_uid] = ManifestSeries(
                    series_uid=series_uid,
                    study_uid=study_uid,
                    modality=modality,
                    description=series_desc,
                )

            manifest.instances.append(
                ManifestInstance(
                    sop_instance_uid=sop_uid,
                    series_uid=series_uid,
                    file_path=str(path.relative_to(self.source_root)),
                    modality=modality,
                )
            )

            label_sources: set[str] = set()
            detect_label_from_dataset(ds, label_sources)
            for src in sorted(label_sources):
                key = (study_uid, src, series_uid)
                if key in annotations_seen:
                    continue
                annotations_seen.add(key)
                manifest.annotations.append(
                    ManifestAnnotation(
                        annotation_id=f"{study_uid}:{series_uid}:{src}",
                        study_uid=study_uid,
                        series_uid=series_uid,
                        source=src,
                    )
                )

        manifest.patients = sorted(patients.values(), key=lambda p: p.patient_id)
        manifest.studies = sorted(studies.values(), key=lambda s: s.study_uid)
        manifest.series = sorted(series_map.values(), key=lambda s: s.series_uid)
        return manifest

    def _maybe_collect_annotation(
        self,
        path: Path,
        manifest: RegistryManifestV1,
        annotations_seen: set[tuple[str, str, str | None]],
    ) -> None:
        lowered = path.name.lower()
        if is_label_json(path):
            key = ("UNKNOWN_STUDY", "label_json", None)
            if key not in annotations_seen:
                annotations_seen.add(key)
                manifest.annotations.append(
                    ManifestAnnotation(
                        annotation_id=f"unknown:{path.name}",
                        study_uid="UNKNOWN_STUDY",
                        series_uid=None,
                        source="label_json",
                        file_path=str(path.relative_to(self.source_root)),
                    )
                )
            return

        for pattern in LABEL_FILE_PATTERNS:
            if path.match(pattern) or lowered.endswith(pattern.replace("*", "")):
                key = ("UNKNOWN_STUDY", f"file_pattern:{pattern}", None)
                if key in annotations_seen:
                    return
                annotations_seen.add(key)
                manifest.annotations.append(
                    ManifestAnnotation(
                        annotation_id=f"unknown:{path.name}",
                        study_uid="UNKNOWN_STUDY",
                        series_uid=None,
                        source=f"file_pattern:{pattern}",
                        file_path=str(path.relative_to(self.source_root)),
                    )
                )
                return


def export_manifest(manifest: RegistryManifestV1, output_path: str | Path, fmt: str) -> None:
    """Экспортирует manifest в json/csv/parquet."""
    output_path = Path(output_path)
    fmt = fmt.lower()

    if fmt == "json":
        output_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return

    rows = _flatten_manifest(manifest)

    if fmt == "csv":
        if not rows:
            rows = [{"entity_type": "manifest", "manifest_version": manifest.manifest_version}]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
            writer.writeheader()
            writer.writerows(rows)
        return

    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Экспорт parquet требует установленный pyarrow") from exc

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, output_path)
        return

    raise ValueError(f"Неподдерживаемый формат экспорта: {fmt}")


def build_summary_report(manifest: RegistryManifestV1) -> dict[str, Any]:
    """Строит summary report по manifest."""
    modality_counts = Counter(item.modality for item in manifest.instances)
    labeled_studies = {
        annotation.study_uid for annotation in manifest.annotations if annotation.study_uid != "UNKNOWN_STUDY"
    }

    missing_fields = {
        "patients_without_id": sum(1 for p in manifest.patients if not p.patient_id),
        "studies_without_date": sum(1 for s in manifest.studies if not s.study_date),
        "instances_without_modality": sum(1 for i in manifest.instances if not i.modality or i.modality == "UNKNOWN"),
    }

    return {
        "manifest_version": manifest.manifest_version,
        "schema_revision": manifest.schema_revision,
        "volumes": {
            "patients": len(manifest.patients),
            "studies": len(manifest.studies),
            "series": len(manifest.series),
            "instances": len(manifest.instances),
            "annotations": len(manifest.annotations),
            "split_assignments": len(manifest.split_assignments),
            "preprocessing_artifacts": len(manifest.preprocessing_artifacts),
        },
        "modalities": dict(sorted(modality_counts.items())),
        "label_coverage": {
            "studies_with_labels": len(labeled_studies),
            "total_studies": len(manifest.studies),
            "coverage_ratio": round((len(labeled_studies) / len(manifest.studies)), 4) if manifest.studies else 0.0,
        },
        "missing": missing_fields,
    }


def _flatten_manifest(manifest: RegistryManifestV1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping: dict[str, list[Any]] = {
        "patient": manifest.patients,
        "study": manifest.studies,
        "series": manifest.series,
        "instance": manifest.instances,
        "annotation": manifest.annotations,
        "split_assignment": manifest.split_assignments,
        "preprocessing_artifact": manifest.preprocessing_artifacts,
    }

    for entity_type, values in mapping.items():
        for value in values:
            row = asdict(value)
            row["entity_type"] = entity_type
            row["manifest_version"] = manifest.manifest_version
            row["schema_revision"] = manifest.schema_revision
            rows.append(row)
    return rows
