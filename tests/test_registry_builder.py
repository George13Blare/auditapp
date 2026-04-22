"""Тесты registry builder v1."""

from __future__ import annotations

import json


def test_manifest_schema_versions(tmp_path):
    from dcmmetatest.registry import MANIFEST_VERSION, SCHEMA_REVISION, RegistryBuilderV1

    dcm = tmp_path / "scan1.dcm"
    dcm.write_bytes(b"not-a-dicom")

    manifest = RegistryBuilderV1(tmp_path).build()
    assert manifest.manifest_version == MANIFEST_VERSION
    assert manifest.schema_revision == SCHEMA_REVISION


def test_export_json_csv_and_summary(tmp_path):
    from dcmmetatest.registry import (
        ManifestInstance,
        ManifestPatient,
        ManifestSeries,
        ManifestStudy,
        RegistryManifestV1,
        build_summary_report,
        export_manifest,
    )

    manifest = RegistryManifestV1(source_root=str(tmp_path), generated_at="2026-01-01T00:00:00Z")
    manifest.patients.append(ManifestPatient(patient_id="p1"))
    manifest.studies.append(ManifestStudy(study_uid="s1", patient_id="p1"))
    manifest.series.append(ManifestSeries(series_uid="se1", study_uid="s1", modality="CT"))
    manifest.instances.append(
        ManifestInstance(sop_instance_uid="i1", series_uid="se1", file_path="scan1.dcm", modality="CT")
    )

    out_json = tmp_path / "manifest.json"
    out_csv = tmp_path / "manifest.csv"
    export_manifest(manifest, out_json, "json")
    export_manifest(manifest, out_csv, "csv")

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["manifest_version"] == "1.0"
    assert out_csv.exists()

    summary = build_summary_report(manifest)
    assert summary["volumes"]["patients"] == 1
    assert summary["modalities"]["CT"] == 1


def test_parquet_export_requires_pyarrow(tmp_path):
    from dcmmetatest.registry import RegistryManifestV1, export_manifest

    manifest = RegistryManifestV1(source_root=str(tmp_path), generated_at="2026-01-01T00:00:00Z")
    out_path = tmp_path / "manifest.parquet"

    try:
        export_manifest(manifest, out_path, "parquet")
    except RuntimeError as exc:
        assert "pyarrow" in str(exc)
    else:
        assert out_path.exists()
