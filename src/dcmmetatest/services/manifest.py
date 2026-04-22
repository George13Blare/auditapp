"""Сервисы формирования артефактов отчётов (manifest/export)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from src.dcmmetatest.models import AnalysisReport


@dataclass(frozen=True)
class ReportManifestConfig:
    """Параметры экспорта отчёта."""

    include_errors_limit: int = 10


@dataclass(frozen=True)
class ReportManifestArtifact:
    """Сериализованные артефакты отчёта."""

    json_payload: str
    text_payload: str


def build_report_manifest(
    report: AnalysisReport, metrics: dict, config: ReportManifestConfig
) -> ReportManifestArtifact:
    report_dict = asdict(report)
    json_payload = json.dumps(report_dict, indent=2, ensure_ascii=False, default=str)

    lines = [
        "=" * 60,
        "DICOM ANALYSIS REPORT",
        "=" * 60,
        f"Total Studies: {metrics['total_studies']}",
        f"Total Files: {metrics['total_files']}",
        f"Unique Patients: {metrics['unique_patients']}",
        f"Labeled: {metrics['labeled_percent']}%",
        f"Non-Anonymized: {metrics['non_anon_percent']}%",
        "",
        "Modality Stats:",
    ]
    for modality, count in report.modality_stats.items():
        lines.append(f"  {modality}: {count}")

    if report.errors:
        lines.append("")
        lines.append(f"Errors ({len(report.errors)}):")
        for err in report.errors[: config.include_errors_limit]:
            lines.append(f"  - {err}")

    text_payload = "\n".join(lines)
    return ReportManifestArtifact(json_payload=json_payload, text_payload=text_payload)
