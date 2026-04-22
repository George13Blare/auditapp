"""Валидация и поиск аномалий в медицинских датасетах."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def _find_duplicate_values(items: list[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Группирует дубликаты по ключу.

    Args:
        items: Список пар (key, payload)

    Returns:
        Словарь key -> список payload, только для key с 2+ значениями
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, payload in items:
        grouped[key].append(payload)
    return {key: payloads for key, payloads in grouped.items() if len(payloads) > 1}


def scan_dataset_anomalies(root_path: str, max_files: int = 5000) -> dict[str, Any]:
    """
    Выполняет базовый поиск аномалий:
    - битые DICOM-файлы
    - пустые SEG-маски
    - дубликаты SOPInstanceUID
    """
    root = Path(root_path)
    result: dict[str, Any] = {
        "scanned_files": 0,
        "broken_files": [],
        "empty_seg_masks": [],
        "duplicate_sop_instance_uid": {},
        "errors": [],
    }

    if not root.exists() or not root.is_dir():
        result["errors"].append(f"Путь не существует или не является директорией: {root_path}")
        return result

    try:
        import pydicom
    except Exception as exc:  # pragma: no cover - зависит от окружения
        result["errors"].append(f"Требуется pydicom для сканирования аномалий: {exc}")
        return result

    candidate_files = list(root.rglob("*.dcm")) + list(root.rglob("*.dicom"))
    sop_pairs: list[tuple[str, str]] = []

    for file_path in candidate_files[:max_files]:
        result["scanned_files"] += 1
        try:
            ds = pydicom.dcmread(str(file_path), force=True)
            sop_uid = getattr(ds, "SOPInstanceUID", None)
            if sop_uid:
                sop_pairs.append((str(sop_uid), str(file_path)))

            if getattr(ds, "Modality", "") == "SEG":
                try:
                    pixel_data = ds.pixel_array
                    if getattr(pixel_data, "size", 0) > 0 and float(pixel_data.max()) == 0.0:
                        result["empty_seg_masks"].append(str(file_path))
                except Exception:
                    result["broken_files"].append(str(file_path))
        except Exception:
            result["broken_files"].append(str(file_path))

    result["duplicate_sop_instance_uid"] = _find_duplicate_values(sop_pairs)
    return result
