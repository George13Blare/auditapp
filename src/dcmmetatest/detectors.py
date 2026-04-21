"""Детекторы разметки и DICOM-файлов."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Tuple, Optional

try:
    import pydicom
    from pydicom.dataset import Dataset
    HAS_PYDICOM = True
except ImportError:
    pydicom = None  # type: ignore
    HAS_PYDICOM = False

import logging

logger = logging.getLogger(__name__)

# UID классов SOP, указывающих на разметку
LABEL_SOP_CLASS_UIDS: Set[str] = {
    "1.2.840.10008.5.1.4.1.1.481.3",  # RT Structure Set Storage
    "1.2.840.10008.5.1.4.1.1.66.4",   # Segmentation Storage
    "1.2.840.10008.5.1.4.1.1.130",    # Surface Segmentation Storage
    "1.2.840.10008.5.1.4.1.1.481.5",  # RT Dose Storage
    "1.2.840.10008.5.1.4.1.1.481.2",  # RT Plan Storage
}

# Ключевые слова для поиска в описании серии
LABEL_SERIES_KEYWORDS: Set[str] = {
    "seg", "mask", "label", "roi", "annotation", "dose", "structure", "contour", "markup"
}

# Шаблоны имён файлов разметки
LABEL_FILE_PATTERNS: Tuple[str, ...] = (
    "*.nii", "*.nii.gz", "*.nrrd", "*.seg.nrrd", "*.mha", "*.mhd",
    "*mask*.png", "*mask*.jpg", "*mask*.tif", "*mask*.tiff",
    "*label*.png", "*label*.jpg", "*label*.tif", "*label*.tiff",
    "*.npz", "*.npy", "*.h5"
)


def detect_label_from_dataset(ds: "Dataset", sources: Set[str]) -> None:
    """
    Обнаруживает признаки разметки в DICOM-датасете.
    
    Args:
        ds: DICOM dataset
        sources: Множество для добавления обнаруженных источников разметки
    """
    if not HAS_PYDICOM:
        return
    
    modality = str(ds.get((0x0008, 0x0060), "")).strip().upper()
    if modality in {"RTSTRUCT", "SEG", "RTSEGANN"}:
        sources.add(f"dicom_modality:{modality}")

    sop = str(ds.get((0x0008, 0x0016), "")).strip()
    if sop in LABEL_SOP_CLASS_UIDS:
        sources.add(f"sop_class:{sop}")

    if hasattr(ds, "SegmentSequence"):
        sources.add("segment_sequence")

    series_desc = str(ds.get((0x0008, 0x103E), "")).lower()
    if series_desc:
        for keyword in LABEL_SERIES_KEYWORDS:
            if keyword in series_desc:
                sources.add(f"series_description:{keyword}")
                break

    content_label = str(ds.get("ContentLabel", "")).lower()
    if content_label in {"segmentation", "mask", "roi"}:
        sources.add("content_label")


def is_dicom_file(path: Path) -> bool:
    """
    Проверяет, является ли файл DICOM.
    
    Args:
        path: Путь к файлу
        
    Returns:
        True если файл является DICOM
    """
    if not HAS_PYDICOM:
        return False
    
    path = Path(path)
    if not path.is_file():
        return False
    
    # Быстрая проверка преамбулы
    try:
        with open(path, "rb") as f:
            head = f.read(132)
            if len(head) >= 132 and head[128:132] == b"DICM":
                return True
    except Exception:
        pass
    
    # Fallback через pydicom
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def is_label_json(path: Path) -> bool:
    """
    Проверяет, является ли JSON-файл файлом разметки.
    
    Args:
        path: Путь к JSON-файлу
        
    Returns:
        True если файл содержит структуру разметки
    """
    path = Path(path)
    if not path.suffix.lower() == ".json":
        return False
    
    # Проверка по имени
    name_lower = path.name.lower()
    if any(kw in name_lower for kw in ["seg", "mask", "label", "annotation"]):
        return True
    
    # Проверка по структуре
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            label_keys = {"annotations", "segments", "labels", "regions", "roi"}
            if any(key in data for key in label_keys):
                return True
    except Exception:
        pass
    
    return False


def extract_modality(ds: Optional["Dataset"]) -> Optional[str]:
    """
    Извлекает модальность из DICOM-датасета.
    
    Args:
        ds: DICOM dataset или None
        
    Returns:
        Строка модальности или None
    """
    if ds is None:
        return None
    
    try:
        modality_tag = ds.get((0x0008, 0x0060))
        if modality_tag:
            return str(modality_tag.value).strip().upper()
    except Exception:
        pass
    
    return "UNKNOWN"
