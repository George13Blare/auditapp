
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dcmmetatest_plus.py — расширенная версия анализатора DICOM.

Что добавлено по сравнению с исходным скриптом:
- Более надёжное определение DICOM (без строгой привязки к 'DICM' на смещении 128, с fallback через pydicom).
- Поддержка метки разметки по модальности DICOM SEG (сохранена обратная совместимость с RTSTRUCT и RTSEGANN).
- Более аккуратная проверка JSON-файлов разметки (по имени и/или структуре).
- Опция группировки «исследований» по StudyInstanceUID (вместо «одна директория = одно исследование»).
- Защита от деления на ноль в отчёте.
- Управляемая параллелизация: выбор типа пула (process/thread) и числа воркеров.
- Контроль обхода: глубина, следование симлинкам, опциональный вывод пустых папок.
- Улучшенные сообщения об ошибках и диагностика.
- Форматы отчёта сохранены (TXT/CSV/JSON), имена флагов остаются совместимыми.
- Кеширование заголовков DICOM для снижения количества повторных чтений.
- Расширенная детекция разметки: анализ SOPClassUID, SeriesDescription, SegmentSequence,
  поддержка NIfTI/NRRD/MHA и масок по шаблонам файлов.
- Более информативные отчёты: статистика по сериям, среднее количество файлов, сводка по пациентам,
  подсчёт источников определения разметки.
- Новый CLI-функционал: фильтры по modality, выбор только размеченных/неанонимных исследований,
  исключение директорий по шаблонам, управление прогресс-баром, батчинг заданий и конфигурация через YAML/JSON.
- Настройка логирования (уровень, файл), строгий режим остановки при ошибках и расширяемые шаблоны обработки.

ВАЖНО: Логику анонимизации мы НЕ меняли — она берётся из исходного файла dcmmetatest.py,
и вызывается как есть (через import исходной функции check_dicom_anonymization).
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
import logging
import fnmatch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Set, Union
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional
    yaml = None

import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

logger = logging.getLogger(__name__)

LABEL_SOP_CLASS_UIDS: Set[str] = {
    "1.2.840.10008.5.1.4.1.1.481.3",  # RT Structure Set Storage
    "1.2.840.10008.5.1.4.1.1.66.4",   # Segmentation Storage
    "1.2.840.10008.5.1.4.1.1.130",    # Surface Segmentation Storage
    "1.2.840.10008.5.1.4.1.1.481.5",  # RT Dose Storage
    "1.2.840.10008.5.1.4.1.1.481.2",  # RT Plan Storage
}

LABEL_SERIES_KEYWORDS: Set[str] = {
    "seg", "mask", "label", "roi", "annotation", "dose", "structure", "contour", "markup"
}

LABEL_FILE_PATTERNS: Tuple[str, ...] = (
    "*.nii", "*.nii.gz", "*.nrrd", "*.seg.nrrd", "*.mha", "*.mhd",
    "*mask*.png", "*mask*.jpg", "*mask*.tif", "*mask*.tiff",
    "*label*.png", "*label*.jpg", "*label*.tif", "*label*.tiff",
    "*.npz", "*.npy", "*.h5"
)

# --- ВСТАВЛЕНА ИСХОДНАЯ ФУНКЦИЯ АНОНИМИЗАЦИИ (без изменений) ---
def check_dicom_anonymization(dicom_file):
    """Проверяет только PatientName на анонимность и возвращает Modality."""
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        modality_tag = ds.get((0x0008, 0x0060))
        modality = str(modality_tag.value).strip().upper() if modality_tag else "UNKNOWN"
        anonymous_values = {
            "anonymous",
            "anon",
            "anonymized",
            "unknown",
            "na",
            "n/a",
            "none",
            "xxxx",
            "yyyy",
            "zzzz",
            "анонимно",
            "аноним",
            "неизвестно",
            "ai_test",
            "test",
            "demo",
        }
        if (0x10, 0x10) in ds:
            patient_name = str(ds[0x10, 0x10].value or "").lower().strip()
            if patient_name and not any(anon in patient_name for anon in anonymous_values):
                return False, modality, f"PatientName: '{ds[0x10, 0x10].value}'"
        return True, modality, None
    except Exception as e:
        return False, "ERROR", f"Ошибка чтения: {str(e)}"


# ---------------- Вспомогательные функции ----------------

def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    level = level.upper()
    numeric_level = getattr(logging, level, logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


@lru_cache(maxsize=2048)
def read_dicom_header(path: str) -> Optional[pydicom.dataset.Dataset]:
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception as exc:
        logger.debug("Не удалось прочитать DICOM %s: %s", path, exc)
        return None


def should_exclude(path: Path, patterns: Tuple[str, ...]) -> bool:
    if not patterns:
        return False
    path_str = str(path)
    for pattern in patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True
    return False


def batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def is_dicom_file(path: Union[str, Path]) -> bool:
    """
    Надёжная проверка, является ли файл DICOM.
    Логика:
      1) Если доступна pydicom.misc.is_dicom — используем её.
      2) Иначе: быстрый пролог (проверка преамбулы 'DICM' при наличии).
      3) Fallback: пробуем прочитать заголовок через pydicom.dcmread(..., force=True, stop_before_pixels=True).
    """
    path = str(path)
    if not os.path.isfile(path):
        return False
    try:
        # Вариант 1: официальная функция (есть не во всех версиях pydicom)
        try:
            from pydicom.misc import is_dicom as _is_dicom
            return bool(_is_dicom(path))
        except Exception:
            pass

        # Вариант 2: пролог
        try:
            with open(path, "rb") as f:
                head = f.read(132)
                if len(head) >= 132 and head[128:132] == b"DICM":
                    return True
        except Exception:
            # игнорируем, перейдём к fallback
            pass

        # Вариант 3: грубая попытка распарсить без пикселей
        try:
            ds = read_dicom_header(path)
            return ds is not None
        except (InvalidDicomError, Exception):
            return False
    except Exception:
        return False


def is_label_json(path: Union[str, Path]) -> bool:
    """
    Эвристика для распознавания JSON-файлов разметки.
    - Сначала по имени файла (частые варианты).
    - Затем быстрая проверка структуры (ограничиваем размер).
    """
    p = Path(path)
    name = p.name.lower()
    # Часто встречающиеся имена файлов разметки
    name_hits = {"labels.json", "instances.json", "annotations.json", "segmentations.json"}
    if name in name_hits:
        return True

    # Ограничим размер, чтобы не заглатывать гигабайты случайного JSON
    try:
        size = p.stat().st_size
        if size > 20 * 1024 * 1024:  # 20 MB
            return False
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            # Пробуем распарсить маленькие/средние JSON
            data = json.load(f)
        if isinstance(data, dict):
            keys = {k.lower() for k in data.keys()}
            # Наиболее распространённые поля в разметке:
            if keys & {"annotations", "labels", "instances", "objects", "rois", "regions", "shapes"}:
                return True
        if isinstance(data, list) and data and isinstance(data[0], dict):
            keys0 = {k.lower() for k in data[0].keys()}
            if keys0 & {"annotations", "label", "category", "mask", "polygon"}:
                return True
    except Exception:
        return False
    return False


def extract_modality(ds) -> str:
    """Возвращает значение Modality как верхний регистр либо 'UNKNOWN'."""
    elem = ds.get((0x0008, 0x0060))
    if elem is None:
        return "UNKNOWN"
    value = getattr(elem, "value", elem)
    if value is None:
        return "UNKNOWN"
    value_str = str(value).strip().upper()
    return value_str or "UNKNOWN"


@dataclass
class StudyResult:
    study_key: str                # путь директории или UID исследования
    has_label: bool
    non_anon_patients: List[str]
    modalities: List[str]
    errors: List[str] = field(default_factory=list)
    study_path_rep: Optional[str] = None  # человекочитаемое представление (путь к первой папке и т.п.)
    label_sources: Set[str] = field(default_factory=set)
    series: Dict[str, Dict[str, Union[str, int, List[str]]]] = field(default_factory=dict)
    file_count: int = 0
    patient_ids: List[str] = field(default_factory=list)


@dataclass
class WorkerConfig:
    modality_filter: Optional[Set[str]] = None
    strict: bool = False
    exclude_patterns: Tuple[str, ...] = ()
    detect_series_keywords: Set[str] = field(default_factory=lambda: set(LABEL_SERIES_KEYWORDS))
    detect_label_file_patterns: Tuple[str, ...] = LABEL_FILE_PATTERNS
    detect_label_json: bool = True


def detect_label_from_dataset(ds: pydicom.dataset.Dataset, sources: Set[str]) -> None:
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

    if str(ds.get("ContentLabel", "")).lower() in {"segmentation", "mask", "roi"}:
        sources.add("content_label")

# ---------------- Обход и группировка ----------------

def iter_all_files(root: Union[str, Path],
                   follow_symlinks: bool = False,
                   max_depth: Optional[int] = None,
                   exclude_patterns: Tuple[str, ...] = ()) -> Iterable[Path]:
    root = Path(root)
    root_parts = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        d = Path(dirpath)
        # Ограничение глубины
        if max_depth is not None:
            if len(Path(dirpath).parts) - root_parts >= max_depth:
                # Обрезаем обход глубже
                dirnames[:] = []
        # Фильтр симлинков на директории
        if not follow_symlinks:
            dirnames[:] = [dn for dn in dirnames if not Path(dirpath, dn).is_symlink()]
        if exclude_patterns:
            dirnames[:] = [dn for dn in dirnames if not should_exclude(Path(dirpath) / dn, exclude_patterns)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if not follow_symlinks and p.is_symlink():
                continue
            if should_exclude(p, exclude_patterns):
                continue
            yield p


def find_dicom_studies_by_dir(folder_path: Union[str, Path],
                              list_empty: bool = False,
                              follow_symlinks: bool = False,
                              max_depth: Optional[int] = None,
                              exclude_patterns: Tuple[str, ...] = ()) -> Tuple[List[str], List[str]]:
    """Старое правило: «директория с ≥1 DICOM — это исследование»."""
    dicom_studies: List[str] = []
    empty_folders: List[str] = []
    folder_path = Path(folder_path)

    for dirpath, dirnames, filenames in os.walk(folder_path, followlinks=follow_symlinks):
        d = Path(dirpath)
        # Глубина
        if max_depth is not None:
            if len(d.parts) - len(folder_path.parts) > max_depth:
                dirnames[:] = []
                continue
        # Симлинки
        if not follow_symlinks:
            dirnames[:] = [dn for dn in dirnames if not (d / dn).is_symlink()]
        if exclude_patterns:
            dirnames[:] = [dn for dn in dirnames if not should_exclude(d / dn, exclude_patterns)]

        if not dirnames and not filenames:
            if list_empty:
                empty_folders.append(str(d))
            continue

        has_dcm = False
        for fn in filenames:
            p = d / fn
            if not follow_symlinks and p.is_symlink():
                continue
            if should_exclude(p, exclude_patterns):
                continue
            try:
                if is_dicom_file(p):
                    has_dcm = True
                    break
            except Exception:
                continue
        if has_dcm:
            dicom_studies.append(str(d))

    return dicom_studies, empty_folders


def find_dicom_studies_by_uid(folder_path: Union[str, Path],
                              follow_symlinks: bool = False,
                              max_depth: Optional[int] = None,
                              exclude_patterns: Tuple[str, ...] = ()) -> Dict[str, List[str]]:
    """
    Новая опция: группировать файлы по StudyInstanceUID.
    Возвращает dict: UID -> список файлов DICOM.
    """
    studies: Dict[str, List[str]] = defaultdict(list)
    for p in iter_all_files(
        folder_path,
        follow_symlinks=follow_symlinks,
        max_depth=max_depth,
        exclude_patterns=exclude_patterns,
    ):
        if not p.is_file():
            continue
        try:
            if not is_dicom_file(p):
                continue
            ds = read_dicom_header(str(p))
            if ds is None:
                continue
            uid = None
            # Тег (0020,000D) — StudyInstanceUID
            if (0x0020, 0x000D) in ds:
                uid = str(ds[0x0020, 0x000D].value).strip()
            if uid:
                studies[uid].append(str(p))
        except Exception:
            continue
    return studies


# ---------------- Обработка одного исследования ----------------



def _process_dir_study(study_path: str, config: WorkerConfig, debug: bool = False) -> StudyResult:
    non_anon_patients: Set[str] = set()
    patient_ids: Set[str] = set()
    errors: List[str] = []
    modalities: List[str] = []
    label_sources: Set[str] = set()
    series_raw: Dict[str, Dict[str, Union[str, int, Set[str]]]] = {}
    file_count = 0

    try:
        for f in Path(study_path).rglob("*"):
            if not f.is_file():
                continue
            if should_exclude(f, config.exclude_patterns):
                continue

    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            modality = extract_modality(ds)
            if modality and modality != "UNKNOWN":
                modalities.append(modality)
                if modality in {"RTSTRUCT", "SEG", "RTSEGANN"}:
                    has_label = True

            # Проверка анонимности — строго через исходную функцию, "как есть"
            if True:
                is_anon, _, _ = check_dicom_anonymization(str(f))
                if not is_anon and (0x0010, 0x0010) in ds:
                    pn = str(ds[0x0010, 0x0010].value or "").strip()
                    if pn:
                        non_anon_patients.add(pn)
        except Exception as e:
            errors.append(f"{f}: {e!s}")

    # Дополнительные признаки разметки (NNRD/JSON)
    label_files = list(Path(study_path).rglob("*.seg.nrrd"))
    if label_files:
        has_label = True
    # Фильтр JSON по имени/структуре
    for jf in Path(study_path).rglob("*.json"):
        try:
            if is_label_json(jf):
                has_label = True
                break
        except Exception:
            # игнорируем ошибки парсинга
            pass

    if debug:
        print(
            f"[DIR] {study_path} -> has_label={has_label}, modalities={sorted(set(modalities))}",
            flush=True,
        )

    series_serializable: Dict[str, Dict[str, Union[str, int, List[str]]]] = {}
    for uid, info in series_raw.items():
        modalities_set = info.get("modalities", set())
        if isinstance(modalities_set, set):
            modalities_list = sorted(modalities_set)
        else:
            modalities_list = list(modalities_set)
        series_serializable[uid] = {
            "description": info.get("description", ""),
            "modalities": modalities_list,
            "files": int(info.get("files", 0)),
        }

    return StudyResult(
        study_key=study_path,
        has_label=has_label,
        non_anon_patients=sorted(non_anon_patients),
        modalities=modalities,
        errors=errors,
        study_path_rep=study_path,
        label_sources=label_sources,
        series=series_serializable,
        file_count=file_count,
        patient_ids=sorted(patient_ids),
    )



def _process_uid_study(uid: str, files: List[str], config: WorkerConfig, debug: bool = False) -> StudyResult:
    non_anon_patients: Set[str] = set()
    patient_ids: Set[str] = set()
    errors: List[str] = []
    modalities: List[str] = []
    label_sources: Set[str] = set()
    series_raw: Dict[str, Dict[str, Union[str, int, Set[str]]]] = {}
    file_count = 0
    rep_path = str(Path(files[0]).parent) if files else None

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            modality = extract_modality(ds)
            if modality and modality != "UNKNOWN":
                if config.modality_filter is None or modality in config.modality_filter:
                    modalities.append(modality)
                    detect_label_from_dataset(ds, label_sources)

            is_anon, _, _ = check_dicom_anonymization(str(f))
            if not is_anon and (0x0010, 0x0010) in ds:
                pn = str(ds[0x0010, 0x0010].value or "").strip()
                if pn:
                    non_anon_patients.add(pn)
            if (0x0010, 0x0020) in ds:
                pid = str(ds[0x0010, 0x0020].value or "").strip()
                if pid:
                    patient_ids.add(pid)

            series_uid = str(ds.get((0x0020, 0x000E), "")).strip()
            series_desc = str(ds.get((0x0008, 0x103E), "")).strip()
            if series_uid:
                info = series_raw.setdefault(
                    series_uid,
                    {"description": series_desc, "modalities": set(), "files": 0},
                )
                info["files"] = int(info.get("files", 0)) + 1
                if modality:
                    info.setdefault("modalities", set()).add(modality)
                if series_desc:
                    lowered = series_desc.lower()
                    for keyword in config.detect_series_keywords:
                        if keyword in lowered:
                            label_sources.add(f"series_description:{keyword}")
                            break

        except Exception as err:
            if config.strict:
                raise RuntimeError(f"Ошибка обработки {f}: {err}") from err
            errors.append(f"{f}: {err!s}")

    for base in {str(Path(f).parent) for f in files}:
        base_path = Path(base)
        if should_exclude(base_path, config.exclude_patterns):
            continue
        for pattern in config.detect_label_file_patterns:
            matches = list(base_path.rglob(pattern))
            if matches:
                label_sources.add(f"file_pattern:{pattern}")
                break
        if config.detect_label_json:
            for jf in base_path.rglob("*.json"):
                if should_exclude(jf, config.exclude_patterns):
                    continue
                try:
                    if is_label_json(jf):
                        label_sources.add("label_json")
                        break
                except Exception as err:
                    if config.strict:
                        raise RuntimeError(f"Ошибка анализа JSON {jf}: {err}") from err
                    errors.append(f"{jf}: {err!s}")
        if label_sources:
            break

    has_label = bool(label_sources)

    if debug:
        print(
            f"[UID] {uid} ({rep_path}) -> has_label={has_label}, modalities={sorted(set(modalities))}",
            flush=True,
        )

    series_serializable: Dict[str, Dict[str, Union[str, int, List[str]]]] = {}
    for series_uid, info in series_raw.items():
        modalities_set = info.get("modalities", set())
        if isinstance(modalities_set, set):
            modalities_list = sorted(modalities_set)
        else:
            modalities_list = list(modalities_set)
        series_serializable[series_uid] = {
            "description": info.get("description", ""),
            "modalities": modalities_list,
            "files": int(info.get("files", 0)),
        }

    return StudyResult(
        study_key=uid,
        has_label=has_label,
        non_anon_patients=sorted(non_anon_patients),
        modalities=modalities,
        errors=errors,
        study_path_rep=rep_path,
        label_sources=label_sources,
        series=series_serializable,
        file_count=file_count,
        patient_ids=sorted(patient_ids),
    )


def _process_dir_batch(study_paths: List[str], config: WorkerConfig, debug: bool = False) -> List[StudyResult]:
    batch_results: List[StudyResult] = []
    for study_path in study_paths:
        batch_results.append(_process_dir_study(study_path, config=config, debug=debug))
    return batch_results


def _process_uid_batch(items: List[Tuple[str, List[str]]], config: WorkerConfig, debug: bool = False) -> List[StudyResult]:
    batch_results: List[StudyResult] = []
    for uid, files in items:
        batch_results.append(_process_uid_study(uid, files, config=config, debug=debug))
    return batch_results

# ---------------- Основной конвейер ----------------



def analyze_dataset(folder_path: Union[str, Path],
                    debug: bool = False,
                    group_by: str = "dir",
                    executor_kind: str = "process",
                    workers: Optional[int] = None,
                    follow_symlinks: bool = False,
                    max_depth: Optional[int] = None,
                    list_empty: bool = False,
                    modality_filter: Optional[Iterable[str]] = None,
                    only_labeled: bool = False,
                    only_nonanon: bool = False,
                    exclude_patterns: Optional[Iterable[str]] = None,
                    strict: bool = False,
                    no_progress: bool = False,
                    batch_size: int = 1) -> Dict:
    """Возвращает словарь с агрегатами по результатам."""
    folder_path = Path(folder_path)

    if group_by not in {"dir", "study"}:
        raise ValueError("--group-by должен быть 'dir' или 'study'")

    if executor_kind not in {"process", "thread"}:
        raise ValueError("--executor должен быть 'process' или 'thread'")

    exclude_patterns = tuple(exclude_patterns or ())
    modality_filter_set: Optional[Set[str]] = None
    if modality_filter:
        modality_filter_set = {m.strip().upper() for m in modality_filter if m.strip()}
        if not modality_filter_set:
            modality_filter_set = None

    config = WorkerConfig(
        modality_filter=modality_filter_set,
        strict=strict,
        exclude_patterns=exclude_patterns,
    )

    results = {
        "total_studies": 0,
        "processed_studies": 0,
        "filtered_out_studies": 0,
        "labeled_studies": 0,
        "non_anonymous_studies": defaultdict(list),
        "modality_stats": Counter(),
        "empty_folders": [],
        "errors": [],
        "debug_info": [],
        "group_by": group_by,
        "study_stats": {},
        "patient_summary": defaultdict(list),
        "study_file_counts": {},
        "series_summary": {},
        "label_source_stats": Counter(),
    }

    uid_map: Dict[str, List[str]] = {}
    if group_by == "dir":
        studies, empty = find_dicom_studies_by_dir(
            folder_path,
            list_empty=list_empty,
            follow_symlinks=follow_symlinks,
            max_depth=max_depth,
            exclude_patterns=exclude_patterns,
        )
        study_items = studies
        results["empty_folders"] = empty
    else:
        uid_map = find_dicom_studies_by_uid(
            folder_path,
            follow_symlinks=follow_symlinks,
            max_depth=max_depth,
            exclude_patterns=exclude_patterns,
        )
        study_items = list(uid_map.keys())

    results["total_studies"] = len(study_items)

    if results["total_studies"] == 0:
        if debug:
            print("DICOM-исследований не найдено.")
        return results

    Exec = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
    max_workers = workers or os.cpu_count() or 1
    batch_size = max(1, int(batch_size or 1))

    futures = []
    with Exec(max_workers=max_workers) as pool:
        if group_by == "dir":
            if batch_size > 1:
                for chunk in batched(study_items, batch_size):
                    futures.append(pool.submit(_process_dir_batch, list(chunk), config, debug))
            else:
                for sp in study_items:
                    futures.append(pool.submit(_process_dir_study, sp, config, debug))
        else:
            if batch_size > 1:
                uid_items = [(uid, uid_map.get(uid, [])) for uid in study_items]
                for chunk in batched(uid_items, batch_size):
                    futures.append(pool.submit(_process_uid_batch, list(chunk), config, debug))
            else:
                for uid in study_items:
                    futures.append(pool.submit(_process_uid_study, uid, uid_map.get(uid, []), config, debug))

        iterator = as_completed(futures)
        if not no_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Анализ исследований")

        for fut in iterator:
            try:
                result = fut.result()
            except Exception as e:
                logger.error("Ошибка обработки исследования: %s", e)
                results["errors"].append(str(e))
                if strict:
                    raise
                continue

            batch_results = result if isinstance(result, list) else [result]
            for r in batch_results:
                include = True
                if modality_filter_set is not None and not r.modalities:
                    include = False
                if only_labeled and not r.has_label:
                    include = False
                if only_nonanon and not r.non_anon_patients:
                    include = False

                if not include:
                    results["filtered_out_studies"] += 1
                    continue

                results["processed_studies"] += 1
                if r.has_label:
                    results["labeled_studies"] += 1
                if r.non_anon_patients:
                    results["non_anonymous_studies"][r.study_key] = r.non_anon_patients
                results["modality_stats"].update(r.modalities)
                results["errors"].extend(r.errors)
                results["study_file_counts"][r.study_key] = r.file_count
                results["study_stats"][r.study_key] = {
                    "modalities": sorted(set(r.modalities)),
                    "has_label": r.has_label,
                    "label_sources": sorted(r.label_sources),
                    "non_anonymous_patients": r.non_anon_patients,
                    "patient_ids": r.patient_ids,
                    "file_count": r.file_count,
                    "series": r.series,
                    "errors": r.errors,
                }
                for pid in r.patient_ids:
                    results["patient_summary"][pid].append(r.study_key)
                for source in r.label_sources:
                    results["label_source_stats"][source] += 1
                for series_uid, info in r.series.items():
                    entry = results["series_summary"].setdefault(
                        series_uid,
                        {
                            "description": info.get("description", ""),
                            "modalities": Counter(),
                            "studies": set(),
                            "files": 0,
                        },
                    )
                    entry["files"] += int(info.get("files", 0))
                    entry["studies"].add(r.study_key)
                    for mod in info.get("modalities", []):
                        entry["modalities"][mod] += 1
                if debug:
                    results["debug_info"].append({
                        "study_key": r.study_key,
                        "has_label": r.has_label,
                        "modalities": r.modalities,
                        "errors": r.errors,
                        "study_path_rep": r.study_path_rep,
                        "label_sources": sorted(r.label_sources),
                    })

    if results["study_file_counts"]:
        total_files = sum(results["study_file_counts"].values())
        results["average_files_per_study"] = total_files / max(results["processed_studies"], 1)
    else:
        results["average_files_per_study"] = 0.0

    results["unique_patients"] = len(results["patient_summary"])

    return results

# ---------------- Отчёты ----------------

def print_report(results: Dict) -> None:
    print("\n=== Результаты анализа ===")
    print(f"Всего исследований (до фильтрации): {results['total_studies']}")
    print(f"Обработано после фильтров: {results['processed_studies']}")
    if results.get("filtered_out_studies"):
        print(f"Отфильтровано исследований: {results['filtered_out_studies']}")

    processed = results.get("processed_studies", 0)
    if processed > 0:
        percent = results['labeled_studies'] / processed
    else:
        percent = 0.0
    print(f"Размеченных исследований: {results['labeled_studies']} ({percent:.1%})")

    avg_files = results.get("average_files_per_study", 0.0)
    print(f"Среднее число DICOM-файлов на исследование: {avg_files:.1f}")
    print(f"Уникальных пациентов: {results.get('unique_patients', 0)}")

    if results.get("label_source_stats"):
        print("\nИсточники определения разметки:")
        for source, count in results["label_source_stats"].most_common():
            print(f"  {source}: {count}")

    print("\nСтатистика по Modality:")
    for modality, count in results["modality_stats"].most_common():
        print(f"  {modality}: {count} файлов")

    if results.get("series_summary"):
        print("\nСерии с максимальным количеством файлов:")
        series_items = sorted(
            results["series_summary"].items(),
            key=lambda item: item[1].get("files", 0),
            reverse=True,
        )[:10]
        for uid, info in series_items:
            studies = info.get("studies", set())
            if isinstance(studies, set):
                study_count = len(studies)
            else:
                study_count = len(list(studies))
            modality_counts = info.get("modalities", Counter())
            if isinstance(modality_counts, Counter):
                mod_str = ", ".join(f"{m}:{c}" for m, c in modality_counts.most_common())
            else:
                mod_str = ", ".join(f"{m}:{modality_counts[m]}" for m in modality_counts)
            print(
                f"  {uid} | файлов: {info.get('files', 0)}, исследований: {study_count}, "
                f"описание: {info.get('description', '')}, modality: {mod_str}"
            )

    if results.get("patient_summary"):
        print("\nПациенты с наибольшим числом исследований:")
        patient_items = sorted(
            results["patient_summary"].items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )[:10]
        for pid, studies in patient_items:
            print(f"  {pid or '[empty]'}: {len(studies)} исследований")

    print(f"\nПустых папок найдено: {len(results.get('empty_folders', []))}")
    if results.get("empty_folders"):
        print("\nПолный список пустых папок:")
        for folder in results["empty_folders"]:
            print(f"  - {folder}")

    non_anon = len(results["non_anonymous_studies"])
    print(f"\nНеанонимизированных исследований: {non_anon}")
    if non_anon:
        print("\nДетали по исследованиям:")
        for study_key, patients in results["non_anonymous_studies"].items():
            print(f"\nИсследование: {study_key}")
            print("Неанонимизированные PatientName:")
            for p in patients:
                print(f"  - {p}")


def save_report_to_txt(results: Dict, output_file: Union[str, Path]) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Результаты анализа ===\n")
        f.write(f"Всего исследований (до фильтрации): {results['total_studies']}\n")
        f.write(f"Обработано после фильтров: {results['processed_studies']}\n")
        f.write(f"Отфильтровано исследований: {results.get('filtered_out_studies', 0)}\n")

        processed = results.get("processed_studies", 0)
        if processed > 0:
            percent = results['labeled_studies'] / processed
        else:
            percent = 0.0
        f.write(f"Размеченных исследований: {results['labeled_studies']} ({percent:.1%})\n")
        f.write(f"Среднее число DICOM-файлов на исследование: {results.get('average_files_per_study', 0.0):.1f}\n")
        f.write(f"Уникальных пациентов: {results.get('unique_patients', 0)}\n")

        if results.get("label_source_stats"):
            f.write("\nИсточники определения разметки:\n")
            for source, count in results["label_source_stats"].most_common():
                f.write(f"  {source}: {count}\n")

        f.write("\nСтатистика по Modality:\n")
        for modality, count in results["modality_stats"].most_common():
            f.write(f"  {modality}: {count} файлов\n")

        if results.get("series_summary"):
            f.write("\nСерии с максимальным количеством файлов:\n")
            series_items = sorted(
                results["series_summary"].items(),
                key=lambda item: item[1].get("files", 0),
                reverse=True,
            )[:10]
            for uid, info in series_items:
                studies = info.get("studies", set())
                if isinstance(studies, set):
                    study_count = len(studies)
                else:
                    study_count = len(list(studies))
                modality_counts = info.get("modalities", Counter())
                if isinstance(modality_counts, Counter):
                    mod_str = ", ".join(f"{m}:{c}" for m, c in modality_counts.most_common())
                else:
                    mod_str = ", ".join(f"{m}:{modality_counts[m]}" for m in modality_counts)
                f.write(
                    f"  {uid} | файлов: {info.get('files', 0)}, исследований: {study_count}, "
                    f"описание: {info.get('description', '')}, modality: {mod_str}\n"
                )

        if results.get("patient_summary"):
            f.write("\nПациенты с наибольшим числом исследований:\n")
            patient_items = sorted(
                results["patient_summary"].items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )[:10]
            for pid, studies in patient_items:
                f.write(f"  {pid or '[empty]'}: {len(studies)} исследований\n")

        f.write(f"\nПустых папок найдено: {len(results.get('empty_folders', []))}\n")
        if results.get("empty_folders"):
            f.write("\nПолный список пустых папок:\n")
            for folder in results["empty_folders"]:
                f.write(f"  - {folder}\n")

        non_anon = len(results["non_anonymous_studies"])
        f.write(f"\nНеанонимизированных исследований: {non_anon}\n")
        if non_anon:
            f.write("\nДетали по исследованиям:\n")
            for study_key, patients in results["non_anonymous_studies"].items():
                f.write(f"\nИсследование: {study_key}\n")
                f.write("Неанонимизированные PatientName:\n")
                for p in patients:
                    f.write(f"  - {p}\n")


def save_modality_to_csv(results: Dict, csv_file: Union[str, Path]) -> None:
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Modality", "Count"])
        for modality, count in results["modality_stats"].most_common():
            w.writerow([modality, count])


def save_nonanon_to_csv(results: Dict, csv_file: Union[str, Path]) -> None:
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Study Key", "Patient Name"])
        for study_key, patients in results["non_anonymous_studies"].items():
            for p in patients:
                w.writerow([study_key, p])


def save_report_to_json(results: Dict, json_file: Union[str, Path]) -> None:
    results_copy = dict(results)
    results_copy["non_anonymous_studies"] = {
        key: list(values)
        for key, values in dict(results_copy["non_anonymous_studies"]).items()
    }
    results_copy["modality_stats"] = dict(results_copy["modality_stats"])
    results_copy["patient_summary"] = {
        patient: list(studies)
        for patient, studies in results_copy.get("patient_summary", {}).items()
    }
    results_copy["label_source_stats"] = dict(results_copy.get("label_source_stats", {}))
    results_copy["study_stats"] = {
        key: value
        for key, value in results_copy.get("study_stats", {}).items()
    }

    series_serialized: Dict[str, Dict[str, Union[str, int, Dict[str, int], List[str]]]] = {}
    for uid, info in results_copy.get("series_summary", {}).items():
        studies = info.get("studies", set())
        if isinstance(studies, set):
            studies_list = sorted(studies)
        else:
            studies_list = list(studies)
        modality_counts = info.get("modalities", Counter())
        if isinstance(modality_counts, Counter):
            modality_counts = dict(modality_counts)
        else:
            modality_counts = dict(modality_counts)
        series_serialized[uid] = {
            "description": info.get("description", ""),
            "modalities": modality_counts,
            "studies": studies_list,
            "files": int(info.get("files", 0)),
        }
    results_copy["series_summary"] = series_serialized

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_copy, f, ensure_ascii=False, indent=2)




def main(argv: Optional[List[str]] = None) -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="Путь к файлу конфигурации (YAML или JSON)")
    config_args, remaining_argv = config_parser.parse_known_args(argv)

    config_defaults: Dict[str, Union[str, int, bool, List[str], None]] = {}
    config_dataset_path: Optional[str] = None
    if config_args.config:
        config_path = Path(config_args.config)
        if not config_path.is_file():
            print(f"Ошибка: файл конфигурации не найден: {config_path}", file=sys.stderr)
            return 2
        try:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                if yaml is None:
                    print("Ошибка: для YAML-конфигурации требуется пакет PyYAML", file=sys.stderr)
                    return 2
                with open(config_path, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh)
            else:
                with open(config_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
        except Exception as exc:
            print(f"Ошибка чтения конфигурации: {exc}", file=sys.stderr)
            return 2
        if not isinstance(loaded, dict):
            print("Ошибка: конфигурационный файл должен содержать объект верхнего уровня", file=sys.stderr)
            return 2
        config_defaults = dict(loaded)
        config_dataset_path = config_defaults.pop("dataset_path", None)

    parser = argparse.ArgumentParser(
        description="Анализатор DICOM (расширенный): поддержка SEG, группировка по StudyInstanceUID и пр.",
        parents=[config_parser],
    )
    parser.add_argument("dataset_path", nargs="?", help="Путь к датасету")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    parser.add_argument("--output", default="report.txt", help="Файл для текстового отчёта")
    parser.add_argument("--csv", action="store_true", help="Сохранять CSV-отчёты (modality и non-anonymous)")
    parser.add_argument("--json", action="store_true", help="Сохранять JSON-отчёт")
    parser.add_argument("--csv_modality", default="report_modality.csv", help="CSV-файл cо статистикой по modality")
    parser.add_argument("--csv_nonanon", default="report_nonanon.csv", help="CSV-файл с неанонимными PatientName")
    parser.add_argument("--json_file", default="report.json", help="JSON-файл для сохранения полного отчёта")

    parser.add_argument("--group-by", choices=["dir", "study"], default="dir",
                        help="Группировка исследований: dir (по папкам) или study (по StudyInstanceUID)")
    parser.add_argument("--executor", choices=["process", "thread"], default="process",
                        help="Тип пула для параллельной обработки")
    parser.add_argument("--workers", type=int, default=None, help="Число воркеров (по умолчанию = CPU count)")

    parser.add_argument("--follow-symlinks", action="store_true", help="Следовать симлинкам при обходе")
    parser.add_argument("--max-depth", type=int, default=None, help="Максимальная глубина обхода (уровней)")
    parser.add_argument("--list-empty", action="store_true", help="Собирать и выводить пустые папки")

    parser.add_argument("--modality-filter", nargs='+', help="Ограничить анализ указанными Modality (например, CT MR)")
    parser.add_argument("--only-labeled", action="store_true", help="Учитывать только исследования с разметкой")
    parser.add_argument("--only-nonanon", action="store_true", help="Учитывать только неанонимные исследования")
    parser.add_argument("--exclude-pattern", action="append", default=None,
                        help="Шаблон (fnmatch) для исключения файлов и директорий; можно указывать несколько раз")
    parser.add_argument("--strict", action="store_true", help="Строгий режим: останавливать анализ при первой ошибке")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Размер пакета исследований для одного задания пула")
    parser.add_argument("--no-progress", action="store_true", help="Отключить прогресс-бар tqdm")
    parser.add_argument("--log-level", default="INFO", help="Уровень логирования (по умолчанию INFO)")
    parser.add_argument("--log-file", help="Файл для записи логов")

    parser.set_defaults(**{k: v for k, v in config_defaults.items()
                           if k in {action.dest for action in parser._actions}})

    args = parser.parse_args(remaining_argv)

    if args.dataset_path is None:
        if config_dataset_path:
            args.dataset_path = str(config_dataset_path)
        else:
            print("Ошибка: не указан путь к датасету", file=sys.stderr)
            return 2

    configure_logging(args.log_level, args.log_file)

    if not os.path.isdir(args.dataset_path):
        print(f"Ошибка: путь не существует или не директория: {args.dataset_path}", file=sys.stderr)
        return 2

    exclude_patterns: List[str] = []
    if args.exclude_pattern:
        for item in args.exclude_pattern:
            if isinstance(item, (list, tuple, set)):
                exclude_patterns.extend(str(x) for x in item)
            else:
                exclude_patterns.append(str(item))

    modality_filter = args.modality_filter
    if isinstance(modality_filter, str):
        modality_filter = [modality_filter]

    print(f"Анализ датасета: {args.dataset_path}")
    results = analyze_dataset(
        args.dataset_path,
        debug=args.debug,
        group_by=args.group_by,
        executor_kind=args.executor,
        workers=args.workers,
        follow_symlinks=args.follow_symlinks,
        max_depth=args.max_depth,
        list_empty=args.list_empty,
        modality_filter=modality_filter,
        only_labeled=args.only_labeled,
        only_nonanon=args.only_nonanon,
        exclude_patterns=exclude_patterns,
        strict=args.strict,
        no_progress=args.no_progress,
        batch_size=args.batch_size,
    )
    print_report(results)
    save_report_to_txt(results, args.output)



    if args.csv:
        save_modality_to_csv(results, args.csv_modality)
        save_nonanon_to_csv(results, args.csv_nonanon)
        print(f"\nCSV-отчёты сохранены: {args.csv_modality}, {args.csv_nonanon}")
    if args.json:
        save_report_to_json(results, args.json_file)
        print(f"\nJSON-отчёт сохранён: {args.json_file}")
    print(f"\nОтчёт сохранён в файл: {args.output}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
