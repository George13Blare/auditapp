
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
- Авто-детект структуры датасета и поддержка внешних схем (JSON/YAML).
- Улучшенные сообщения об ошибках и диагностика.
- Форматы отчёта сохранены (TXT/CSV/JSON), имена флагов остаются совместимыми.

ВАЖНО: Логику анонимизации мы НЕ меняли — она берётся из исходного файла dcmmetatest.py,
и вызывается как есть (через import исходной функции check_dicom_anonymization).
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Iterable, Optional, Set, Union, Any
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    import pydicom  # type: ignore
    from pydicom.errors import InvalidDicomError
    HAS_PYDICOM = True
except ImportError:  # pragma: no cover - зависит от окружения
    pydicom = None  # type: ignore
    HAS_PYDICOM = False

    class InvalidDicomError(Exception):
        """Заглушка, когда pydicom недоступен."""

        pass

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm необязателен для CLI-помощи

    def tqdm(iterable=None, *_, **__):
        return iterable

# --- ВСТАВЛЕНА ИСХОДНАЯ ФУНКЦИЯ АНОНИМИЗАЦИИ (без изменений) ---
def check_dicom_anonymization(dicom_file):
    """Проверяет только PatientName на анонимность и возвращает Modality."""
    if not HAS_PYDICOM:
        raise RuntimeError(
            "Для проверки анонимизации требуется установленный пакет pydicom"
        )
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

def _has_dicom_preamble(path: str) -> bool:
    """Быстрая проверка преамбулы "DICM" без использования pydicom."""

    try:
        with open(path, "rb") as f:
            head = f.read(132)
            return len(head) >= 132 and head[128:132] == b"DICM"
    except Exception:
        return False


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
    if not HAS_PYDICOM:
        return _has_dicom_preamble(path)

    try:
        # Вариант 1: официальная функция (есть не во всех версиях pydicom)
        try:
            from pydicom.misc import is_dicom as _is_dicom
            return bool(_is_dicom(path))
        except Exception:
            pass

        # Вариант 2: пролог
        if _has_dicom_preamble(path):
            return True

        # Вариант 3: грубая попытка распарсить без пикселей
        try:
            pydicom.dcmread(path, stop_before_pixels=True, force=True)
            return True
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


# ---------------- Конфигурации и авто-детект ----------------


@dataclass
class SchemaConfig:
    """Структурированное представление пользовательской схемы датасета."""

    name: Optional[str] = None
    group_by: Optional[str] = None
    max_depth: Optional[int] = None
    follow_symlinks: Optional[bool] = None
    require_labels: Optional[bool] = None
    expected_modalities: Optional[List[str]] = None
    artifact_hints: Optional[Dict[str, Any]] = None


@dataclass
class SchemaSuggestion:
    """Результат авто-детекта структуры датасета."""

    dataset_path: str
    total_files_scanned: int
    dicom_candidates: int
    label_candidates: int
    other_files: int
    top_level_dirs: List[str]
    probable_grouping: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["schema_stub"] = {
            "name": f"Auto-detected schema for {self.dataset_path}",
            "group_by": self.probable_grouping,
            "max_depth": None,
            "follow_symlinks": False,
            "require_labels": self.label_candidates > 0,
        }
        return data


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_schema_config(config_path: Optional[Union[str, Path]]) -> Optional[SchemaConfig]:
    """Загружает файл конфигурации (JSON или YAML) и возвращает SchemaConfig."""

    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema config not found: {config_path}")

    data: Dict[str, Any]
    if path.suffix.lower() in {".json"}:
        data = _read_json(path)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Для чтения YAML-конфигурации требуется пакет PyYAML"
            ) from exc
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        raise ValueError("Поддерживаются только JSON/YAML схемы")

    if not isinstance(data, dict):
        raise ValueError("Некорректный формат файла схемы: требуется JSON/YAML-объект")

    return SchemaConfig(
        name=data.get("name"),
        group_by=data.get("group_by"),
        max_depth=data.get("max_depth"),
        follow_symlinks=data.get("follow_symlinks"),
        require_labels=data.get("require_labels"),
        expected_modalities=data.get("expected_modalities"),
        artifact_hints=data.get("artifact_hints"),
    )


def detect_dataset_structure(
    folder_path: Union[str, Path],
    sample_limit: int = 200,
    max_depth: int = 3,
    follow_symlinks: bool = False,
) -> SchemaSuggestion:
    """Быстрая эвристическая оценка структуры датасета перед основным анализом."""

    folder = Path(folder_path)
    total_files = 0
    dicom_hits = 0
    label_hits = 0
    other = 0
    per_dir_hits: Counter = Counter()
    sampled_paths: List[Path] = []

    for idx, file_path in enumerate(
        iter_all_files(folder, follow_symlinks=follow_symlinks, max_depth=max_depth)
    ):
        if idx >= sample_limit:
            break
        if not file_path.is_file():
            continue
        sampled_paths.append(file_path)
        parent = str(file_path.parent.relative_to(folder)) or "."
        total_files += 1
        try:
            if is_dicom_file(file_path):
                dicom_hits += 1
                per_dir_hits[parent] += 1
                continue
        except Exception:
            pass

        try:
            if file_path.suffix.lower() == ".json" and is_label_json(file_path):
                label_hits += 1
                per_dir_hits[parent] += 1
                continue
        except Exception:
            pass

        other += 1

    probable_group = "dir"
    notes: List[str] = []
    if dicom_hits > 0:
        # Считаем, сколько разных папок содержат DICOM/разметку
        unique_dirs = len(per_dir_hits)
        if unique_dirs > 0:
            avg_per_dir = dicom_hits / unique_dirs
            if avg_per_dir < 2:
                probable_group = "study"
                notes.append(
                    "Файлы DICOM распределены по многим подпапкам — рекомендуем группировку по StudyInstanceUID"
                )
            else:
                notes.append(
                    "Файлы DICOM сосредоточены в отдельных директориях — подойдёт группировка по папкам"
                )
    else:
        notes.append("DICOM-файлы не обнаружены в выборке — проверьте путь или увеличьте глубину")

    top_level_dirs = sorted({p.parts[0] if p.parts else "." for p in sampled_paths})

    if label_hits == 0:
        notes.append(
            "Файлы разметки не обнаружены в выборке — при необходимости добавьте правила в схему"
        )

    if total_files >= sample_limit:
        notes.append("Достигнут лимит выборки — структура может быть сложнее")

    return SchemaSuggestion(
        dataset_path=str(folder),
        total_files_scanned=total_files,
        dicom_candidates=dicom_hits,
        label_candidates=label_hits,
        other_files=other,
        top_level_dirs=top_level_dirs,
        probable_grouping=probable_group,
        notes=notes,
    )


# ---------------- Обход и группировка ----------------

def iter_all_files(root: Union[str, Path],
                   follow_symlinks: bool = False,
                   max_depth: Optional[int] = None) -> Iterable[Path]:
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
        for fn in filenames:
            p = Path(dirpath) / fn
            if not follow_symlinks and p.is_symlink():
                continue
            yield p


def find_dicom_studies_by_dir(folder_path: Union[str, Path],
                              list_empty: bool = False,
                              follow_symlinks: bool = False,
                              max_depth: Optional[int] = None) -> Tuple[List[str], List[str]]:
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

        if not dirnames and not filenames:
            if list_empty:
                empty_folders.append(str(d))
            continue

        has_dcm = False
        for fn in filenames:
            p = d / fn
            if not follow_symlinks and p.is_symlink():
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
                              max_depth: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Новая опция: группировать файлы по StudyInstanceUID.
    Возвращает dict: UID -> список файлов DICOM.
    """
    if not HAS_PYDICOM:
        raise RuntimeError("Группировка по UID требует установленный пакет pydicom")

    studies: Dict[str, List[str]] = defaultdict(list)
    for p in iter_all_files(folder_path, follow_symlinks=follow_symlinks, max_depth=max_depth):
        if not p.is_file():
            continue
        try:
            if not is_dicom_file(p):
                continue
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            except Exception:
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

def _process_dir_study(study_path: str, debug: bool = False) -> StudyResult:
    if not HAS_PYDICOM:
        raise RuntimeError("Обработка исследований по директории требует pydicom")

    non_anon_patients: Set[str] = set()
    errors: List[str] = []
    modalities: List[str] = []
    has_label = False

    all_files = list(Path(study_path).rglob("*"))
    dicom_files = [f for f in all_files if f.is_file() and is_dicom_file(f)]

    if debug:
        print(f"[DIR] Обработка: {study_path}, DICOM файлов: {len(dicom_files)}", flush=True)

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
        print(f"[DIR] {study_path} -> has_label={has_label}, modalities={sorted(set(modalities))}", flush=True)

    return StudyResult(
        study_key=study_path,
        has_label=has_label,
        non_anon_patients=sorted(non_anon_patients),
        modalities=modalities,
        errors=errors,
        study_path_rep=study_path,
    )


def _process_uid_study(uid: str, files: List[str], debug: bool = False) -> StudyResult:
    if not HAS_PYDICOM:
        raise RuntimeError("Обработка исследований по UID требует pydicom")

    non_anon_patients: Set[str] = set()
    errors: List[str] = []
    modalities: List[str] = []
    has_label = False
    rep_path = str(Path(files[0]).parent) if files else None

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            modality = extract_modality(ds)
            if modality and modality != "UNKNOWN":
                modalities.append(modality)
                if modality in {"RTSTRUCT", "SEG", "RTSEGANN"}:
                    has_label = True

            if True:
                is_anon, _, _ = check_dicom_anonymization(str(f))
                if not is_anon and (0x0010, 0x0010) in ds:
                    pn = str(ds[0x0010, 0x0010].value or "").strip()
                    if pn:
                        non_anon_patients.add(pn)

        except Exception as e:
            errors.append(f"{f}: {e!s}")

    # Дополнительные признаки разметки в директориях, где лежат файлы исследования
    for base in {str(Path(f).parent) for f in files}:
        # NRRD
        if list(Path(base).rglob("*.seg.nrrd")):
            has_label = True
            break
        # JSON
        for jf in Path(base).rglob("*.json"):
            try:
                if is_label_json(jf):
                    has_label = True
                    break
            except Exception:
                pass
        if has_label:
            break

    if debug:
        print(f"[UID] {uid} ({rep_path}) -> has_label={has_label}, modalities={sorted(set(modalities))}", flush=True)

    return StudyResult(
        study_key=uid,
        has_label=has_label,
        non_anon_patients=sorted(non_anon_patients),
        modalities=modalities,
        errors=errors,
        study_path_rep=rep_path,
    )


# ---------------- Основной конвейер ----------------

def analyze_dataset(folder_path: Union[str, Path],
                    debug: bool = False,
                    group_by: str = "dir",
                    executor_kind: str = "process",
                    workers: Optional[int] = None,
                    follow_symlinks: bool = False,
                    max_depth: Optional[int] = None,
                    list_empty: bool = False,
                    schema_config: Optional[SchemaConfig] = None) -> Dict:
    """
    Возвращает словарь с агрегатами по результатам.
    """
    folder_path = Path(folder_path)

    if not HAS_PYDICOM:
        raise RuntimeError("Для анализа датасета необходим установленный пакет pydicom")

    if schema_config:
        if schema_config.group_by and not group_by:
            group_by = schema_config.group_by
        if schema_config.max_depth is not None and max_depth is None:
            max_depth = schema_config.max_depth
        if schema_config.follow_symlinks is not None:
            follow_symlinks = schema_config.follow_symlinks

    if group_by not in {"dir", "study"}:
        raise ValueError("--group-by должен быть 'dir' или 'study'")

    if executor_kind not in {"process", "thread"}:
        raise ValueError("--executor должен быть 'process' или 'thread'")

    results = {
        "total_studies": 0,
        "labeled_studies": 0,
        "non_anonymous_studies": defaultdict(list),  # study_key -> [names]
        "modality_stats": Counter(),
        "empty_folders": [],
        "errors": [],
        "debug_info": [],
        "group_by": group_by,
        "schema_warnings": [],
    }

    # 1) Получаем список «исследований»
    uid_map: Dict[str, List[str]] = {}
    if group_by == "dir":
        studies, empty = find_dicom_studies_by_dir(
            folder_path, list_empty=list_empty, follow_symlinks=follow_symlinks, max_depth=max_depth
        )
        study_items = studies
        results["empty_folders"] = empty
    else:
        uid_map = find_dicom_studies_by_uid(
            folder_path, follow_symlinks=follow_symlinks, max_depth=max_depth
        )
        study_items = list(uid_map.keys())

    results["total_studies"] = len(study_items)

    if results["total_studies"] == 0:
        if debug:
            print("DICOM-исследований не найдено.")
        return results

    # 2) Параллельная обработка
    Exec = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
    max_workers = workers or os.cpu_count() or 1

    futures = []
    with Exec(max_workers=max_workers) as pool:
        if group_by == "dir":
            for sp in study_items:
                futures.append(pool.submit(_process_dir_study, sp, debug))
        else:
            for uid in study_items:
                futures.append(pool.submit(_process_uid_study, uid, uid_map.get(uid, []), debug))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Анализ исследований"):
            try:
                r: StudyResult = fut.result()
            except Exception as e:
                results["errors"].append(str(e))
                continue

            if r.has_label:
                results["labeled_studies"] += 1
            if r.non_anon_patients:
                results["non_anonymous_studies"][r.study_key] = r.non_anon_patients
            results["modality_stats"].update(r.modalities)
            results["errors"].extend(r.errors)
            if debug:
                results["debug_info"].append({
                    "study_key": r.study_key,
                    "has_label": r.has_label,
                    "modalities": r.modalities,
                    "errors": r.errors,
                    "study_path_rep": r.study_path_rep,
                })

    if schema_config:
        if schema_config.require_labels and results["total_studies"]:
            if results["labeled_studies"] < results["total_studies"]:
                results["schema_warnings"].append(
                    "Схема требует разметку для каждого исследования, но найдены не все"
                )
        if schema_config.expected_modalities:
            missing = set(schema_config.expected_modalities) - set(results["modality_stats"].keys())
            if missing:
                results["schema_warnings"].append(
                    f"Не найдены ожидаемые Modality: {', '.join(sorted(missing))}"
                )

    return results


# ---------------- Отчёты ----------------

def print_report(results: Dict) -> None:
    print("\n=== Результаты анализа ===")
    print(f"Всего исследований: {results['total_studies']}")
    if results["total_studies"] > 0:
        percent = results['labeled_studies'] / results['total_studies']
        print(f"Размеченных исследований: {results['labeled_studies']} ({percent:.1%})")
    else:
        print(f"Размеченных исследований: {results['labeled_studies']} (0.0%)")

    print("\nСтатистика по Modality:")
    for modality, count in results["modality_stats"].most_common():
        print(f"  {modality}: {count} файлов")

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

    if results.get("schema_warnings"):
        print("\nПредупреждения схемы:")
        for warn in results["schema_warnings"]:
            print(f"  - {warn}")


def save_report_to_txt(results: Dict, output_file: Union[str, Path]) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Результаты анализа ===\n")
        f.write(f"Всего исследований: {results['total_studies']}\n")
        if results["total_studies"] > 0:
            percent = results['labeled_studies'] / results['total_studies']
            f.write(f"Размеченных исследований: {results['labeled_studies']} ({percent:.1%})\n")
        else:
            f.write(f"Размеченных исследований: {results['labeled_studies']} (0.0%)\n")

        f.write("\nСтатистика по Modality:\n")
        for modality, count in results["modality_stats"].most_common():
            f.write(f"  {modality}: {count} файлов\n")

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

        if results.get("schema_warnings"):
            f.write("\nПредупреждения схемы:\n")
            for warn in results["schema_warnings"]:
                f.write(f"  - {warn}\n")


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
    results_copy["non_anonymous_studies"] = dict(results_copy["non_anonymous_studies"])
    results_copy["modality_stats"] = dict(results_copy["modality_stats"])
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_copy, f, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Анализатор DICOM (расширенный): поддержка SEG, группировка по StudyInstanceUID и пр."
    )
    parser.add_argument("dataset_path", help="Путь к датасету")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    parser.add_argument("--output", default="report.txt", help="Файл для текстового отчёта")
    parser.add_argument("--csv", action="store_true", help="Сохранять CSV-отчёты (modality и non-anonymous)")
    parser.add_argument("--json", action="store_true", help="Сохранять JSON-отчёт")
    parser.add_argument("--csv_modality", default="report_modality.csv", help="CSV-файл cо статистикой по modality")
    parser.add_argument("--csv_nonanon", default="report_nonanon.csv", help="CSV-файл с неанонимными PatientName")
    parser.add_argument("--json_file", default="report.json", help="JSON-файл для сохранения полного отчёта")

    # Новые опции
    parser.add_argument("--group-by", choices=["dir", "study"], default=None,
                        help="Группировка исследований: dir (по папкам) или study (по StudyInstanceUID)")
    parser.add_argument("--executor", choices=["process", "thread"], default="process",
                        help="Тип пула для параллельной обработки")
    parser.add_argument("--workers", type=int, default=None, help="Число воркеров (по умолчанию = CPU count)")

    parser.add_argument("--follow-symlinks", action="store_true", help="Следовать симлинкам при обходе")
    parser.add_argument("--max-depth", type=int, default=None, help="Максимальная глубина обхода (уровней)")
    parser.add_argument("--list-empty", action="store_true", help="Собирать и выводить пустые папки")
    parser.add_argument("--schema-config", help="Путь к файлу схемы датасета (JSON/YAML)")
    parser.add_argument("--schema-output", help="Путь для сохранения результатов авто-детекта схемы")
    parser.add_argument("--auto-detect-schema", action="store_true",
                        help="Выполнить быстрый анализ структуры перед основным запуском")
    parser.add_argument("--detect-only", action="store_true",
                        help="Только авто-детект структуры (без запуска основного анализа)")
    parser.add_argument("--schema-sample-limit", type=int, default=200,
                        help="Количество файлов для выборки при авто-детекте структуры")
    parser.add_argument("--schema-depth", type=int, default=3,
                        help="Глубина обхода при авто-детекте структуры")

    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset_path):
        print(f"Ошибка: путь не существует или не директория: {args.dataset_path}", file=sys.stderr)
        return 2

    # Проверяем наличие функции анонимизации
    
    schema_cfg = load_schema_config(args.schema_config)

    if args.auto_detect_schema or args.detect_only:
        suggestion = detect_dataset_structure(
            args.dataset_path,
            sample_limit=args.schema_sample_limit,
            max_depth=args.schema_depth,
            follow_symlinks=(schema_cfg.follow_symlinks if schema_cfg and schema_cfg.follow_symlinks is not None else args.follow_symlinks),
        )
        suggestion_dict = suggestion.to_dict()
        print("\n=== Авто-детект структуры датасета ===")
        print(f"Проанализировано файлов: {suggestion.total_files_scanned}")
        print(f"Найдено DICOM-кандидатов: {suggestion.dicom_candidates}")
        print(f"Найдено файлов разметки: {suggestion.label_candidates}")
        print(f"Прочие файлы: {suggestion.other_files}")
        print(f"Предполагаемая группировка: {suggestion.probable_grouping}")
        if suggestion.top_level_dirs:
            print("Верхние директории выборки:")
            for item in suggestion.top_level_dirs:
                print(f"  - {item}")
        if suggestion.notes:
            print("Комментарии:")
            for note in suggestion.notes:
                print(f"  - {note}")

        if args.schema_output:
            with open(args.schema_output, "w", encoding="utf-8") as f:
                json.dump(suggestion_dict, f, ensure_ascii=False, indent=2)
            print(f"\nРезультаты авто-детекта сохранены в {args.schema_output}")

        if args.detect_only:
            return 0

        # Если схема не указана явно, подставляем предположенное значение group_by
        if schema_cfg is None:
            schema_cfg = SchemaConfig(group_by=suggestion.probable_grouping)
        elif schema_cfg.group_by is None:
            schema_cfg.group_by = suggestion.probable_grouping

    if not HAS_PYDICOM:
        print(
            "Ошибка: для полного анализа требуется установленный пакет pydicom. "
            "Установите его командой 'pip install pydicom'.",
            file=sys.stderr,
        )
        return 2

    group_by_value = args.group_by or (schema_cfg.group_by if schema_cfg and schema_cfg.group_by else "dir")
    max_depth_value = args.max_depth if args.max_depth is not None else (
        schema_cfg.max_depth if schema_cfg and schema_cfg.max_depth is not None else None
    )
    follow_symlinks_value = args.follow_symlinks
    if schema_cfg and schema_cfg.follow_symlinks is not None:
        follow_symlinks_value = schema_cfg.follow_symlinks

    print(f"Анализ датасета: {args.dataset_path}")
    results = analyze_dataset(
        args.dataset_path,
        debug=args.debug,
        group_by=group_by_value,
        executor_kind=args.executor,
        workers=args.workers,
        follow_symlinks=follow_symlinks_value,
        max_depth=max_depth_value,
        list_empty=args.list_empty,
        schema_config=schema_cfg,
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
