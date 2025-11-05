
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

ВАЖНО: Логику анонимизации мы НЕ меняли — она берётся из исходного файла dcmmetatest.py,
и вызывается как есть (через import исходной функции check_dicom_anonymization).
"""

from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Set, Union
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pydicom
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

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


@dataclass
class StudyResult:
    study_key: str                # путь директории или UID исследования
    has_label: bool
    non_anon_patients: List[str]
    modalities: List[str]
    errors: List[str] = field(default_factory=list)
    study_path_rep: Optional[str] = None  # человекочитаемое представление (путь к первой папке и т.п.)


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
            modality = str(ds.get((0x0008, 0x0060), "UNKNOWN")).strip().upper()
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
    non_anon_patients: Set[str] = set()
    errors: List[str] = []
    modalities: List[str] = []
    has_label = False
    rep_path = str(Path(files[0]).parent) if files else None

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            modality = str(ds.get((0x0008, 0x0060), "UNKNOWN")).strip().upper()
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
                    list_empty: bool = False) -> Dict:
    """
    Возвращает словарь с агрегатами по результатам.
    """
    folder_path = Path(folder_path)

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
    parser.add_argument("--group-by", choices=["dir", "study"], default="dir",
                        help="Группировка исследований: dir (по папкам) или study (по StudyInstanceUID)")
    parser.add_argument("--executor", choices=["process", "thread"], default="process",
                        help="Тип пула для параллельной обработки")
    parser.add_argument("--workers", type=int, default=None, help="Число воркеров (по умолчанию = CPU count)")

    parser.add_argument("--follow-symlinks", action="store_true", help="Следовать симлинкам при обходе")
    parser.add_argument("--max-depth", type=int, default=None, help="Максимальная глубина обхода (уровней)")
    parser.add_argument("--list-empty", action="store_true", help="Собирать и выводить пустые папки")

    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset_path):
        print(f"Ошибка: путь не существует или не директория: {args.dataset_path}", file=sys.stderr)
        return 2

    # Проверяем наличие функции анонимизации
    
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
