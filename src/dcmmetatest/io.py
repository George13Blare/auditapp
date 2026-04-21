"""Модуль ввода-вывода для анализатора DICOM."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]

    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    HAS_YAML = False

try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    pydicom = None  # type: ignore
    HAS_PYDICOM = False

import logging

from .detectors import (
    LABEL_FILE_PATTERNS,
    detect_label_from_dataset,
    extract_modality,
    is_dicom_file,
    is_label_json,
)
from .models import AnalysisReport, StudyResult, WorkerConfig
from .utils import (
    check_dicom_anonymization,
    read_dicom_header,
    should_exclude,
)

logger = logging.getLogger(__name__)


def iter_all_files(
    root: str | Path,
    follow_symlinks: bool = False,
    max_depth: int | None = None,
    exclude_patterns: tuple[str, ...] = (),
) -> list[Path]:
    """
    Итерация по всем файлам в директории.

    Args:
        root: Корневая директория
        follow_symlinks: Следовать за симлинками
        max_depth: Максимальная глубина обхода
        exclude_patterns: Шаблоны для исключения

    Yields:
        Пути к файлам
    """
    root = Path(root)
    root_parts = len(root.parts)
    result: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        Path(dirpath)

        # Ограничение глубины
        if max_depth is not None:
            if len(Path(dirpath).parts) - root_parts >= max_depth:
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
            result.append(p)

    return result


def find_dicom_studies_by_dir(
    folder_path: str | Path,
    list_empty: bool = False,
    follow_symlinks: bool = False,
    max_depth: int | None = None,
    exclude_patterns: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """
    Находит исследования по директориям (старое правило).

    Args:
        folder_path: Корневая директория
        list_empty: Включать пустые директории
        follow_symlinks: Следовать за симлинками
        max_depth: Максимальная глубина
        exclude_patterns: Шаблоны исключения

    Returns:
        Кортеж (список исследований, список пустых папок)
    """
    dicom_studies: list[str] = []
    empty_folders: list[str] = []
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
            except Exception as e:
                logger.debug("Ошибка проверки DICOM файла %s: %s", p, e)
                continue

        if has_dcm:
            dicom_studies.append(str(d))

    return dicom_studies, empty_folders


def find_dicom_studies_by_uid(
    folder_path: str | Path,
    follow_symlinks: bool = False,
    max_depth: int | None = None,
    exclude_patterns: tuple[str, ...] = (),
) -> dict[str, list[str]]:
    """
    Группирует файлы по StudyInstanceUID.

    Args:
        folder_path: Корневая директория
        follow_symlinks: Следовать за симлинками
        max_depth: Максимальная глубина
        exclude_patterns: Шаблоны исключения

    Returns:
        Словарь UID -> список файлов
    """
    if not HAS_PYDICOM:
        raise RuntimeError("Группировка по UID требует установленный пакет pydicom")

    studies: dict[str, list[str]] = defaultdict(list)

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
            if (0x0020, 0x000D) in ds:
                uid = str(ds[0x0020, 0x000D].value).strip()
            if uid:
                studies[uid].append(str(p))
        except Exception as e:
            logger.debug("Ошибка обработки файла %s: %s", p, e)
            continue

    return studies


def process_study_dir(
    study_path: str,
    config: WorkerConfig,
    debug: bool = False,
) -> StudyResult:
    """
    Обрабатывает исследование по директории.

    Args:
        study_path: Путь к исследованию
        config: Конфигурация воркера
        debug: Режим отладки

    Returns:
        Результат анализа исследования
    """
    non_anon_patients: set[str] = set()
    patient_ids: set[str] = set()
    errors: list[str] = []
    modalities: list[str] = []
    label_sources: set[str] = set()
    series_raw: dict[str, dict[str, str | set[str] | int]] = {}
    file_count = 0
    has_label = False
    age_groups: set[str] = set()
    quality_issues: dict[str, int] = {}
    study_date: str = ""

    dicom_files: list[Path] = []
    try:
        for f in Path(study_path).rglob("*"):
            if not f.is_file():
                continue
            if should_exclude(f, config.exclude_patterns):
                continue
            if is_dicom_file(f):
                dicom_files.append(f)
                file_count += 1
    except Exception as e:
        errors.append(f"Ошибка обхода директории {study_path}: {e!s}")

    for f in dicom_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            modality = extract_modality(ds)
            if modality and modality != "UNKNOWN":
                modalities.append(modality)
                if modality in {"RTSTRUCT", "SEG", "RTSEGANN"}:
                    has_label = True

            # Проверка анонимности
            is_anon, _, pn_info = check_dicom_anonymization(str(f))
            if not is_anon and pn_info:
                patient_name = pn_info.replace("PatientName: '", "").rstrip("'")
                non_anon_patients.add(patient_name)
                non_anon_files.append(str(f.relative_to(Path(study_path)) if Path(study_path) in Path(f).parents else str(f)))

            # Сбор StudyDate для аналитики
            study_date_tag = ds.get((0x0008, 0x0020), None)
            if study_date_tag and study_date_tag.value:
                study_date = str(study_date_tag.value).strip()
                if study_date and not result.study_date:
                    result.study_date = study_date[:8] if len(study_date) >= 8 else study_date

            # Сбор PatientAge для аналитики
            patient_age_tag = ds.get((0x0010, 0x1010), None)
            if patient_age_tag and patient_age_tag.value:
                age_str = str(patient_age_tag.value).strip()
                # Извлечение возраста (может быть в формате "045Y" или просто "45")
                age_value = None
                if age_str:
                    import re
                    match = re.search(r'(\d+)', age_str)
                    if match:
                        age_value = int(match.group(1))
                
                if age_value is not None:
                    # Группировка по возрастным категориям
                    if age_value < 18:
                        age_group = "0-17"
                    elif age_value < 30:
                        age_group = "18-29"
                    elif age_value < 45:
                        age_group = "30-44"
                    elif age_value < 60:
                        age_group = "45-59"
                    elif age_value < 75:
                        age_group = "60-74"
                    else:
                        age_group = "75+"
                    age_groups.add(age_group)

            # Детекция разметки
            detect_label_from_dataset(ds, label_sources)

            # Сбор информации о сериях
            series_uid = str(ds.get((0x0020, 0x000E), "")).strip()
            if series_uid:
                if series_uid not in series_raw:
                    series_raw[series_uid] = {
                        "description": "",
                        "modalities": set(),
                        "files": 0,
                    }
                files_count = series_raw[series_uid]["files"]
                assert isinstance(files_count, int)
                series_raw[series_uid]["files"] = files_count + 1
                modalities_set = series_raw[series_uid]["modalities"]
                assert isinstance(modalities_set, set)
                if modality:
                    modalities_set.add(modality)
                series_desc = str(ds.get((0x0008, 0x103E), "")).strip()
                if series_desc and not series_raw[series_uid]["description"]:
                    series_raw[series_uid]["description"] = series_desc

        except Exception as e:
            errors.append(f"{f}: {e!s}")
            quality_issues["read_errors"] = quality_issues.get("read_errors", 0) + 1

    # Дополнительные признаки разметки
    label_files = list(Path(study_path).rglob("*.seg.nrrd"))
    if label_files:
        has_label = True
        label_sources.add("seg_nrrd_file")

    for jf in Path(study_path).rglob("*.json"):
        try:
            if is_label_json(jf):
                has_label = True
                label_sources.add("label_json")
                break
        except Exception as e:
            logger.debug("Ошибка проверки JSON файла %s: %s", jf, e)
            pass

    # Проверка файлов масок по шаблонам
    for pattern in config.detect_label_file_patterns or LABEL_FILE_PATTERNS:
        for mf in Path(study_path).rglob(pattern):
            if mf.is_file():
                has_label = True
                label_sources.add(f"file_pattern:{pattern}")
                break
        if has_label:
            break

    if debug:
        logger.info("[DIR] %s -> has_label=%s, modalities=%s", study_path, has_label, sorted(set(modalities)))

    series_serializable: dict[str, dict[str, str | int | list[str]]] = {}
    for uid, info in series_raw.items():
        modalities_set = info.get("modalities", set())
        assert isinstance(modalities_set, set)
        modalities_list = sorted(modalities_set)
        files_count = info.get("files", 0)
        assert isinstance(files_count, int)
        description = info.get("description", "")
        assert isinstance(description, str)
        series_serializable[uid] = {
            "description": description,
            "modalities": modalities_list,
            "files": files_count,
        }

    return StudyResult(
        study_key=study_path,
        has_label=has_label,
        non_anon_patients=sorted(non_anon_patients),
        non_anon_files=non_anon_files[:10],  # Ограничим количество файлов для производительности
        modalities=modalities,
        errors=errors,
        study_path_rep=study_path,
        label_sources=label_sources,
        series=series_serializable,
        file_count=file_count,
        patient_ids=sorted(patient_ids),
        study_date=study_date,
    )


def process_study_uid(
    uid: str,
    files: list[str],
    config: WorkerConfig,
    debug: bool = False,
) -> StudyResult:
    """
    Обрабатывает исследование по UID.

    Args:
        uid: StudyInstanceUID
        files: Список файлов
        config: Конфигурация воркера
        debug: Режим отладки

    Returns:
        Результат анализа исследования
    """
    non_anon_patients: set[str] = set()
    patient_ids: set[str] = set()
    errors: list[str] = []
    modalities: list[str] = []
    label_sources: set[str] = set()
    series_raw: dict[str, dict[str, str | set[str] | int]] = {}
    file_count = len(files)
    has_label = False

    for f_path in files:
        try:
            ds = pydicom.dcmread(f_path, stop_before_pixels=True, force=True)
            modality = extract_modality(ds)
            if modality and modality != "UNKNOWN":
                modalities.append(modality)
                if modality in {"RTSTRUCT", "SEG", "RTSEGANN"}:
                    has_label = True

            is_anon, _, pn_info = check_dicom_anonymization(f_path)
            if not is_anon and pn_info:
                non_anon_patients.add(pn_info.replace("PatientName: '", "").rstrip("'"))

            detect_label_from_dataset(ds, label_sources)

            series_uid = str(ds.get((0x0020, 0x000E), "")).strip()
            if series_uid:
                if series_uid not in series_raw:
                    series_raw[series_uid] = {
                        "description": "",
                        "modalities": set(),
                        "files": 0,
                    }
                files_count = series_raw[series_uid]["files"]
                assert isinstance(files_count, int)
                series_raw[series_uid]["files"] = files_count + 1
                modalities_set = series_raw[series_uid]["modalities"]
                assert isinstance(modalities_set, set)
                if modality:
                    modalities_set.add(modality)
                series_desc = str(ds.get((0x0008, 0x103E), "")).strip()
                if series_desc and not series_raw[series_uid]["description"]:
                    series_raw[series_uid]["description"] = series_desc

        except Exception as e:
            errors.append(f"{f_path}: {e!s}")

    # Проверка на наличие файлов разметки в той же директории
    if files:
        base_dir = Path(files[0]).parent
        for pattern in config.detect_label_file_patterns or LABEL_FILE_PATTERNS:
            for mf in base_dir.rglob(pattern):
                if mf.is_file():
                    has_label = True
                    label_sources.add(f"file_pattern:{pattern}")
                    break
            if has_label:
                break

    if debug:
        logger.info(
            "[UID] %s -> has_label=%s, files=%d, modalities=%s", uid, has_label, len(files), sorted(set(modalities))
        )

    series_serializable: dict[str, dict[str, str | int | list[str]]] = {}
    for uid_series, info in series_raw.items():
        modalities_set = info.get("modalities", set())
        assert isinstance(modalities_set, set)
        modalities_list = sorted(modalities_set)
        files_count = info.get("files", 0)
        assert isinstance(files_count, int)
        description = info.get("description", "")
        assert isinstance(description, str)
        series_serializable[uid_series] = {
            "description": description,
            "modalities": modalities_list,
            "files": files_count,
        }

    # Получаем PatientID из первого файла
    patient_id = ""
    if files:
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
            patient_id = str(ds.get((0x0010, 0x0020), "")).strip()
            if patient_id:
                patient_ids.add(patient_id)
        except Exception as e:
            logger.debug("Ошибка чтения PatientID из %s: %s", files[0], e)
            pass

    return StudyResult(
        study_key=uid,
        has_label=has_label,
        non_anon_patients=sorted(non_anon_patients),
        modalities=modalities,
        errors=errors,
        study_path_rep=uid,
        label_sources=label_sources,
        series=series_serializable,
        file_count=file_count,
        patient_ids=sorted(patient_ids),
    )


def load_config_file(path: str) -> dict[str, Any]:
    """
    Загружает конфигурацию из YAML или JSON файла.

    Args:
        path: Путь к файлу конфигурации

    Returns:
        Словарь конфигурации
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    suffix = path_obj.suffix.lower()

    with open(path_obj, encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            if not HAS_YAML:
                raise ImportError("Для загрузки YAML требуется пакет PyYAML")
            result = yaml.safe_load(f)
            return result if isinstance(result, dict) else {}
        elif suffix == ".json":
            return json.load(f)  # type: ignore[no-any-return]
        else:
            raise ValueError(f"Неподдерживаемый формат конфигурации: {suffix}")


def save_report_txt(report: AnalysisReport, output_path: str) -> None:
    """Сохраняет отчёт в текстовом формате."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Результаты анализа DICOM-датасета ===\n\n")
        f.write(f"Всего исследований: {report.total_studies}\n")
        f.write(f"Обработано после фильтров: {report.processed_studies}\n")
        if report.filtered_out_studies > 0:
            f.write(f"Отфильтровано: {report.filtered_out_studies}\n")
        f.write(f"Размеченных: {report.labeled_studies}\n")
        f.write(f"Неанонимизированных: {report.non_anon_studies}\n")
        f.write(f"Всего DICOM-файлов: {report.total_dicom_files}\n")
        f.write(f"Уникальных пациентов: {report.unique_patients}\n\n")

        if report.modality_stats:
            f.write("Статистика по Modality:\n")
            for modality, count in sorted(report.modality_stats.items()):
                f.write(f"  {modality}: {count}\n")
            f.write("\n")

        if report.label_source_stats:
            f.write("Источники определения разметки:\n")
            for source, count in sorted(report.label_source_stats.items()):
                f.write(f"  {source}: {count}\n")
            f.write("\n")

        if report.errors:
            f.write(f"Ошибки ({len(report.errors)}):\n")
            for err in report.errors[:20]:
                f.write(f"  - {err}\n")
            if len(report.errors) > 20:
                f.write(f"  ... и ещё {len(report.errors) - 20} ошибок\n")


def save_report_csv(report: AnalysisReport, output_path: str) -> None:
    """Сохраняет отчёт в CSV формате."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["study_key", "has_label", "non_anon_patients", "modalities", "file_count", "patient_ids", "errors"]
        )
        for result in report.results:
            writer.writerow(
                [
                    result.study_key,
                    result.has_label,
                    ";".join(result.non_anon_patients),
                    ";".join(result.modalities),
                    result.file_count,
                    ";".join(result.patient_ids),
                    ";".join(result.errors),
                ]
            )


def save_report_json(report: AnalysisReport, output_path: str) -> None:
    """Сохраняет отчёт в JSON формате."""
    from dataclasses import asdict

    output = {
        "summary": {
            "total_studies": report.total_studies,
            "processed_studies": report.processed_studies,
            "filtered_out_studies": report.filtered_out_studies,
            "labeled_studies": report.labeled_studies,
            "non_anon_studies": report.non_anon_studies,
            "total_dicom_files": report.total_dicom_files,
            "unique_patients": report.unique_patients,
            "modality_stats": report.modality_stats,
            "label_source_stats": report.label_source_stats,
        },
        "results": [asdict(r) for r in report.results],
        "errors": report.errors,
        "empty_folders": report.empty_folders,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
