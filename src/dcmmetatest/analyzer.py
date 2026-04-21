"""Модуль анализа DICOM-датасетов."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .io import (
    find_dicom_studies_by_dir,
    find_dicom_studies_by_uid,
    process_study_dir,
    process_study_uid,
)
from .models import AnalysisReport, StudyResult, WorkerConfig

logger = logging.getLogger(__name__)


def run_analysis(
    folder_path: str,
    config: WorkerConfig,
    debug: bool = False,
) -> AnalysisReport:
    """
    Запускает анализ DICOM-датасета.

    Args:
        folder_path: Путь к корневой директории датасета
        config: Конфигурация воркера
        debug: Режим отладки

    Returns:
        Отчёт по анализу
    """
    report = AnalysisReport()

    # Определение исследований в зависимости от режима группировки
    if config.group_by == "study":
        studies_dict = find_dicom_studies_by_uid(
            folder_path,
            follow_symlinks=config.follow_symlinks,
            max_depth=config.max_depth,
            exclude_patterns=config.exclude_patterns,
        )
        study_items: list[tuple[str, str | list[str]]] = list(studies_dict.items())
    else:
        study_list, empty_folders = find_dicom_studies_by_dir(
            folder_path,
            list_empty=config.list_empty,
            follow_symlinks=config.follow_symlinks,
            max_depth=config.max_depth,
            exclude_patterns=config.exclude_patterns,
        )
        study_items = [(s, s) for s in study_list]
        report.empty_folders = empty_folders

    report.total_studies = len(study_items)

    if not study_items:
        logger.warning("DICOM-исследований не найдено.")
        return report

    # Настройка пула воркеров
    max_workers = config.max_workers
    if max_workers is None:
        import os

        max_workers = os.cpu_count() or 4

    executor_class = ProcessPoolExecutor if config.pool_type == "process" else ThreadPoolExecutor

    results: list[StudyResult] = []

    if config.show_progress and HAS_TQDM:
        tqdm(study_items, desc="Обработка исследований")

    with executor_class(max_workers=max_workers) as executor:
        futures = {}

        for key, value in study_items:
            if config.group_by == "study":
                uid = key
                files = value if isinstance(value, list) else []
                future = executor.submit(process_study_uid, uid, files, config, debug)
            else:
                study_path = value if isinstance(value, str) else str(value)
                future = executor.submit(process_study_dir, study_path, config, debug)
            futures[future] = key

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                key = futures[future]
                error_msg = f"Ошибка обработки {key}: {e!s}"
                logger.error(error_msg)
                report.errors.append(error_msg)
                if config.strict:
                    raise

    report.results = results

    # Применение фильтров
    filtered_results: list[StudyResult] = []
    for r in results:
        # Фильтр по модальности
        if config.modality_filter:
            if not any(m in config.modality_filter for m in r.modalities):
                continue

        # Фильтр только размеченные
        if config.only_labeled and not r.has_label:
            continue

        # Фильтр только неанонимные
        if config.only_non_anon and not r.non_anon_patients:
            continue

        # Фильтр по минимальному количеству файлов
        if config.min_files > 0 and r.file_count < config.min_files:
            continue

        filtered_results.append(r)

    report.filtered_out_studies = len(results) - len(filtered_results)
    report.processed_studies = len(filtered_results)

    # Сбор статистики
    labeled_count = 0
    non_anon_count = 0
    total_files = 0
    all_patient_ids: set[str] = set()
    modality_counter: Counter = Counter()
    label_source_counter: Counter = Counter()
    series_max: list[tuple[str, int]] = []
    patient_study_map: dict[str, list[str]] = defaultdict(list)

    for r in filtered_results:
        if r.has_label:
            labeled_count += 1
        if r.non_anon_patients:
            non_anon_count += 1

        total_files += r.file_count
        all_patient_ids.update(r.patient_ids)

        for mod in r.modalities:
            modality_counter[mod] += 1

        for source in r.label_sources:
            label_source_counter[source] += 1

        for series_uid, series_info in r.series.items():
            files_count = series_info.get("files", 0)
            if isinstance(files_count, int):
                series_max.append((series_uid, files_count))

        # Группировка по пациенту
        for pid in r.patient_ids:
            patient_study_map[pid].append(r.study_key)

    report.labeled_studies = labeled_count
    report.non_anon_studies = non_anon_count
    report.total_dicom_files = total_files
    report.unique_patients = len(all_patient_ids)
    report.modality_stats = dict(modality_counter)
    report.label_source_stats = dict(label_source_counter)

    # Топ серий по количеству файлов
    series_max.sort(key=lambda x: x[1], reverse=True)
    report.series_max_files = series_max[:10]

    # Статистика по пациентам
    report.patient_study_counts = {pid: len(studies) for pid, studies in patient_study_map.items()}

    return report


def print_summary(report: AnalysisReport) -> None:
    """
    Выводит сводку результатов в консоль.

    Args:
        report: Отчёт для вывода
    """
    print("\n=== Результаты анализа ===")
    print(f"Всего исследований (до фильтрации): {report.total_studies}")
    print(f"Обработано после фильтров: {report.processed_studies}")

    if report.filtered_out_studies > 0:
        print(f"Отфильтровано исследований: {report.filtered_out_studies}")

    percent = report.labeled_studies / report.processed_studies if report.processed_studies > 0 else 0.0
    print(f"Размеченных исследований: {report.labeled_studies} ({percent:.1%})")

    avg_files = report.total_dicom_files / report.processed_studies if report.processed_studies > 0 else 0.0
    print(f"Среднее число DICOM-файлов на исследование: {avg_files:.1f}")
    print(f"Уникальных пациентов: {report.unique_patients}")

    if report.label_source_stats:
        print("\nИсточники определения разметки:")
        for source, count in sorted(report.label_source_stats.items()):
            print(f"  {source}: {count}")

    print("\nСтатистика по Modality:")
    for modality, count in sorted(report.modality_stats.items()):
        print(f"  {modality}: {count} файлов")

    if report.series_max_files:
        print("\nСерии с максимальным количеством файлов:")
        for series_uid, count in report.series_max_files[:5]:
            print(f"  {series_uid}: {count} файлов")

    if report.patient_study_counts:
        # Топ пациентов по числу исследований
        top_patients = sorted(report.patient_study_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nПациенты с наибольшим числом исследований:")
        for pid, count in top_patients:
            print(f"  {pid or '[empty]'}: {count} исследований")

    print(f"\nПустых папок найдено: {len(report.empty_folders)}")

    if report.empty_folders:
        print("\nПолный список пустых папок:")
        for folder in report.empty_folders[:20]:
            print(f"  - {folder}")
        if len(report.empty_folders) > 20:
            print(f"  ... и ещё {len(report.empty_folders) - 20}")

    if report.errors:
        print(f"\nОшибки ({len(report.errors)}):")
        for err in report.errors[:10]:
            print(f"  - {err}")
        if len(report.errors) > 10:
            print(f"  ... и ещё {len(report.errors) - 10}")
