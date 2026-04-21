"""CLI модуль для анализатора DICOM."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .analyzer import print_summary, run_analysis
from .io import load_config_file, save_report_csv, save_report_json, save_report_txt
from .models import WorkerConfig
from .utils import (
    configure_logging,
    prompt_choice,
    prompt_with_default,
    prompt_yes_no,
)

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Создаёт парсер аргументов командной строки.

    Returns:
        Настроенный ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="dcmmetatest-plus",
        description="Расширенный анализатор DICOM-датасетов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s /path/to/dicom/data --group-by study --format json -o report.json
  %(prog)s /path/to/data --modality CT,MR --only-labeled
  %(prog)s /path/to/data --interactive
        """,
    )

    parser.add_argument(
        "input_path",
        help="Путь к корневой директории с DICOM-данными",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Путь к файлу отчёта (расширение определяет формат)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["txt", "csv", "json"],
        default="txt",
        help="Формат отчёта (по умолчанию: txt)",
    )
    parser.add_argument(
        "--group-by",
        choices=["dir", "study"],
        default="dir",
        help="Режим группировки: dir (по директориям) или study (по StudyInstanceUID)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        help="Количество воркеров (по умолчанию: число CPU)",
    )
    parser.add_argument(
        "--pool-type",
        choices=["process", "thread"],
        default="process",
        help="Тип пула воркеров (по умолчанию: process)",
    )
    parser.add_argument(
        "--modality",
        help="Фильтр по модальностям (через запятую, например: CT,MR)",
    )
    parser.add_argument(
        "--only-labeled",
        action="store_true",
        help="Показывать только размеченные исследования",
    )
    parser.add_argument(
        "--only-non-anon",
        action="store_true",
        help="Показывать только неанонимизированные исследования",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=0,
        help="Минимальное количество файлов в исследовании",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Шаблоны путей для исключения",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Максимальная глубина обхода директорий",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Следовать за символическими ссылками",
    )
    parser.add_argument(
        "--list-empty",
        action="store_true",
        help="Включать пустые директории в отчёт",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Отключить прогресс-бар",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Останавливаться при первой ошибке",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Режим отладки",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Увеличить уровень логирования (-v, -vv)",
    )
    parser.add_argument(
        "--log-file",
        help="Путь к файлу лога",
    )
    parser.add_argument(
        "--config",
        help="Путь к файлу конфигурации (YAML/JSON)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Интерактивный режим настройки",
    )

    return parser


def interactive_setup() -> tuple[WorkerConfig, str, str, str]:
    """
    Запускает интерактивный режим настройки.

    Returns:
        Кортеж (конфигурация, путь_ввода, формат_отчёта, путь_вывода)
    """
    print("\n=== Интерактивный режим настройки ===")
    print("Ответьте на несколько вопросов, чтобы выбрать нужный функционал.\n")

    # Путь к данным
    input_path = prompt_with_default(
        "Путь к директории с DICOM-данными",
        default=".",
    )

    if not Path(input_path).exists():
        print("Указанный путь не найден или не является директорией.")
        sys.exit(1)

    # Режим группировки
    group_by = prompt_choice(
        "Режим группировки исследований",
        choices=["dir", "study"],
        default="dir",
    )

    # Фильтры
    only_labeled = prompt_yes_no("Показывать только размеченные исследования?", default=False)
    only_non_anon = prompt_yes_no("Показывать только неанонимные исследования?", default=False)

    modality_filter = None
    if prompt_yes_no("Фильтровать по модальности?", default=False):
        mod_input = prompt_with_default(
            "Введите модальности через запятую (например, CT,MR)",
            default="",
        )
        if mod_input:
            modality_filter = {m.strip().upper() for m in mod_input.split(",")}

    # Параметры обработки
    max_workers_str = prompt_with_default(
        "Количество воркеров (оставьте пустым для авто)",
        default="",
        allow_empty=True,
    )
    max_workers = int(max_workers_str) if max_workers_str.isdigit() else None

    pool_type = prompt_choice(
        "Тип пула воркеров",
        choices=["process", "thread"],
        default="process",
    )

    follow_symlinks = prompt_yes_no("Следовать за симлинками?", default=False)

    max_depth_str = prompt_with_default(
        "Максимальная глубина обхода (пусто = без ограничений)",
        default="",
        allow_empty=True,
    )
    max_depth = int(max_depth_str) if max_depth_str.isdigit() else None

    list_empty = prompt_yes_no("Включать пустые директории в отчёт?", default=False)
    show_progress = prompt_yes_no("Показывать прогресс-бар?", default=True)

    # Формат отчёта
    report_format = prompt_choice(
        "Формат отчёта",
        choices=["txt", "csv", "json"],
        default="txt",
    )

    output_path = prompt_with_default(
        "Путь к файлу отчёта (пусто = только консоль)",
        default="",
        allow_empty=True,
    )

    print("\nНастройка завершена. Запускаем анализ...\n")

    config = WorkerConfig(
        group_by=group_by,
        only_labeled=only_labeled,
        only_non_anon=only_non_anon,
        modality_filter=modality_filter,
        max_workers=max_workers,
        pool_type=pool_type,
        follow_symlinks=follow_symlinks,
        max_depth=max_depth,
        list_empty=list_empty,
        show_progress=show_progress,
    )
    return config, input_path, report_format, output_path


def main(argv: list[str] | None = None) -> int:
    """
    Точка входа CLI.

    Args:
        argv: Аргументы командной строки (по умолчанию sys.argv[1:])

    Returns:
        Код выхода
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Интерактивный режим
    if args.interactive:
        config, input_path, report_format, output_path = interactive_setup()
        args.input_path = input_path
        args.format = report_format
        args.output = output_path
    else:
        config = WorkerConfig()

    # Загрузка конфигурации из файла
    if args.config:
        try:
            file_config = load_config_file(args.config)
            # Применение настроек из файла
            if "group_by" in file_config:
                config.group_by = file_config["group_by"]
            if "only_labeled" in file_config:
                config.only_labeled = file_config["only_labeled"]
            if "only_non_anon" in file_config:
                config.only_non_anon = file_config["only_non_anon"]
            if "modality_filter" in file_config:
                config.modality_filter = set(file_config["modality_filter"])
            if "max_workers" in file_config:
                config.max_workers = file_config["max_workers"]
            if "pool_type" in file_config:
                config.pool_type = file_config["pool_type"]
            if "follow_symlinks" in file_config:
                config.follow_symlinks = file_config["follow_symlinks"]
            if "max_depth" in file_config:
                config.max_depth = file_config["max_depth"]
            if "list_empty" in file_config:
                config.list_empty = file_config["list_empty"]
            if "show_progress" in file_config:
                config.show_progress = file_config["show_progress"]
            if "exclude_patterns" in file_config:
                config.exclude_patterns = tuple(file_config["exclude_patterns"])
        except Exception as e:
            logger.error("Ошибка загрузки конфигурации: %s", e)
            return 1

    # Переопределение аргументами CLI
    if args.group_by:
        config.group_by = args.group_by
    if args.workers:
        config.max_workers = args.workers
    if args.pool_type:
        config.pool_type = args.pool_type
    if args.modality:
        config.modality_filter = {m.strip().upper() for m in args.modality.split(",")}
    if args.only_labeled:
        config.only_labeled = args.only_labeled
    if args.only_non_anon:
        config.only_non_anon = args.only_non_anon
    if args.min_files:
        config.min_files = args.min_files
    if args.exclude:
        config.exclude_patterns = tuple(args.exclude)
    if args.max_depth is not None:
        config.max_depth = args.max_depth
    if args.follow_symlinks:
        config.follow_symlinks = args.follow_symlinks
    if args.list_empty:
        config.list_empty = args.list_empty
    if args.no_progress:
        config.show_progress = False
    if args.strict:
        config.strict = args.strict

    # Настройка логирования
    log_level = "INFO"
    if args.verbose >= 2:
        log_level = "DEBUG"
    elif args.verbose == 1:
        log_level = "INFO"
    elif args.debug:
        log_level = "DEBUG"

    configure_logging(level=log_level, log_file=args.log_file)

    # Проверка входного пути
    if not Path(args.input_path).exists():
        logger.error("Директория не найдена: %s", args.input_path)
        return 1

    # Запуск анализа
    try:
        report = run_analysis(args.input_path, config, debug=args.debug)
    except Exception as e:
        logger.error("Ошибка анализа: %s", e)
        return 1

    # Вывод результатов
    print_summary(report)

    # Сохранение отчёта
    if args.output:
        output_path = args.output
        fmt = args.format

        # Автоопределение формата по расширению
        if output_path.endswith(".csv"):
            fmt = "csv"
        elif output_path.endswith(".json"):
            fmt = "json"
        elif output_path.endswith(".txt"):
            fmt = "txt"

        try:
            if fmt == "txt":
                save_report_txt(report, output_path)
            elif fmt == "csv":
                save_report_csv(report, output_path)
            elif fmt == "json":
                save_report_json(report, output_path)
            logger.info("Отчёт сохранён: %s", output_path)
        except Exception as e:
            logger.error("Ошибка сохранения отчёта: %s", e)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
