"""Утилиты для анализатора DICOM."""

from __future__ import annotations

import fnmatch
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union, Any

try:
    import pydicom
    from pydicom.dataset import Dataset
    HAS_PYDICOM = True
except ImportError:
    pydicom = None  # type: ignore
    HAS_PYDICOM = False

from .detectors import is_dicom_file as _is_dicom_file

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Настраивает логирование.
    
    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу лога (опционально)
    """
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
def read_dicom_header(path: str) -> Optional["Dataset"]:
    """
    Читает заголовок DICOM-файла с кешированием.
    
    Args:
        path: Путь к файлу
        
    Returns:
        Dataset или None если чтение не удалось
    """
    if not HAS_PYDICOM:
        return None
    
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception as exc:
        logger.debug("Не удалось прочитать DICOM %s: %s", path, exc)
        return None


def should_exclude(path: Path, patterns: Tuple[str, ...]) -> bool:
    """
    Проверяет, должен ли путь быть исключён по шаблонам.
    
    Args:
        path: Путь для проверки
        patterns: Шаблоны для исключения
        
    Returns:
        True если путь должен быть исключён
    """
    if not patterns:
        return False
    path_str = str(path)
    for pattern in patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True
    return False


def prompt_with_default(
    prompt: str,
    default: Optional[str] = None,
    validator: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
) -> str:
    """
    Запрашивает ввод у пользователя со значением по умолчанию.
    
    Args:
        prompt: Текст приглашения
        default: Значение по умолчанию
        validator: Функция валидации ввода
        allow_empty: Разрешить пустой ввод
        
    Returns:
        Введённая строка
    """
    while True:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{prompt}{suffix}: ").strip()
        if not answer:
            if default is not None:
                answer = default
            elif allow_empty:
                return ""
            else:
                print("Пожалуйста, введите значение.")
                continue
        if validator and not validator(answer):
            print("Введено некорректное значение, попробуйте ещё раз.")
            continue
        return answer


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Запрашивает ответ да/нет.
    
    Args:
        prompt: Текст вопроса
        default: Значение по умолчанию (True=да, False=нет)
        
    Returns:
        True для 'да', False для 'нет'
    """
    default_str = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "да", "д", "1", "true", "t"}:
            return True
        if answer in {"n", "no", "нет", "н", "0", "false", "f"}:
            return False
        print("Ответ не распознан. Введите 'да' или 'нет'.")


def prompt_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    """
    Запрашивает выбор из списка вариантов.
    
    Args:
        prompt: Текст приглашения
        choices: Список допустимых значений
        default: Значение по умолчанию
        
    Returns:
        Выбранное значение
    """
    choices_display = "/".join(choices)
    while True:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{prompt} ({choices_display}){suffix}: ").strip().lower()
        if not answer and default:
            return default
        if answer in [c.lower() for c in choices]:
            return answer
        print(f"Пожалуйста, выберите одно из значений: {choices_display}.")


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


def check_dicom_anonymization(dicom_file: str) -> Tuple[bool, str, Optional[str]]:
    """
    Проверяет анонимизацию DICOM-файла.
    
    Args:
        dicom_file: Путь к DICOM-файлу
        
    Returns:
        Кортеж (is_anonymous, modality, error_message)
    """
    if not HAS_PYDICOM:
        raise RuntimeError(
            "Для проверки анонимизации требуется установленный пакет pydicom"
        )
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        modality_tag = ds.get((0x0008, 0x0060))
        modality = str(modality_tag.value).strip().upper() if modality_tag else "UNKNOWN"
        anonymous_values = {
            "anonymous", "anon", "anonymized", "unknown", "na", "n/a",
            "none", "xxxx", "yyyy", "zzzz", "анонимно", "аноним",
            "неизвестно", "ai_test", "test", "demo",
        }
        if (0x0010, 0x0010) in ds:
            patient_name = str(ds[0x0010, 0x0010].value or "").lower().strip()
            if patient_name and not any(anon in patient_name for anon in anonymous_values):
                return False, modality, f"PatientName: '{ds[0x0010, 0x0010].value}'"
        return True, modality, None
    except Exception as e:
        return False, "ERROR", f"Ошибка чтения: {str(e)}"
