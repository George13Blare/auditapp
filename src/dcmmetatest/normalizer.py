"""Модуль для нормализации и разделения DICOM-датасетов."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ClassMapping:
    """Маппинг классов сегментации."""

    class_name: str
    class_id: int
    description: str = ""
    color: tuple[int, int, int] = (255, 255, 255)


@dataclass
class SegmentationMask:
    """Информация о маске сегментации."""

    file_path: str
    mask_name: str
    classes: list[ClassMapping] = field(default_factory=list)
    modality: str = "SEG"
    series_description: str = ""


@dataclass
class NormalizationConfig:
    """Конфигурация нормализации."""

    # Целевая структура
    target_structure: str = "patient_study_series"  # patient_study_series, flat, custom
    
    # Переименование файлов
    rename_files: bool = True
    file_pattern: str = "{patient}_{study}_{series}_{index}.dcm"
    
    # Обработка модальностей
    normalize_modalities: bool = True
    allowed_modalities: list[str] = field(default_factory=lambda: ["CT", "MR", "PT", "SEG"])
    
    # Обработка метаданных
    extract_metadata: bool = True
    metadata_file: str = "metadata.json"
    
    # Сегментация
    process_segmentations: bool = True
    segmentation_output_dir: str = "segmentations"


@dataclass
class SplitConfig:
    """Конфигурация разделения датасета."""

    # Пропорции сплита
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Стратификация
    stratify_by: str = "patient"  # patient, study, modality, label
    ensure_balance: bool = True
    
    # Фильтры
    min_files_per_study: int = 1
    require_segmentation: bool = False
    modalities_to_include: list[str] = field(default_factory=list)
    
    # Вывод
    output_dir: str = ""
    create_manifest: bool = True
    seed: int = 42


@dataclass
class NormalizationStats:
    """Статистика нормализации."""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_studies: int = 0
    total_patients: int = 0
    segmentations_found: int = 0
    classes_extracted: int = 0
    output_structure: dict = field(default_factory=dict)


@dataclass
class SplitStats:
    """Статистика разделения."""

    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_patients: int = 0
    val_patients: int = 0
    test_patients: int = 0
    split_manifest: dict = field(default_factory=dict)


def extract_segmentation_info(dcm_path: Path) -> SegmentationMask | None:
    """
    Извлекает информацию о маске сегментации из DICOM SEG файла.
    
    Args:
        dcm_path: Путь к DICOM файлу
        
    Returns:
        SegmentationMask или None
    """
    try:
        import pydicom
        
        ds = pydicom.dcmread(dcm_path, force=True)
        
        # Проверка, что это SEG
        if not hasattr(ds, 'Modality') or ds.Modality != 'SEG':
            return None
        
        mask = SegmentationMask(
            file_path=str(dcm_path),
            mask_name=getattr(ds, 'SeriesDescription', f'SEG_{dcm_path.stem}'),
            modality='SEG',
            series_description=getattr(ds, 'SeriesDescription', ''),
        )
        
        # Извлечение информации о классах из Segment Sequence
        if hasattr(ds, 'SegmentSequence'):
            for idx, segment in enumerate(ds.SegmentSequence):
                class_mapping = ClassMapping(
                    class_name=getattr(segment, 'SegmentLabel', f'Class_{idx}'),
                    class_id=idx + 1,
                    description=getattr(segment, 'SegmentDescription', ''),
                )
                
                # Попытка извлечь цвет
                if hasattr(segment, 'RecommendedDisplayCIELabValue'):
                    # Конвертация из CIELab в RGB (упрощённо)
                    class_mapping.color = (255, 255, 255)  # Заглушка
                    
                mask.classes.append(class_mapping)
        
        return mask
        
    except Exception as e:
        logger.warning(f"Ошибка при чтении сегментации {dcm_path}: {e}")
        return None


def load_class_dictionary(json_path: Path) -> dict[str, ClassMapping]:
    """
    Загружает словарь классов из JSON файла.
    
    Ожидаемый формат:
    {
        "classes": [
            {"id": 1, "name": "Liver", "description": "...", "color": [255, 0, 0]},
            ...
        ]
    }
    
    Args:
        json_path: Путь к JSON файлу
        
    Returns:
        Словарь маппинга классов
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        class_dict = {}
        for cls in data.get('classes', []):
            mapping = ClassMapping(
                class_name=cls.get('name', f"Class_{cls.get('id')}"),
                class_id=cls.get('id', 0),
                description=cls.get('description', ''),
                color=tuple(cls.get('color', [255, 255, 255])),
            )
            class_dict[mapping.class_name] = mapping
        
        return class_dict
        
    except Exception as e:
        logger.error(f"Ошибка загрузки словаря классов: {e}")
        return {}


def save_class_dictionary(class_dict: dict[str, ClassMapping], json_path: Path) -> bool:
    """
    Сохраняет словарь классов в JSON файл.
    
    Args:
        class_dict: Словарь маппинга классов
        json_path: Путь для сохранения
        
    Returns:
        True если успешно
    """
    try:
        data = {
            'classes': [
                {
                    'id': mapping.class_id,
                    'name': mapping.class_name,
                    'description': mapping.description,
                    'color': list(mapping.color),
                }
                for mapping in class_dict.values()
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка сохранения словаря классов: {e}")
        return False


def normalize_dataset(
    input_path: str,
    output_path: str,
    config: NormalizationConfig,
    class_dict_path: str | None = None,
) -> NormalizationStats:
    """
    Нормализует структуру DICOM-датасета.
    
    Args:
        input_path: Путь к исходному датасету
        output_path: Путь для нормализованного датасета
        config: Конфигурация нормализации
        class_dict_path: Опциональный путь к словарю классов
        
    Returns:
        Статистика нормализации
    """
    stats = NormalizationStats()
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    
    # Загрузка словаря классов если указан
    class_dict = {}
    if class_dict_path and Path(class_dict_path).exists():
        class_dict = load_class_dictionary(Path(class_dict_path))
    
    # Создание выходной директории
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сбор информации о файлах
    all_files = list(input_dir.rglob('*'))
    dcm_files = [f for f in all_files if f.suffix.lower() in ['.dcm', '.dicom', '']]
    seg_files = []
    
    stats.total_files = len(dcm_files)
    
    # Группировка по пациентам/исследованиям
    patient_studies = {}  # {patient_id: {study_id: [files]}}
    
    for dcm_file in dcm_files:
        try:
            import pydicom
            
            ds = pydicom.dcmread(dcm_file, force=True)
            
            patient_id = getattr(ds, 'PatientID', 'UNKNOWN')
            study_id = getattr(ds, 'StudyInstanceUID', 'UNKNOWN')
            series_id = getattr(ds, 'SeriesInstanceUID', 'UNKNOWN')
            
            if patient_id not in patient_studies:
                patient_studies[patient_id] = {}
            if study_id not in patient_studies[patient_id]:
                patient_studies[patient_id][study_id] = {'series': {}, 'files': []}
            
            patient_studies[patient_id][study_id]['files'].append(dcm_file)
            
            if series_id not in patient_studies[patient_id][study_id]['series']:
                patient_studies[patient_id][study_id]['series'][series_id] = []
            patient_studies[patient_id][study_id]['series'][series_id].append(dcm_file)
            
            # Проверка на сегментацию
            if hasattr(ds, 'Modality') and ds.Modality == 'SEG':
                seg_files.append(dcm_file)
                stats.segmentations_found += 1
                
        except Exception as e:
            logger.warning(f"Ошибка чтения файла {dcm_file}: {e}")
            stats.failed_files += 1
    
    stats.total_patients = len(patient_studies)
    stats.total_studies = sum(len(studies) for studies in patient_studies.values())
    
    # Нормализация структуры
    if config.target_structure == "patient_study_series":
        for patient_id, studies in patient_studies.items():
            patient_dir = output_dir / f"patient_{patient_id}"
            patient_dir.mkdir(exist_ok=True)
            
            for study_id, study_data in studies.items():
                study_dir = patient_dir / f"study_{study_id}"
                study_dir.mkdir(exist_ok=True)
                
                # Копирование файлов изображений
                for series_id, series_files in study_data['series'].items():
                    series_dir = study_dir / f"series_{series_id}"
                    series_dir.mkdir(exist_ok=True)
                    
                    for idx, src_file in enumerate(series_files):
                        if config.rename_files:
                            new_name = config.file_pattern.format(
                                patient=patient_id,
                                study=study_id[:8],
                                series=series_id[:8],
                                index=idx,
                            )
                            if not new_name.endswith('.dcm'):
                                new_name += '.dcm'
                        else:
                            new_name = src_file.name
                        
                        dst_file = series_dir / new_name
                        try:
                            shutil.copy2(src_file, dst_file)
                            stats.processed_files += 1
                        except Exception as e:
                            logger.error(f"Ошибка копирования {src_file}: {e}")
                            stats.failed_files += 1
                
                # Копирование сегментаций
                if config.process_segmentations and seg_files:
                    seg_dir = study_dir / config.segmentation_output_dir
                    seg_dir.mkdir(exist_ok=True)
                    
                    for seg_file in seg_files:
                        try:
                            shutil.copy2(seg_file, seg_dir / seg_file.name)
                        except Exception as e:
                            logger.error(f"Ошибка копирования сегментации {seg_file}: {e}")
    
    # Сохранение метаданных
    if config.extract_metadata:
        metadata = {
            'total_patients': stats.total_patients,
            'total_studies': stats.total_studies,
            'total_files': stats.processed_files,
            'segmentations': stats.segmentations_found,
            'patients': list(patient_studies.keys()),
        }
        
        metadata_file = output_dir / config.metadata_file
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Сохранение структуры
    stats.output_structure = {
        'patients': stats.total_patients,
        'studies': stats.total_studies,
        'files_processed': stats.processed_files,
    }
    
    return stats


def split_dataset(
    normalized_path: str,
    output_path: str,
    config: SplitConfig,
) -> SplitStats:
    """
    Разделяет нормализованный датасет на train/val/test.
    
    Args:
        normalized_path: Путь к нормализованному датасету
        output_path: Путь для разделённого датасета
        config: Конфигурация разделения
        
    Returns:
        Статистика разделения
    """
    stats = SplitStats()
    input_dir = Path(normalized_path)
    output_dir = Path(output_path)
    
    # Установка seed для воспроизводимости
    random.seed(config.seed)
    
    # Сбор пациентов/исследований
    patients = []
    for item in input_dir.iterdir():
        if item.is_dir() and item.name.startswith('patient_'):
            patients.append(item)
    
    if not patients:
        logger.warning("Пациенты не найдены. Проверьте структуру датасета.")
        return stats
    
    # Перемешивание
    random.shuffle(patients)
    
    # Разделение
    n_total = len(patients)
    n_train = max(1, int(n_total * config.train_ratio))
    n_val = max(1, int(n_total * config.val_ratio))
    n_test = n_total - n_train - n_val
    
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    
    stats.train_patients = len(train_patients)
    stats.val_patients = len(val_patients)
    stats.test_patients = len(test_patients)
    
    # Функция копирования датасета
    def copy_split(source_patients: list[Path], split_name: str) -> int:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        sample_count = 0
        for patient_dir in source_patients:
            try:
                dst_patient_dir = split_dir / patient_dir.name
                shutil.copytree(patient_dir, dst_patient_dir)
                sample_count += 1
            except Exception as e:
                logger.error(f"Ошибка копирования {patient_dir}: {e}")
        
        return sample_count
    
    stats.train_samples = copy_split(train_patients, 'train')
    stats.val_samples = copy_split(val_patients, 'val')
    stats.test_samples = copy_split(test_patients, 'test')
    
    # Создание манифеста
    if config.create_manifest:
        manifest = {
            'config': {
                'train_ratio': config.train_ratio,
                'val_ratio': config.val_ratio,
                'test_ratio': config.test_ratio,
                'seed': config.seed,
            },
            'stats': {
                'train': {'patients': stats.train_patients, 'samples': stats.train_samples},
                'val': {'patients': stats.val_patients, 'samples': stats.val_samples},
                'test': {'patients': stats.test_patients, 'samples': stats.test_samples},
            },
            'train_patients': [p.name for p in train_patients],
            'val_patients': [p.name for p in val_patients],
            'test_patients': [p.name for p in test_patients],
        }
        
        manifest_file = output_dir / 'split_manifest.json'
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        stats.split_manifest = manifest
    
    return stats


def analyze_segmentation_masks(dataset_path: str) -> dict[str, Any]:
    """
    Анализирует маски сегментации в датасете.
    
    Args:
        dataset_path: Путь к датасету
        
    Returns:
        Информация о найденных масках и классах
    """
    dataset_dir = Path(dataset_path)
    masks_info = []
    all_classes = set()
    
    # Поиск SEG файлов
    seg_files = list(dataset_dir.rglob('*.dcm')) + list(dataset_dir.rglob('*.dicom'))
    
    for seg_file in seg_files:
        mask_info = extract_segmentation_info(seg_file)
        if mask_info:
            masks_info.append({
                'file': str(seg_file),
                'name': mask_info.mask_name,
                'classes': [
                    {
                        'id': cls.class_id,
                        'name': cls.class_name,
                        'description': cls.description,
                    }
                    for cls in mask_info.classes
                ],
            })
            
            for cls in mask_info.classes:
                all_classes.add(cls.class_name)
    
    return {
        'total_masks': len(masks_info),
        'masks': masks_info,
        'unique_classes': list(all_classes),
        'total_classes': len(all_classes),
    }
