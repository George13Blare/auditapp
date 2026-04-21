#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функционала нормализации DICOM-датасетов.
Симулирует создание тестовых DICOM файлов включая SEG маски.
"""

import tempfile
import shutil
from pathlib import Path
import json
import sys

# Добавляем путь к модулю
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dcmmetatest.normalizer import (
    normalize_dataset,
    split_dataset,
    analyze_segmentation_masks,
    load_class_dictionary,
    save_class_dictionary,
    extract_segmentation_info,
    NormalizationConfig,
    SplitConfig,
)


def create_test_dicom_file(filepath: Path, modality: str = "CT", 
                           patient_id: str = "TEST001",
                           study_uid: str = "1.2.3.4.5",
                           series_uid: str = "1.2.3.4.5.1",
                           is_seg: bool = False,
                           segment_classes: list = None):
    """
    Создаёт тестовый DICOM файл с минимально необходимыми атрибутами.
    Для SEG файлов добавляет информацию о классах сегментации.
    """
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian
        import datetime
    except ImportError:
        print("⚠️ pydicom не установлен, пропускаем создание DICOM")
        return False

    # Создаём минимальный dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" if not is_seg else "1.2.840.10008.5.1.4.1.1.66.4"
    file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.5.{series_uid.split('.')[-1]}.{len(str(filepath))}"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    
    # Базовые атрибуты
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = modality
    ds.StudyDescription = "Test Study"
    ds.SeriesDescription = "Test Series"
    ds.PatientName = "Test^Patient"
    ds.PatientBirthDate = "19900101"
    ds.PatientSex = "M"
    ds.StudyDate = datetime.date.today().strftime("%Y%m%d")
    ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    ds.InstitutionName = "Test Hospital"
    ds.Manufacturer = "Test Manufacturer"
    
    if is_seg and segment_classes:
        # Добавляем Segment Sequence для SEG файла
        segment_sequence = []
        for idx, cls_info in enumerate(segment_classes):
            segment = Dataset()
            segment.SegmentNumber = idx + 1
            segment.SegmentLabel = cls_info.get('label', f'Seg{idx+1}')
            segment.SegmentDescription = cls_info.get('description', f'Segment {idx+1}')
            segment.SegmentAlgorithmType = "MANUAL"
            segment.SegmentAlgorithmName = "Test Algorithm"
            
            # Recommended Display CIELab Value - должен быть списком целых чисел
            color = cls_info.get('color', (255, 255, 255))
            segment.RecommendedDisplayCIELabValue = [int(c) for c in color]
            
            segment_sequence.append(segment)
        
        ds.SegmentSequence = segment_sequence
        ds.ContentQualification = "PRODUCT"
        ds.ImageType = ["DERIVED", "PRIMARY", "SEGMENTATION"]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = 512
        ds.Columns = 512
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
    
    # Сохраняем файл
    ds.save_as(filepath, write_like_original=False)
    return True


def create_test_dataset(base_dir: Path) -> dict:
    """
    Создаёт тестовый датасет с различной структурой:
    - Несколько пациентов
    - Несколько исследований на пациента
    - Разные модальности (CT, MR, SEG)
    - Сегментации с классами
    """
    print("\n📁 Создание тестового датасета...")
    
    # Структура: raw_data/PATIENT_STUDY_SERIES/
    patients = [
        {"id": "PAT001", "studies": ["STUDY001", "STUDY002"]},
        {"id": "PAT002", "studies": ["STUDY003"]},
        {"id": "PAT003", "studies": ["STUDY004", "STUDY005"]},
    ]
    
    segment_classes = [
        {"label": "Liver", "description": "Печень", "color": (255, 200, 100)},
        {"label": "Tumor", "description": "Опухоль", "color": (255, 0, 0)},
        {"label": "Vessel", "description": "Сосуд", "color": (0, 0, 255)},
    ]
    
    files_created = {"ct": 0, "mr": 0, "seg": 0}
    
    for patient in patients:
        for study_idx, study_uid in enumerate(patient["studies"]):
            # Создаём директорию для исследования
            study_dir = base_dir / f"{patient['id']}_{study_idx}" / study_uid
            study_dir.mkdir(parents=True, exist_ok=True)
            
            # Создаём CT серию
            ct_series_uid = f"1.2.3.{patient['id']}.{study_uid}.1"
            ct_dir = study_dir / f"CT_{ct_series_uid.split('.')[-1]}"
            ct_dir.mkdir(exist_ok=True)
            
            for i in range(5):  # 5 срезов CT
                ct_file = ct_dir / f"image_{i:03d}.dcm"
                if create_test_dicom_file(
                    ct_file, 
                    modality="CT",
                    patient_id=patient["id"],
                    study_uid=study_uid,
                    series_uid=ct_series_uid
                ):
                    files_created["ct"] += 1
            
            # Создаём MR серию для некоторых пациентов
            if study_idx == 0:
                mr_series_uid = f"1.2.3.{patient['id']}.{study_uid}.2"
                mr_dir = study_dir / f"MR_{mr_series_uid.split('.')[-1]}"
                mr_dir.mkdir(exist_ok=True)
                
                for i in range(3):  # 3 среза MR
                    mr_file = mr_dir / f"image_{i:03d}.dcm"
                    if create_test_dicom_file(
                        mr_file,
                        modality="MR",
                        patient_id=patient["id"],
                        study_uid=study_uid,
                        series_uid=mr_series_uid
                    ):
                        files_created["mr"] += 1
                
                # Создаём SEG файл для первого исследования каждого пациента
                seg_series_uid = f"1.2.3.{patient['id']}.{study_uid}.3"
                seg_dir = study_dir / f"SEG_{seg_series_uid.split('.')[-1]}"
                seg_dir.mkdir(exist_ok=True)
                
                seg_file = seg_dir / "segmentation.dcm"
                if create_test_dicom_file(
                    seg_file,
                    modality="SEG",
                    patient_id=patient["id"],
                    study_uid=study_uid,
                    series_uid=seg_series_uid,
                    is_seg=True,
                    segment_classes=segment_classes
                ):
                    files_created["seg"] += 1
    
    print(f"✅ Создано файлов: CT={files_created['ct']}, MR={files_created['mr']}, SEG={files_created['seg']}")
    print(f"📊 Всего пациентов: {len(patients)}")
    print(f"📊 Всего исследований: {sum(len(p['studies']) for p in patients)}")
    
    return files_created


def test_normalization(test_dir: Path):
    """Тестирует функцию нормализации датасета."""
    print("\n🔧 Тестирование нормализации...")
    
    input_dir = test_dir / "raw_dataset"
    output_dir = test_dir / "normalized_dataset"
    
    config = NormalizationConfig(
        target_structure="patient_study_series",
        rename_files=True,
        normalize_modalities=True,
        extract_metadata=True,
        process_segmentations=True,
        allowed_modalities=["CT", "MR", "PT", "SEG"],
    )
    
    stats = normalize_dataset(input_dir, output_dir, config)
    
    print(f"✅ Нормализация завершена:")
    print(f"   - Всего файлов: {stats.total_files}")
    print(f"   - Обработано: {stats.processed_files}")
    print(f"   - Пациентов: {stats.total_patients}")
    print(f"   - Исследований: {stats.total_studies}")
    print(f"   - Найдено сегментаций: {stats.segmentations_found}")
    print(f"   - Ошибок: {stats.failed_files}")
    
    # Проверка структуры
    assert output_dir.exists(), "Выходная директория не создана"
    patient_dirs = list(output_dir.glob("patient_*"))
    assert len(patient_dirs) == stats.total_patients, f"Директории пациентов не созданы (ожидалось {stats.total_patients}, найдено {len(patient_dirs)})"
    print(f"✅ Создано директорий пациентов: {len(patient_dirs)}")
    
    # Проверка метаданных
    metadata_file = output_dir / "metadata.json"
    assert metadata_file.exists(), "Файл metadata.json не создан"
    with open(metadata_file) as f:
        metadata = json.load(f)
    assert "patients" in metadata, "Отсутствует поле patients в metadata.json"
    print(f"✅ Metadata.json создан корректно")
    
    return stats


def test_split(normalized_dir: Path, test_dir: Path):
    """Тестирует функцию разделения датасета."""
    print("\n🔀 Тестирование разделения (split)...")
    
    output_dir = test_dir / "split_dataset"
    
    config = SplitConfig(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        stratify_by="patient",
        create_manifest=True,
    )
    
    stats = split_dataset(normalized_dir, output_dir, config)
    
    print(f"✅ Разделение завершено:")
    print(f"   - Train: {stats.train_patients} пациентов, {stats.train_samples} сэмплов")
    print(f"   - Val: {stats.val_patients} пациентов, {stats.val_samples} сэмплов")
    print(f"   - Test: {stats.test_patients} пациентов, {stats.test_samples} сэмплов")
    
    # Проверка структуры
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir_split = output_dir / "test"
    
    assert train_dir.exists(), "Train директория не создана"
    assert val_dir.exists(), "Val директория не создана"
    assert test_dir_split.exists(), "Test директория не создана"
    
    # Проверка манифеста
    manifest_file = output_dir / "split_manifest.json"
    assert manifest_file.exists(), "Файл split_manifest.json не создан"
    with open(manifest_file) as f:
        manifest = json.load(f)
    assert "train_patients" in manifest, "Отсутствует поле train_patients"
    assert "val_patients" in manifest, "Отсутствует поле val_patients"
    assert "test_patients" in manifest, "Отсутствует поле test_patients"
    print(f"✅ Split manifest создан корректно")
    
    return stats


def test_segmentation_analysis(normalized_dir: Path):
    """Тестирует анализ масок сегментации."""
    print("\n🏷️ Тестирование анализа сегментаций...")
    
    result = analyze_segmentation_masks(normalized_dir)
    
    print(f"✅ Анализ завершён:")
    print(f"   - Найдено масок: {result['total_masks']}")
    print(f"   - Уникальных классов: {len(result['unique_classes'])}")
    print(f"   - Классы: {', '.join(result['unique_classes'])}")
    
    if result['masks']:
        first_mask = result['masks'][0]
        print(f"\n📋 Информация о первой маске:")
        print(f"   - Файл: {Path(first_mask['file_path']).name}")
        print(f"   - Пациент: {first_mask['patient_id']}")
        print(f"   - Классов в маске: {len(first_mask['classes'])}")
        for cls in first_mask['classes']:
            print(f"      • {cls['class_name']} (ID: {cls['segment_id']})")
    
    assert result['total_masks'] > 0, "Маски сегментации не найдены"
    assert len(result['unique_classes']) > 0, "Классы сегментации не найдены"
    
    return result


def test_class_dictionary(test_dir: Path):
    """Тестирует загрузку и сохранение словаря классов."""
    print("\n📖 Тестирование словаря классов...")
    
    # Создаём тестовый словарь
    test_classes = {
        "Liver": {"segment_id": 1, "class_name": "Liver", "description": "Печень", "color": (255, 200, 100)},
        "Tumor": {"segment_id": 2, "class_name": "Tumor", "description": "Опухоль", "color": (255, 0, 0)},
        "Vessel": {"segment_id": 3, "class_name": "Vessel", "description": "Сосуд", "color": (0, 0, 255)},
    }
    
    # Сохраняем словарь
    json_path = test_dir / "test_class_dict.json"
    success = save_class_dictionary(test_classes, json_path)
    assert success, "Не удалось сохранить словарь классов"
    print(f"✅ Словарь сохранён в {json_path}")
    
    # Загружаем словарь
    loaded_dict = load_class_dictionary(json_path)
    assert len(loaded_dict) == 3, "Неверное количество загруженных классов"
    assert "Liver" in loaded_dict, "Класс Liver не найден"
    assert loaded_dict["Liver"].segment_id == 1, "Неверный segment_id для Liver"
    print(f"✅ Словарь загружен корректно: {len(loaded_dict)} классов")
    
    # Проверяем содержимое JSON файла
    with open(json_path) as f:
        data = json.load(f)
    assert "classes" in data, "Отсутствует поле classes в JSON"
    assert len(data["classes"]) == 3, "Неверное количество классов в JSON"
    print(f"✅ JSON файл имеет корректную структуру")
    
    return loaded_dict


def test_extract_segmentation_info(normalized_dir: Path):
    """Тестирует извлечение информации о конкретной маске."""
    print("\n🔍 Тестирование извлечения информации о SEG файле...")
    
    # Находим первый SEG файл
    seg_files = list(normalized_dir.rglob("*.dcm"))
    seg_file = None
    
    for f in seg_files:
        info = extract_segmentation_info(f)
        if info:
            seg_file = f
            print(f"✅ Найден SEG файл: {f.name}")
            print(f"   - Пациент: {info.patient_id}")
            print(f"   - Исследование: {info.study_id}")
            print(f"   - Классов: {len(info.classes)}")
            for cls in info.classes:
                print(f"      • {cls.class_name} (ID: {cls.segment_id})")
            break
    
    if seg_file:
        info = extract_segmentation_info(seg_file)
        assert info is not None, "Не удалось извлечь информацию о SEG файле"
        assert len(info.classes) > 0, "Классы в SEG файле не найдены"
    else:
        print("⚠️ SEG файлы не найдены")
    
    return info if seg_file else None


def main():
    """Основная функция тестирования."""
    print("=" * 70)
    print("🧪 ВСЕСТОРОННЕЕ ТЕСТИРОВАНИЕ ФУНКЦИОНАЛА НОРМАЛИЗАЦИИ DICOM")
    print("=" * 70)
    
    # Создаём временную директорию
    test_dir = Path(tempfile.mkdtemp(prefix="dicom_test_"))
    print(f"\n📂 Временная директория: {test_dir}")
    
    try:
        # 1. Создание тестового датасета
        files_created = create_test_dataset(test_dir / "raw_dataset")
        
        # 2. Тестирование нормализации
        norm_stats = test_normalization(test_dir / "raw_dataset")
        
        # 3. Тестирование разделения (используем output из нормализации)
        normalized_dir = test_dir / "normalized_dataset"
        split_stats = test_split(normalized_dir, test_dir)
        
        # 4. Тестирование анализа сегментаций
        seg_result = test_segmentation_analysis(normalized_dir)
        
        # 5. Тестирование словаря классов
        class_dict = test_class_dictionary(test_dir)
        
        # 6. Тестирование извлечения информации о SEG
        seg_info = test_extract_segmentation_info(normalized_dir)
        
        # Итоговый отчёт
        print("\n" + "=" * 70)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНУ УСПЕШНО!")
        print("=" * 70)
        print(f"\n📊 Итоговая статистика:")
        print(f"   - Создано тестовых файлов: {sum(files_created.values())}")
        print(f"   - Нормализовано пациентов: {norm_stats.total_patients}")
        print(f"   - Разделено на train/val/test: {split_stats.train_patients + split_stats.val_patients + split_stats.test_patients}")
        print(f"   - Найдено масок сегментации: {seg_result['total_masks']}")
        print(f"   - Уникальных классов: {len(seg_result['unique_classes'])}")
        print(f"   - Протестировано словарей классов: 1")
        
        print(f"\n💡 Функционал готов к использованию в Streamlit приложении!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ТЕСТИРОВАНИИ: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Очистка временной директории
        print(f"\n🧹 Очистка временной директории...")
        shutil.rmtree(test_dir, ignore_errors=True)
        print("✅ Очистка завершена")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
