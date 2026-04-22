# Новые функции DCMMETATEST Plus v1.0.0

## 📦 Добавленные модули

### 1. Модуль аугментации (`augmentations.py`)

Полноценный модуль для аугментации медицинских изображений с поддержкой как 2D, так и 3D данных.

#### Поддерживаемые типы аугментаций:

| Тип | Описание | Параметры |
|-----|----------|-----------|
| **Rotation** | Случайные повороты | `rotation_range`, `rotation_prob` |
| **Flip** | Отражения (горизонтальное, вертикальное, аксиальное) | `flip_*_prob` |
| **Elastic Deformation** | Эластичные деформации | `elastic_alpha`, `elastic_sigma`, `elastic_prob` |
| **Gaussian Noise** | Гауссовский шум | `gaussian_noise_mean`, `gaussian_noise_std` |
| **Salt & Pepper Noise** | Импульсный шум | `salt_pepper_amount`, `salt_pepper_prob` |
| **Zoom** | Масштабирование | `zoom_range`, `zoom_prob` |
| **Shift** | Сдвиги | `shift_range`, `shift_prob` |
| **Brightness/Contrast** | Яркость и контраст | `brightness_range`, `contrast_range` |

#### Пример использования:

```python
from src.dcmmetatest import (
    AugmentationConfig,
    apply_augmentation,
    generate_augmented_dataset,
)
import numpy as np

# Создание конфигурации
config = AugmentationConfig(
    rotation_range=(-15, 15),
    rotation_prob=0.5,
    flip_horizontal_prob=0.5,
    elastic_prob=0.3,
    gaussian_noise_prob=0.2,
)

# Применение к изображению
image = np.random.rand(256, 256).astype(np.float32)
mask = (np.random.rand(256, 256) > 0.5).astype(np.uint8)

aug_image, aug_mask = apply_augmentation(
    image,
    mask=mask,
    config=config,
    seed=42,
)

# Генерация аугментированного датасета
stats = generate_augmented_dataset(
    input_dir="./data/original",
    output_dir="./data/augmented",
    config=config,
    num_augmentations=5,
)
```

---

### 2. Модуль генерации конфигов (`config_generator.py`)

Генерация конфигурационных файлов для популярных ML-фреймворков.

#### Поддерживаемые фреймворки:

| Фреймворк | Формат | Назначение |
|-----------|--------|------------|
| **YOLOv8** | YAML | Сегментация и детекция |
| **MONAI** | JSON | Медицинские задачи (сегментация, классификация) |
| **nnU-Net** | JSON | Автоматическая сегментация |
| **Hugging Face** | Python + JSON | Публикация датасетов |

#### Пример использования:

```python
from src.dcmmetatest import (
    DatasetInfo,
    TaskType,
    FrameworkType,
    generate_framework_config,
)

# Информация о датасете
dataset_info = DatasetInfo(
    name="Liver Segmentation",
    description="Датасет для сегментации печени на КТ",
    task_type=TaskType.SEGMENTATION,
    num_classes=3,
    class_names=["background", "liver", "tumor"],
    num_train=100,
    num_val=20,
    num_test=30,
    train_path="/data/train",
    val_path="/data/val",
    test_path="/data/test",
    modality="CT",
)

# Генерация конфига для YOLO
generate_framework_config(
    framework=FrameworkType.YOLO,
    dataset_info=dataset_info,
    output_path="./configs/yolo_data.yaml",
)

# Или напрямую
from src.dcmmetatest import generate_yolo_config, generate_monai_config

generate_yolo_config(dataset_info, "./yolo.yaml")
generate_monai_config(dataset_info, "./monai.json")
generate_nnunet_config(dataset_info, "./nnunet/")
generate_huggingface_config(dataset_info, "./hf_dataset/")
```

---

## 📊 Обновленный план реализации

### ✅ Реализовано (текущая версия):

1. **Углубленная работа с данными**
   - ✅ Конвертация форматов: NIfTI, PNG/JPG/TIFF, Flatten to 2D
   - ✅ Нормализация интенсивности: windowing, Z-score, minmax, sigmoid
   - ✅ **Аугментация: повороты, флипы, эластичные деформации, шум** (NEW!)
   - ⚠️ Ресемплинг (частично)
   - ❌ Кроппинг (air cropping, ROI)

2. **Аналитика и валидация**
   - ✅ Статистический дашборд в Streamlit
   - ⚠️ Визуальный инспектор (базовый)
   - ⚠️ Поиск аномалий (частично)

3. **Управление метаданными**
   - ✅ Извлечение клинических данных из DICOM
   - ⚠️ Редактор словаря классов (без GUI)

4. **Интеграция и экспорт**
   - ✅ Экспорт в CSV, JSON, TXT
   - ✅ **Генераторы конфигов: YOLO, MONAI, nnU-Net, Hugging Face** (NEW!)
   - ❌ Версионирование датасетов

5. **UX и производительность**
   - ✅ Прогресс-бары (tqdm)
   - ❌ Кэширование (симлинки/hardlinks)

---

## 🔧 Следующие приоритеты

1. **Air cropping / ROI кроппинг** - автоматическое обрезание пустых областей
2. **Ресемплинг изображений** - изменение разрешения/spacing
3. **Визуальный DICOM-вьювер** в Streamlit
4. **Детектор аномалий** - проверка битых файлов, пустых масок, утечек
5. **Редактор словаря классов** (GUI)

---

## 📝 Зависимости

Для работы новых модулей требуются:

```bash
pip install scipy  # Для аугментаций
pip install pyyaml  # Для YOLO конфигов
pip install nibabel  # Для NIfTI экспорта
pip install Pillow  # Для работы с изображениями
```

Все зависимости уже указаны в `requirements.txt`.
