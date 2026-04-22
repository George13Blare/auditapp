"""
Модуль генерации конфигурационных файлов для ML-фреймворков.

Поддерживаемые фреймворки:
- YOLO (сегментация/детекция)
- MONAI
- nnU-Net
- Hugging Face Datasets
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FrameworkType(str, Enum):
    """Типы поддерживаемых фреймворков."""

    YOLO = "yolo"
    MONAI = "monai"
    NNU_NET = "nnunet"
    HUGGINGFACE = "huggingface"


class TaskType(str, Enum):
    """Типы задач."""

    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    CLASSIFICATION = "classification"


@dataclass
class DatasetInfo:
    """Информация о датасете."""

    name: str
    description: str = ""
    task_type: TaskType = TaskType.SEGMENTATION
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    class_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    image_size: tuple[int, int] = (512, 512)
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    modality: str = "CT"  # CT, MR, PT, etc.


@dataclass
class YOLOConfig:
    """Конфигурация для YOLO."""

    # Пути
    train: str = ""
    val: str = ""
    test: Optional[str] = None

    # Классы
    nc: int = 0
    names: list[str] = field(default_factory=list)

    # Аугментации
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0

    # Обучение
    imgsz: int = 512
    batch: int = 16
    epochs: int = 100
    patience: int = 50
    lr0: float = 0.01
    lrf: float = 0.1
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    # Модель
    model: str = "yolov8n-seg.yaml"  # или yolov8n.pt для детекции

    # Сохранение
    project: str = "runs"
    name: str = "train"
    exist_ok: bool = False


@dataclass
class MONAIConfig:
    """Конфигурация для MONAI."""

    # Данные
    data_dir: str = ""
    train_list: str = ""
    val_list: str = ""
    test_list: str = ""

    # Задача
    task: str = "segmentation"  # segmentation, classification, detection

    # Параметры данных
    spatial_dims: int = 3
    in_channels: int = 1
    out_channels: int = 1
    num_classes: int = 0

    # Препроцессинг
    intensity_range: tuple[float, float] = (-1000.0, 3000.0)
    clip_outliers: bool = True
    normalize: bool = True

    # Трансформации обучения
    train_transforms: list[dict] = field(default_factory=list)
    val_transforms: list[dict] = field(default_factory=list)

    # Архитектура
    network: str = "UNet"
    network_args: dict = field(
        default_factory=lambda: {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [16, 32, 64, 128, 256],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
        }
    )

    # Обучение
    learning_rate: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 2
    num_workers: int = 4

    # Loss
    loss_function: str = "DiceLoss"
    loss_args: dict = field(default_factory=lambda: {"include_background": True, "sigmoid": True})

    # Метрики
    metrics: list[str] = field(default_factory=lambda: ["dice", "hausdorff"])

    # Сохранение
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"


@dataclass
class NNUNetConfig:
    """Конфигурация для nnU-Net."""

    # Пути
    raw_data_dir: str = ""
    preprocessed_data_dir: str = ""
    results_dir: str = ""

    # Задача
    task_id: str = "Task001"
    task_name: str = "MyDataset"
    task_type: str = "segmentation"  # segmentation, classification

    # Модальности
    modalities: dict[int, str] = field(default_factory=lambda: {0: "CT"})

    # Регион интереса
    foreground_intensity_properties_per_modality: dict = field(default_factory=dict)

    # Планы
    plans_identifier: str = "nnUNetPlans"
    use_compressed_data: bool = False

    # Обучение
    trainer: str = "nnUNetTrainer"
    configurations: list[str] = field(default_factory=lambda: ["2d", "3d_fullres", "3d_lowres"])

    # Кросс-валидация
    num_folds: int = 5
    validation_metric: str = "dice"

    # Постпроцессинг
    determine_postprocessing: bool = True


@dataclass
class HuggingFaceConfig:
    """Конфигурация для Hugging Face Datasets."""

    dataset_name: str = "medical-dataset"
    config_name: str = "default"
    version: str = "1.0.0"
    description: str = ""
    license: str = "MIT"

    # Авторы
    authors: list[str] = field(default_factory=list)

    # Ссылки
    homepage: str = ""
    repository: str = ""

    # Фичи
    features: dict = field(
        default_factory=lambda: {
            "image": "Image",
            "label": "int32",
            "metadata": "dict",
        }
    )

    # Сплиты
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)

    # Публикация
    push_to_hub: bool = False
    hub_organization: str = ""


def generate_yolo_config(dataset_info: DatasetInfo, output_path: Path, config: Optional[YOLOConfig] = None) -> Path:
    """
    Генерирует YAML конфиг для YOLOv8.

    Args:
        dataset_info: Информация о датасете
        output_path: Путь для сохранения конфига
        config: Дополнительные параметры конфига

    Returns:
        Путь к сохраненному файлу
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("Требуется PyYAML: pip install pyyaml")

    if config is None:
        config = YOLOConfig()

    # Заполнение базовых параметров
    config.train = dataset_info.train_path
    config.val = dataset_info.val_path
    if dataset_info.test_path:
        config.test = dataset_info.test_path

    config.nc = dataset_info.num_classes
    config.names = dataset_info.class_names

    # Конвертация в словарь
    config_dict = asdict(config)

    # Удаление None значений
    config_dict = {k: v for k, v in config_dict.items() if v is not None}

    # Сохранение
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".yaml")

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"YOLO конфиг сохранен: {output_path}")
    return output_path


def generate_monai_config(dataset_info: DatasetInfo, output_path: Path, config: Optional[MONAIConfig] = None) -> Path:
    """
    Генерирует JSON конфиг для MONAI.

    Args:
        dataset_info: Информация о датасете
        output_path: Путь для сохранения конфига
        config: Дополнительные параметры конфига

    Returns:
        Путь к сохраненному файлу
    """
    if config is None:
        config = MONAIConfig()

    # Заполнение базовых параметров
    config.data_dir = dataset_info.train_path.rsplit("/", 1)[0] if dataset_info.train_path else ""
    config.num_classes = dataset_info.num_classes

    if dataset_info.modality == "MR":
        config.intensity_range = (0.0, 1.0)
    elif dataset_info.modality == "CT":
        config.intensity_range = (-1000.0, 3000.0)

    # Настройка out_channels
    if dataset_info.task_type == TaskType.SEGMENTATION:
        config.out_channels = dataset_info.num_classes
        config.network_args["out_channels"] = dataset_info.num_classes
        config.loss_function = "DiceCELoss"
        config.loss_args = {"include_background": True, "softmax": True}
    elif dataset_info.task_type == TaskType.CLASSIFICATION:
        config.out_channels = dataset_info.num_classes
        config.network = "DenseNet121"
        config.network_args = {
            "spatial_dims": config.spatial_dims,
            "in_channels": config.in_channels,
            "num_classes": dataset_info.num_classes,
        }

    # Трансформации по умолчанию
    if not config.train_transforms:
        config.train_transforms = [
            {"_target_": "LoadImaged", "keys": ["image", "label"]},
            {"_target_": "EnsureChannelFirstd", "keys": ["image", "label"]},
            {"_target_": "Orientationd", "keys": ["image", "label"], "axcodes": "RAS"},
            {"_target_": "Spacingd", "keys": ["image", "label"], "pixdim": [1.0, 1.0, 1.0]},
            {"_target_": "ScaleIntensityRanged", "keys": ["image"], "a_min": config.intensity_range[0], "a_max": config.intensity_range[1], "b_min": 0.0, "b_max": 1.0, "clip": config.clip_outliers},
            {"_target_": "RandCropByPosNegd", "keys": ["image", "label"], "label_key": "label", "spatial_size": [96, 96, 96], "pos": 1, "neg": 1, "num_samples": 4},
            {"_target_": "RandFlipd", "keys": ["image", "label"], "prob": 0.5, "spatial_axis": 0},
            {"_target_": "RandFlipd", "keys": ["image", "label"], "prob": 0.5, "spatial_axis": 1},
            {"_target_": "RandRotate90d", "keys": ["image", "label"], "prob": 0.5, "max_k": 3},
            {"_target_": "ToTensord", "keys": ["image", "label"]},
        ]

    if not config.val_transforms:
        config.val_transforms = [
            {"_target_": "LoadImaged", "keys": ["image", "label"]},
            {"_target_": "EnsureChannelFirstd", "keys": ["image", "label"]},
            {"_target_": "Orientationd", "keys": ["image", "label"], "axcodes": "RAS"},
            {"_target_": "Spacingd", "keys": ["image", "label"], "pixdim": [1.0, 1.0, 1.0]},
            {"_target_": "ScaleIntensityRanged", "keys": ["image"], "a_min": config.intensity_range[0], "a_max": config.intensity_range[1], "b_min": 0.0, "b_max": 1.0, "clip": config.clip_outliers},
            {"_target_": "ToTensord", "keys": ["image", "label"]},
        ]

    # Сохранение
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".json")

    config_dict = asdict(config)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"MONAI конфиг сохранен: {output_path}")
    return output_path


def generate_nnunet_config(dataset_info: DatasetInfo, output_path: Path, config: Optional[NNUNetConfig] = None) -> Path:
    """
    Генерирует JSON датасет для nnU-Net.

    Args:
        dataset_info: Информация о датасете
        output_path: Путь для сохранения конфига
        config: Дополнительные параметры конфига

    Returns:
        Путь к сохраненному файлу
    """
    if config is None:
        config = NNUNetConfig()

    # Заполнение базовых параметров
    config.task_name = dataset_info.name
    config.raw_data_dir = str(Path(dataset_info.train_path).parent)

    # Модальности
    if dataset_info.modality == "CT":
        config.modalities = {0: "CT"}
    elif dataset_info.modality == "MR":
        config.modalities = {0: "T1", 1: "T2", 2: "FLAIR"}  # Пример для мульти-модал MR
    else:
        config.modalities = {0: dataset_info.modality}

    # Создание структуры dataset.json для nnU-Net
    dataset_json = {
        "name": config.task_name,
        "description": dataset_info.description,
        "reference": "",
        "licence": "MIT",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": config.modalities,
        "labels": {"background": 0},
        "numTraining": dataset_info.num_train,
        "numTest": dataset_info.num_test,
        "training": [],
        "test": [],
        "regions": [],
    }

    # Добавление классов
    for idx, class_name in enumerate(dataset_info.class_names, start=1):
        dataset_json["labels"][class_name] = idx

    # Добавление информации об обучении
    train_dir = Path(dataset_info.train_path)
    images_dir = train_dir.parent / "imagesTr" if train_dir.exists() else train_dir
    labels_dir = train_dir.parent / "labelsTr" if train_dir.exists() else train_dir

    if images_dir.exists():
        for img_file in images_dir.glob("*.nii.gz"):
            case_id = img_file.stem.replace("_0000", "").replace("_0001", "")
            dataset_json["training"].append({
                "image": f"./imagesTr/{img_file.name}",
                "label": f"./labelsTr/{case_id}.nii.gz",
            })

    # Сохранение
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path / "dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)

    # Сохранение конфига nnU-Net отдельно
    config_path = output_path.parent / "nnunet_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    logger.info(f"nnU-Net датасет сохранен: {output_path}")
    logger.info(f"nnU-Net конфиг сохранен: {config_path}")
    return output_path


def generate_huggingface_config(dataset_info: DatasetInfo, output_path: Path, config: Optional[HuggingFaceConfig] = None) -> Path:
    """
    Генерирует конфиг для Hugging Face Datasets.

    Args:
        dataset_info: Информация о датасете
        output_path: Путь для сохранения конфига
        config: Дополнительные параметры конфига

    Returns:
        Путь к сохраненному файлу
    """
    if config is None:
        config = HuggingFaceConfig()

    # Заполнение базовых параметров
    config.dataset_name = dataset_info.name.lower().replace(" ", "_")
    config.description = dataset_info.description

    # Определение фичей в зависимости от задачи
    if dataset_info.task_type == TaskType.SEGMENTATION:
        config.features = {
            "image": {"path": "string", "bytes": None},
            "label": {"path": "string", "bytes": None},
            "patient_id": "string",
            "study_id": "string",
            "series_id": "string",
            "modality": "string",
        }
    elif dataset_info.task_type == TaskType.CLASSIFICATION:
        config.features = {
            "image": {"path": "string", "bytes": None},
            "label": "int32",
            "label_name": "string",
            "patient_id": "string",
        }

    # Сбор файлов
    if dataset_info.train_path:
        train_path = Path(dataset_info.train_path)
        if train_path.exists():
            config.train_files = [str(f) for f in train_path.glob("*")]

    if dataset_info.val_path:
        val_path = Path(dataset_info.val_path)
        if val_path.exists():
            config.val_files = [str(f) for f in val_path.glob("*")]

    if dataset_info.test_path:
        test_path = Path(dataset_info.test_path)
        if test_path.exists():
            config.test_files = [str(f) for f in test_path.glob("*")]

    # Создание script для загрузки датасета
    script_content = _generate_hf_dataset_script(config, dataset_info)

    # Сохранение
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Сохранение конфига
    config_path = output_path / "dataset_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    # Сохранение скрипта загрузки
    script_path = output_path / f"{config.config_name}.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    # README
    readme_content = _generate_hf_readme(config, dataset_info)
    readme_path = output_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"Hugging Face датасет сохранен: {output_path}")
    return output_path


def _generate_hf_dataset_script(config: HuggingFaceConfig, dataset_info: DatasetInfo) -> str:
    """Генерирует Python скрипт для загрузки датасета."""

    script = f'''\"\"\"
Датасет: {config.dataset_name}
Описание: {config.description}
Версия: {config.version}
\"\"\"

import os
from pathlib import Path

import datasets
from datasets import Features, Value, Image, Sequence


_CITATION = \"\"\"\\
@misc{{{config.dataset_name.replace("-", "_")},
  title="{{{dataset_info.name}}}",
  author={{{", ".join(config.authors)}}},
  year={{2024}},
}}
\"\"\"

_DESCRIPTION = \"\"\"\\
{config.description}
\"\"\"

_HOMEPAGE = "{config.homepage}"

_LICENSE = "{config.license}"

_URLS = {{
    "train": "{dataset_info.train_path}",
    "val": "{dataset_info.val_path}",
    "test": "{dataset_info.test_path}",
}}


class {config.dataset_name.replace("-", "_").title()}(datasets.GeneratorBasedBuilder):
    \"\"\"{config.dataset_name} датасет.\"\"\"

    VERSION = datasets.Version("{config.version}")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="{config.config_name}",
            version=VERSION,
            description="{config.description}",
        ),
    ]

    def _info(self):
        features = Features(
            {{
                "image": Image(),
                "label": Value("int32"),
                "label_name": Value("string"),
                "patient_id": Value("string"),
                "study_id": Value("string"),
                "modality": Value("string"),
            }}
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={{"filepath": downloaded_files["train"]}},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={{"filepath": downloaded_files["val"]}},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={{"filepath": downloaded_files["test"]}},
            ),
        ]

    def _generate_examples(self, filepath):
        \"\"\"Генерация примеров из директории.\"\"\"
        path = Path(filepath)
        
        # Маппинг имен классов в ID
        class_to_id = {{name: idx for idx, name in enumerate({dataset_info.class_names})}}
        
        for idx, img_file in enumerate(path.glob("*.png")):
            # Извлечение метаданных из имени файла
            parts = img_file.stem.split("_")
            patient_id = parts[0] if len(parts) > 0 else ""
            study_id = parts[1] if len(parts) > 1 else ""
            
            # Поиск соответствующей маски
            mask_file = path / f"{{img_file.stem}}_mask.png"
            label_id = 0
            label_name = "background"
            
            if mask_file.exists():
                # Здесь можно добавить логику определения класса из маски
                pass
            
            yield idx, {{
                "image": str(img_file),
                "label": label_id,
                "label_name": label_name,
                "patient_id": patient_id,
                "study_id": study_id,
                "modality": "{dataset_info.modality}",
            }}
'''

    return script


def _generate_hf_readme(config: HuggingFaceConfig, dataset_info: DatasetInfo) -> str:
    """Генерирует README для Hugging Face."""

    readme = f'''---
license: {config.license}
tags:
  - medical
  - imaging
  - {dataset_info.modality.lower()}
  - {dataset_info.task_type.value}
task_categories:
  - {dataset_info.task_type.value}
language:
  - en
pretty_name: {dataset_info.name}
---

# {dataset_info.name}

## Описание

{config.description}

## Характеристики

- **Модальность**: {dataset_info.modality}
- **Задача**: {dataset_info.task_type.value}
- **Количество классов**: {dataset_info.num_classes}
- **Классы**: {", ".join(dataset_info.class_names)}

## Статистика

| Сплит | Количество образцов |
|-------|---------------------|
| Train | {dataset_info.num_train} |
| Val   | {dataset_info.num_val} |
| Test  | {dataset_info.num_test} |

## Использование

```python
from datasets import load_dataset

dataset = load_dataset("{config.authors[0] if config.authors else 'username'}/{config.dataset_name}")
```

## Лицензия

{config.license}

## Контакты

{", ".join(config.authors)}
'''

    return readme


def generate_framework_config(
    framework: FrameworkType,
    dataset_info: DatasetInfo,
    output_path: Path,
    custom_config: Optional[Any] = None,
) -> Path:
    """
    Универсальная функция генерации конфига.

    Args:
        framework: Тип фреймворка
        dataset_info: Информация о датасете
        output_path: Путь для сохранения
        custom_config: Пользовательская конфигурация

    Returns:
        Путь к сохраненному файлу
    """
    if framework == FrameworkType.YOLO:
        return generate_yolo_config(dataset_info, output_path, custom_config)
    elif framework == FrameworkType.MONAI:
        return generate_monai_config(dataset_info, output_path, custom_config)
    elif framework == FrameworkType.NNU_NET:
        return generate_nnunet_config(dataset_info, output_path, custom_config)
    elif framework == FrameworkType.HUGGINGFACE:
        return generate_huggingface_config(dataset_info, output_path, custom_config)
    else:
        raise ValueError(f"Неподдерживаемый фреймворк: {framework}")


__all__ = [
    "FrameworkType",
    "TaskType",
    "DatasetInfo",
    "YOLOConfig",
    "MONAIConfig",
    "NNUNetConfig",
    "HuggingFaceConfig",
    "generate_yolo_config",
    "generate_monai_config",
    "generate_nnunet_config",
    "generate_huggingface_config",
    "generate_framework_config",
]
