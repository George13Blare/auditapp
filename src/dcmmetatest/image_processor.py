"""
Модуль обработки медицинских изображений.

Функции для чтения, нормализации, конвертации и визуализации DICOM данных.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class WindowType(str, Enum):
    """Предустановленные типы окон для CT."""

    BRAIN = "brain"
    LUNG = "lung"
    BONE = "bone"
    SOFT_TISSUE = "soft_tissue"
    LIVER = "liver"
    CUSTOM = "custom"


@dataclass
class WindowConfig:
    """Конфигурация окна интенсивности."""

    window_type: WindowType = WindowType.SOFT_TISSUE
    window_center: float = 40.0
    window_width: float = 400.0

    def __post_init__(self):
        """Установка предустановленных значений."""
        if self.window_type != WindowType.CUSTOM:
            presets = {
                WindowType.BRAIN: (40, 80),
                WindowType.LUNG: (-600, 1500),
                WindowType.BONE: (400, 1800),
                WindowType.SOFT_TISSUE: (40, 400),
                WindowType.LIVER: (70, 150),
            }
            if self.window_type in presets:
                center, width = presets[self.window_type]
                self.window_center = center
                self.window_width = width


@dataclass
class ImageMetadata:
    """Метаданные изображения."""

    patient_id: str = ""
    study_id: str = ""
    series_id: str = ""
    modality: str = ""
    rows: int = 0
    columns: int = 0
    slices: int = 1
    pixel_spacing: tuple[float, float] = (1.0, 1.0)
    slice_thickness: float = 1.0
    image_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_orientation: tuple[float, ...] = ()
    rescale_intercept: float = 0.0
    rescale_slope: float = 1.0
    window_center: float | None = None
    window_width: float | None = None
    bits_allocated: int = 16
    photometric_interpretation: str = "MONOCHROME2"


@dataclass
class ProcessedImage:
    """Результат обработки изображения."""

    array: np.ndarray
    metadata: ImageMetadata
    mask: np.ndarray | None = None
    mask_classes: list[dict] | None = None


@dataclass
class AugmentationConfig:
    """Конфигурация аугментаций для 2D/3D массивов."""

    rotate_k: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    add_gaussian_noise: bool = False
    noise_std: float = 0.01
    random_seed: int | None = None


@dataclass
class PreprocessingPipelineConfig:
    """Конфигурация единого preprocessing pipeline."""

    normalization_method: str = "minmax"
    clip_percentile: tuple[float, float] | None = None
    enable_resampling: bool = False
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    crop_nonzero: bool = False
    crop_threshold: float = 0.0
    crop_margin: int = 0
    enable_augmentation: bool = False
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    export_format: str = "png"
    save_mask: bool = True


def apply_windowing(
    array: np.ndarray, window_center: float, window_width: float, output_dtype: type = np.float32
) -> npt.NDArray[np.float32]:
    """
    Применяет windowing к массиву интенсивностей.

    Args:
        array: Входной массив (HU или raw значения)
        window_center: Центр окна
        window_width: Ширина окна
        output_dtype: Тип данных на выходе

    Returns:
        Массив с примененным windowing в диапазоне [0, 1]
    """
    if array.dtype == np.uint16 or array.dtype == np.uint8:
        array = array.astype(np.float32)

    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2

    windowed = np.clip(array, lower_bound, upper_bound)
    windowed = (windowed - lower_bound) / window_width

    return cast(npt.NDArray[np.float32], windowed.astype(output_dtype))


def normalize_intensity(
    array: np.ndarray, method: str = "minmax", clip_percentile: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Нормализует интенсивность изображения.

    Args:
        array: Входной массив
        method: Метод нормализации ("minmax", "zscore", "sigmoid")
        clip_percentile: Процентили для обрезки выбросов (например, (1, 99))

    Returns:
        Нормализованный массив
    """
    if clip_percentile:
        low, high = np.percentile(array, clip_percentile)
        array = np.clip(array, low, high)

    if method == "minmax":
        min_val = array.min()
        max_val = array.max()
        if max_val - min_val < 1e-6:
            return np.zeros_like(array, dtype=np.float32)
        result = ((array - min_val) / (max_val - min_val)).astype(np.float32)
        return cast(npt.NDArray[np.float32], result)

    elif method == "zscore":
        mean = array.mean()
        std = array.std()
        if std < 1e-6:
            return np.zeros_like(array, dtype=np.float32)
        result = ((array - mean) / std).astype(np.float32)
        return cast(npt.NDArray[np.float32], result)

    elif method == "sigmoid":
        mean = array.mean()
        std = array.std()
        if std < 1e-6:
            return np.full_like(array, 0.5, dtype=np.float32)
        result = (1 / (1 + np.exp(-(array - mean) / std))).astype(np.float32)
        return cast(npt.NDArray[np.float32], result)

    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")


def apply_augmentations(
    array: np.ndarray,
    config: AugmentationConfig,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Применяет базовые аугментации к объему или изображению.

    Поддерживаемые операции:
    - поворот на 90° * k по плоскости (Y, X),
    - горизонтальный/вертикальный флип,
    - добавление гауссова шума к изображению (без изменения маски).

    Args:
        array: Входной массив (2D или 3D)
        config: Конфигурация аугментаций
        mask: Опциональная маска того же размера (2D/3D)

    Returns:
        Кортеж (аугментированное изображение, аугментированная маска)
    """
    if array.ndim not in (2, 3):
        raise ValueError(f"Ожидается 2D или 3D массив, получено {array.ndim}D")

    if mask is not None and mask.shape != array.shape:
        raise ValueError(f"Размер маски {mask.shape} не совпадает с размером массива {array.shape}")

    # Для 2D приводим к псевдо-3D [1, H, W], чтобы использовать единый код
    array_work = array[np.newaxis, ...] if array.ndim == 2 else array.copy()
    mask_work = (
        mask[np.newaxis, ...] if mask is not None and mask.ndim == 2 else (mask.copy() if mask is not None else None)
    )

    rotate_k = config.rotate_k % 4
    if rotate_k:
        array_work = np.rot90(array_work, k=rotate_k, axes=(-2, -1))
        if mask_work is not None:
            mask_work = np.rot90(mask_work, k=rotate_k, axes=(-2, -1))

    if config.flip_horizontal:
        array_work = np.flip(array_work, axis=-1)
        if mask_work is not None:
            mask_work = np.flip(mask_work, axis=-1)

    if config.flip_vertical:
        array_work = np.flip(array_work, axis=-2)
        if mask_work is not None:
            mask_work = np.flip(mask_work, axis=-2)

    if config.add_gaussian_noise:
        rng = np.random.default_rng(config.random_seed)
        noise = rng.normal(0.0, config.noise_std, size=array_work.shape).astype(np.float32)
        array_work = array_work.astype(np.float32) + noise

    if array.ndim == 2:
        array_out = array_work[0]
        mask_out = mask_work[0] if mask_work is not None else None
    else:
        array_out = array_work
        mask_out = mask_work

    return array_out, mask_out


def crop_to_nonzero(
    array: np.ndarray,
    threshold: float = 0.0,
    margin: int = 0,
) -> tuple[np.ndarray, tuple[slice, ...]]:
    """
    Обрезает массив до минимального bounding box ненулевой области.

    Args:
        array: 2D или 3D массив
        threshold: Порог для выделения значимых вокселей
        margin: Дополнительный отступ вокруг bbox (в пикселях/вокселях)

    Returns:
        Кортеж (cropped_array, bbox_slices)
    """
    if array.ndim not in (2, 3):
        raise ValueError(f"Ожидается 2D или 3D массив, получено {array.ndim}D")

    coords = np.argwhere(array > threshold)
    if coords.size == 0:
        # Нечего кроппить
        full_bbox = tuple(slice(0, dim) for dim in array.shape)
        return array.copy(), full_bbox

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, np.array(array.shape))

    bbox = tuple(slice(int(mins[i]), int(maxs[i])) for i in range(array.ndim))
    return array[bbox], bbox


def crop_to_roi(
    array: np.ndarray,
    roi_slices: tuple[slice, ...],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Кроп по заданному ROI bbox.

    Args:
        array: 2D/3D массив изображения
        roi_slices: bbox в формате tuple[slice, ...]
        mask: Опциональная маска того же размера

    Returns:
        Кортеж (cropped_image, cropped_mask)
    """
    if len(roi_slices) != array.ndim:
        raise ValueError(f"Размерность roi_slices ({len(roi_slices)}) не совпадает с массивом ({array.ndim})")

    if mask is not None and mask.shape != array.shape:
        raise ValueError(f"Размер маски {mask.shape} не совпадает с размером массива {array.shape}")

    cropped_image = array[roi_slices]
    cropped_mask = mask[roi_slices] if mask is not None else None
    return cropped_image, cropped_mask


def _interp_along_axis(
    array: np.ndarray,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Линейная интерполяция массива вдоль выбранной оси."""
    moved = np.moveaxis(array, axis, 0)
    result_shape = (len(target_coords),) + moved.shape[1:]
    result = np.empty(result_shape, dtype=np.float32)

    flat_in = moved.reshape(moved.shape[0], -1)
    flat_out = result.reshape(result.shape[0], -1)
    for idx in range(flat_in.shape[1]):
        flat_out[:, idx] = np.interp(target_coords, source_coords, flat_in[:, idx])

    return np.moveaxis(result, 0, axis)


def resample_volume(
    volume: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    is_mask: bool = False,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Ресемплинг 3D объема до целевого spacing.

    Args:
        volume: Входной объем [Z, Y, X]
        current_spacing: Текущий spacing (z, y, x) в мм
        target_spacing: Целевой spacing (z, y, x) в мм
        is_mask: Если True, используется nearest-neighbor (через округление)

    Returns:
        Кортеж (resampled_volume, output_spacing)
    """
    if volume.ndim != 3:
        raise ValueError(f"Ожидается 3D объем [Z, Y, X], получено {volume.ndim}D")

    if any(sp <= 0 for sp in current_spacing + target_spacing):
        raise ValueError("Spacing должен быть положительным")

    in_shape = np.array(volume.shape, dtype=np.int32)
    spacing_ratio = np.array(current_spacing, dtype=np.float64) / np.array(target_spacing, dtype=np.float64)
    out_shape = np.maximum(np.round(in_shape * spacing_ratio).astype(np.int32), 1)

    if np.array_equal(in_shape, out_shape):
        return volume.copy(), target_spacing

    # Координаты в индексах исходного массива
    src_coords = [np.arange(in_shape[i], dtype=np.float64) for i in range(3)]
    dst_coords = [np.linspace(0, in_shape[i] - 1, out_shape[i], dtype=np.float64) for i in range(3)]

    if is_mask:
        # NN-ресемплинг для дискретных масок
        z_idx = np.clip(np.round(dst_coords[0]).astype(np.int32), 0, in_shape[0] - 1)
        y_idx = np.clip(np.round(dst_coords[1]).astype(np.int32), 0, in_shape[1] - 1)
        x_idx = np.clip(np.round(dst_coords[2]).astype(np.int32), 0, in_shape[2] - 1)
        resampled = volume[np.ix_(z_idx, y_idx, x_idx)]
        return resampled, target_spacing

    # Линейная интерполяция по осям (separable interpolation)
    resampled = volume.astype(np.float32)
    resampled = _interp_along_axis(resampled, src_coords[0], dst_coords[0], axis=0)
    resampled = _interp_along_axis(resampled, src_coords[1], dst_coords[1], axis=1)
    resampled = _interp_along_axis(resampled, src_coords[2], dst_coords[2], axis=2)
    return resampled.astype(np.float32), target_spacing


def preprocess_volume_pipeline(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    config: PreprocessingPipelineConfig,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float, float]]:
    """
    Применяет единый preprocessing pipeline к уже загруженному объему.
    """
    processed = normalize_intensity(volume, method=config.normalization_method, clip_percentile=config.clip_percentile)
    processed_mask = mask.copy() if mask is not None else None
    out_spacing = spacing

    if config.enable_resampling:
        processed, out_spacing = resample_volume(processed, spacing, config.target_spacing, is_mask=False)
        if processed_mask is not None:
            processed_mask, _ = resample_volume(processed_mask, spacing, config.target_spacing, is_mask=True)

    if config.crop_nonzero:
        processed, roi = crop_to_nonzero(processed, threshold=config.crop_threshold, margin=config.crop_margin)
        if processed_mask is not None:
            processed_mask, _ = crop_to_roi(processed_mask, roi)

    if config.enable_augmentation:
        processed, processed_mask = apply_augmentations(processed, config.augmentation, processed_mask)

    return processed, processed_mask, out_spacing


def preprocess_dicom_series_pipeline(
    input_path: Path,
    output_path: Path,
    config: PreprocessingPipelineConfig,
) -> dict[str, Any]:
    """
    Сквозной pipeline: чтение DICOM серии -> preprocessing -> экспорт.
    """
    stats: dict[str, Any] = {"files_saved": 0, "errors": [], "format": config.export_format}
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        processed_image = read_dicom_series(input_path, apply_rescale=True, window_config=None)
        volume = processed_image.array
        metadata = processed_image.metadata

        spacing = (metadata.slice_thickness, metadata.pixel_spacing[0], metadata.pixel_spacing[1])
        processed_volume, processed_mask, out_spacing = preprocess_volume_pipeline(
            volume=volume,
            spacing=spacing,
            config=config,
            mask=processed_image.mask,
        )

        if config.export_format.lower() == "nifti":
            metadata.slice_thickness = out_spacing[0]
            metadata.pixel_spacing = (out_spacing[1], out_spacing[2])
            convert_to_nifti(processed_volume, metadata, output_path / "volume.nii.gz", mask=processed_mask)
            stats["files_saved"] = 1
        else:
            files = convert_to_image_slices(
                processed_volume,
                output_path / "images",
                mask=processed_mask if config.save_mask else None,
                image_format=config.export_format.lower(),
            )
            stats["files_saved"] = len(files)

    except Exception as exc:
        stats["errors"].append(str(exc))
        logger.exception("Ошибка preprocess_dicom_series_pipeline")

    return stats


def read_dicom_series(
    series_path: Path, apply_rescale: bool = True, window_config: WindowConfig | None = None
) -> ProcessedImage:
    """
    Читает серию DICOM файлов как 3D объем.

    Args:
        series_path: Путь к папке с DICOM файлами серии
        apply_rescale: Применить RescaleSlope/Intercept
        window_config: Конфигурация windowing

    Returns:
        ProcessedImage с 3D массивом и метаданными
    """
    try:
        import pydicom
    except ImportError as e:
        raise ImportError("Требуется pydicom: pip install pydicom") from e

    series_path = Path(series_path)
    if not series_path.exists():
        raise FileNotFoundError(f"Путь не найден: {series_path}")

    # Сбор DICOM файлов
    dcm_files = list(series_path.glob("*.dcm")) + list(series_path.glob("*.dicom"))

    # Файлы без расширения тоже могут быть DICOM
    for f in series_path.iterdir():
        if f.is_file() and f.suffix == "" and f not in dcm_files:
            try:
                ds = pydicom.dcmread(f, force=True)
                if hasattr(ds, "Modality"):
                    dcm_files.append(f)
            except Exception:
                continue

    if not dcm_files:
        raise ValueError(f"DICOM файлы не найдены в {series_path}")

    # Чтение и сортировка по позиции среза
    slices = []
    for dcm_file in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_file, force=True)
            slices.append(ds)
        except Exception as e:
            logger.warning(f"Ошибка чтения {dcm_file}: {e}")

    if not slices:
        raise ValueError("Не удалось прочитать ни одного DICOM файла")

    # Сортировка по позиции среза
    def get_slice_position(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
        elif hasattr(ds, "InstanceNumber"):
            return float(ds.InstanceNumber)
        return 0

    slices.sort(key=get_slice_position)

    # Извлечение метаданных из первого среза
    first_slice = slices[0]
    metadata = ImageMetadata(
        patient_id=getattr(first_slice, "PatientID", ""),
        study_id=getattr(first_slice, "StudyInstanceUID", ""),
        series_id=getattr(first_slice, "SeriesInstanceUID", ""),
        modality=getattr(first_slice, "Modality", ""),
        rows=getattr(first_slice, "Rows", 0),
        columns=getattr(first_slice, "Columns", 0),
        slices=len(slices),
        pixel_spacing=tuple(float(x) for x in getattr(first_slice, "PixelSpacing", [1.0, 1.0]))[:2],  # type: ignore
        slice_thickness=float(getattr(first_slice, "SliceThickness", 1.0)),
        image_position=tuple(float(x) for x in getattr(first_slice, "ImagePositionPatient", [0.0, 0.0, 0.0]))[:3],  # type: ignore
        image_orientation=tuple(map(float, getattr(first_slice, "ImageOrientationPatient", []))),
        rescale_intercept=float(getattr(first_slice, "RescaleIntercept", 0.0)),
        rescale_slope=float(getattr(first_slice, "RescaleSlope", 1.0)),
        window_center=getattr(first_slice, "WindowCenter", None),
        window_width=getattr(first_slice, "WindowWidth", None),
        bits_allocated=int(getattr(first_slice, "BitsAllocated", 16)),
        photometric_interpretation=getattr(first_slice, "PhotometricInterpretation", "MONOCHROME2"),
    )

    # Создание 3D массива
    arrays = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)

        # Применение Rescale
        if apply_rescale:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

        # Переворот для MONOCHROME1
        if metadata.photometric_interpretation == "MONOCHROME1":
            arr = np.max(arr) - arr

        arrays.append(arr)

    volume = np.stack(arrays, axis=0)

    # Применение windowing если задано
    if window_config:
        wc = window_config.window_center
        ww = window_config.window_width
        volume = apply_windowing(volume, wc, ww)

    return ProcessedImage(array=volume, metadata=metadata)


def read_dicom_seg(seg_path: Path, reference_volume: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """
    Читает DICOM SEG файл и извлекает маску с классами.

    Args:
        seg_path: Путь к DICOM SEG файлу
        reference_volume: Опциональный референсный объем для проверки размеров

    Returns:
        Кортеж (маска как 3D массив, список классов)
    """
    try:
        import pydicom
    except ImportError as e:
        raise ImportError("Требуется pydicom: pip install pydicom") from e

    ds = pydicom.dcmread(seg_path, force=True)

    if getattr(ds, "Modality", "") != "SEG":
        raise ValueError(f"Файл не является DICOM SEG: {seg_path}")

    # Извлечение информации о классах
    classes = []
    if hasattr(ds, "SegmentSequence"):
        for idx, segment in enumerate(ds.SegmentSequence):
            class_info = {
                "segment_number": idx + 1,
                "label": getattr(segment, "SegmentLabel", f"Class_{idx}"),
                "description": getattr(segment, "SegmentDescription", ""),
                "algorithm_type": getattr(segment, "SegmentAlgorithmType", ""),
            }

            # Попытка извлечь цвет
            if hasattr(segment, "RecommendedDisplayCIELabValue"):
                # Упрощенная конвертация CIELab -> RGB (заглушка)
                class_info["color"] = (255, 255, 255)

            classes.append(class_info)

    # Извлечение маски
    # Для простоты читаем PixelData и интерпретируем как бинарные маски
    # В реальности нужно парсить PerFrameFunctionalGroupsSequence

    if hasattr(ds, "pixel_array"):
        mask = ds.pixel_array
    else:
        raise ValueError("Не удалось извлечь данные пикселей из SEG файла")

    # Проверка размеров
    if reference_volume is not None:
        if mask.shape != reference_volume.shape:
            logger.warning(f"Размеры маски {mask.shape} не совпадают с референсным объемом {reference_volume.shape}")

    return mask, classes


def convert_to_image_slices(
    volume: np.ndarray,
    output_dir: Path,
    prefix: str = "slice",
    mask: np.ndarray | None = None,
    overlay_alpha: float = 0.5,
    image_format: str = "png",
) -> list[Path]:
    """
    Конвертирует 3D объем в набор 2D изображений (PNG/JPG/TIFF).

    Args:
        volume: 3D numpy массив (H, W, D) или (D, H, W)
        output_dir: Папка для сохранения
        prefix: Префикс имен файлов
        mask: Опциональная маска для наложения
        overlay_alpha: Прозрачность наложения маски
        image_format: Формат выходных файлов (png, jpg/jpeg, tiff)

    Returns:
        Список сохраненных файлов
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Требуется Pillow: pip install Pillow") from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Нормализация осей: ожидаем (D, H, W)
    if volume.ndim == 3:
        if volume.shape[0] < volume.shape[-1]:
            volume = np.transpose(volume, (2, 0, 1))
    else:
        raise ValueError(f"Ожидается 3D массив, получено {volume.ndim}D")

    saved_files = []
    normalized_format = image_format.lower()
    if normalized_format == "jpeg":
        normalized_format = "jpg"

    save_params = {
        "png": ("PNG", ".png"),
        "jpg": ("JPEG", ".jpg"),
        "tiff": ("TIFF", ".tiff"),
    }

    if normalized_format not in save_params:
        raise ValueError(f"Неподдерживаемый формат: {image_format}. Используйте png/jpg/tiff")

    pil_format, extension = save_params[normalized_format]

    for idx, slice_2d in enumerate(volume):
        # Нормализация к [0, 255]
        if slice_2d.dtype == np.float32 or slice_2d.dtype == np.float64:
            slice_normalized = np.clip(slice_2d * 255, 0, 255).astype(np.uint8)
        else:
            slice_normalized = slice_2d

        # Создание RGB изображения
        img = Image.fromarray(slice_normalized, mode="L").convert("RGB")

        # Наложение маски если есть
        if mask is not None:
            if mask.shape == volume.shape:
                mask_slice = mask[idx]
            elif mask.ndim == 2:
                mask_slice = mask
            else:
                mask_slice = None

            if mask_slice is not None:
                # Создание цветной маски
                mask_rgba = Image.fromarray((mask_slice > 0).astype(np.uint8) * 255, mode="L").convert("RGBA")
                alpha_channel = int(np.clip(overlay_alpha, 0.0, 1.0) * 255)
                mask_rgba.putalpha(alpha_channel)

                # Наложение с прозрачностью
                img = img.convert("RGBA")
                mask_rgba = mask_rgba.resize(img.size)
                img = Image.alpha_composite(img, mask_rgba).convert("RGB")

        # Сохранение
        output_path = output_dir / f"{prefix}_{idx:04d}{extension}"
        img.save(output_path, format=pil_format)
        saved_files.append(output_path)

    return saved_files


def convert_to_png(
    volume: np.ndarray,
    output_dir: Path,
    prefix: str = "slice",
    mask: np.ndarray | None = None,
    overlay_alpha: float = 0.5,
) -> list[Path]:
    """
    Backward-compatible обертка для экспорта 2D срезов в PNG.
    """
    return convert_to_image_slices(
        volume=volume,
        output_dir=output_dir,
        prefix=prefix,
        mask=mask,
        overlay_alpha=overlay_alpha,
        image_format="png",
    )


def convert_to_nifti(
    volume: np.ndarray, metadata: ImageMetadata, output_path: Path, mask: np.ndarray | None = None
) -> Path:
    """
    Конвертирует объем в формат NIfTI.

    Args:
        volume: 3D numpy массив
        metadata: Метаданные изображения
        output_path: Путь выходного файла (.nii или .nii.gz)
        mask: Опциональная маска сегментации

    Returns:
        Путь к сохраненному файлу
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("Требуется nibabel: pip install nibabel") from e

    output_path = Path(output_path)

    # Создание аффинной матрицы
    spacing = metadata.pixel_spacing
    slice_thickness = metadata.slice_thickness

    affine = np.array(
        [
            [spacing[0], 0, 0, metadata.image_position[0]],
            [0, spacing[1], 0, metadata.image_position[1]],
            [0, 0, slice_thickness, metadata.image_position[2]],
            [0, 0, 0, 1],
        ]
    )

    # Создание NIfTI образа
    nii_img = nib.Nifti1Image(volume.astype(np.float32), affine)

    # Добавление метаданных
    nii_img.header.set_xyzt_units("mm")

    # Сохранение
    if output_path.suffix not in [".nii", ".nii.gz"]:
        output_path = output_path.with_suffix(".nii.gz")

    nib.save(nii_img, output_path)

    # Сохранение маски если есть
    if mask is not None:
        mask_path = output_path.parent / f"{output_path.stem}_mask{output_path.suffix}"
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
        nib.save(mask_img, mask_path)
        logger.info(f"Маска сохранена: {mask_path}")

    return output_path


def flatten_to_2d(
    input_path: Path,
    output_path: Path,
    window_config: WindowConfig | None = None,
    save_mask: bool = True,
    format: str = "png",
) -> dict:
    """
    Преобразует 3D DICOM серию в набор 2D срезов.

    Args:
        input_path: Путь к папке с DICOM серией
        output_path: Путь к выходной папке
        window_config: Конфигурация windowing
        save_mask: Сохранять ли маски если найдены
        format: Формат выхода ("png", "jpg", "tiff", "nifti")

    Returns:
        Статистика обработки
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int | list] = {"slices_processed": 0, "masks_found": 0, "files_saved": 0, "errors": []}

    try:
        # Чтение объема
        processed = read_dicom_series(input_path, apply_rescale=True, window_config=window_config)

        volume = processed.array
        metadata = processed.metadata

        # Поиск маски в той же папке
        mask = None
        mask_classes = None

        if save_mask:
            seg_files = list(input_path.parent.glob("*.SEG")) + list(input_path.parent.glob("*SEG*.dcm"))

            for seg_file in seg_files[:1]:  # Берем первую найденную
                try:
                    mask, mask_classes = read_dicom_seg(seg_file, volume)
                    stats["masks_found"] += 1  # type: ignore
                    break
                except Exception as e:
                    stats["errors"].append(f"Ошибка чтения маски {seg_file}: {e}")  # type: ignore

        # Конвертация в 2D
        image_formats = {"png", "jpg", "jpeg", "tiff"}

        if format.lower() in image_formats:
            saved_files = convert_to_image_slices(
                volume,
                output_path / "images",
                prefix="slice",
                mask=mask,
                image_format=format.lower(),
            )
            stats["files_saved"] = len(saved_files)

            # Сохранение метаданных
            import json

            meta_file = output_path / "metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "patient_id": metadata.patient_id,
                        "study_id": metadata.study_id,
                        "series_id": metadata.series_id,
                        "modality": metadata.modality,
                        "pixel_spacing": metadata.pixel_spacing,
                        "slice_thickness": metadata.slice_thickness,
                        "total_slices": metadata.slices,
                        "mask_classes": mask_classes,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            stats["files_saved"] += 1  # type: ignore

        elif format.lower() == "nifti":
            convert_to_nifti(volume, metadata, output_path / "volume.nii.gz", mask=mask)
            stats["files_saved"] = 1  # type: ignore

        stats["slices_processed"] = metadata.slices  # type: ignore

    except Exception as e:
        stats["errors"].append(f"Критическая ошибка: {e}")  # type: ignore
        logger.error(f"Ошибка при flatten_to_2d: {e}")

    return stats


def create_overlay_image(image: np.ndarray, mask: np.ndarray, colormap: str = "jet", alpha: float = 0.5) -> np.ndarray:
    """
    Создает изображение с наложенной маской.

    Args:
        image: 2D изображение (H, W) или (H, W, 3)
        mask: 2D маска (H, W) с целочисленными значениями классов
        colormap: Название colormap matplotlib
        alpha: Прозрачность наложения

    Returns:
        RGB изображение с наложенной маской
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except ImportError as e:
        raise ImportError("Требуется matplotlib: pip install matplotlib") from e

    # Нормализация изображения
    if image.dtype == np.float32 or image.dtype == np.float64:
        img_display = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img_display = image

    if img_display.ndim == 2:
        img_rgb = np.stack([img_display] * 3, axis=-1)
    elif img_display.ndim == 3 and img_display.shape[-1] == 1:
        img_rgb = np.concatenate([img_display] * 3, axis=-1)
    else:
        img_rgb = img_display

    # Создание цветной маски
    unique_labels = np.unique(mask)
    num_labels = len(unique_labels)

    cmap = plt.get_cmap(colormap)
    norm = colors.Normalize(vmin=0, vmax=max(num_labels - 1, 1))

    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)

    for label in unique_labels:
        if label == 0:
            continue

        mask_indices = mask == label
        color = cmap(norm(label))[:3]  # RGB без alpha

        mask_rgba[mask_indices, :3] = color
        mask_rgba[mask_indices, 3] = alpha

    # Наложение
    result = img_rgb.astype(np.float32) / 255.0
    alpha_channel = mask_rgba[..., 3:4].astype(np.float32)
    rgb_mask = mask_rgba[..., :3].astype(np.float32)
    blended = result * (1 - alpha_channel) + rgb_mask * alpha_channel
    result = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return cast(npt.NDArray[np.uint8], result)
