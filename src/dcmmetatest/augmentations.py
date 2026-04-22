"""
Модуль аугментации медицинских изображений.

Поддерживает:
- Повороты (rotation)
- Отражения (flips)
- Эластичные деформации (elastic deformation)
- Шум (Gaussian, salt-pepper)
- Масштабирование (zoom)
- Сдвиги (shift)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AugmentationType(str, Enum):
    """Типы аугментаций."""

    ROTATION = "rotation"
    FLIP = "flip"
    ELASTIC = "elastic"
    GAUSSIAN_NOISE = "gaussian_noise"
    SALT_PEPPER_NOISE = "salt_pepper_noise"
    ZOOM = "zoom"
    SHIFT = "shift"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"


@dataclass
class AugmentationConfig:
    """Конфигурация аугментаций."""

    # Повороты
    rotation_range: tuple[float, float] = (-15.0, 15.0)  # Диапазон углов в градусах
    rotation_prob: float = 0.5

    # Отражения
    flip_horizontal_prob: float = 0.5
    flip_vertical_prob: float = 0.3
    flip_axial_prob: float = 0.3  # Для 3D

    # Эластичные деформации
    elastic_alpha: float = 30.0  # Сила деформации
    elastic_sigma: float = 5.0  # Гладкость деформации
    elastic_prob: float = 0.3

    # Шум
    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 0.1
    gaussian_noise_prob: float = 0.2

    salt_pepper_amount: float = 0.01  # Доля зашумленных пикселей
    salt_pepper_prob: float = 0.1

    # Масштабирование
    zoom_range: tuple[float, float] = (0.9, 1.1)
    zoom_prob: float = 0.3

    # Сдвиги
    shift_range: tuple[float, float] = (-0.1, 0.1)  # Доля от размера
    shift_prob: float = 0.3

    # Яркость/контраст
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    intensity_prob: float = 0.2

    # Общие настройки
    random_seed: Optional[int] = None
    interpolation_order: int = 1  # 0=nearest, 1=bilinear, 2=bicubic


@dataclass
class AugmentationStats:
    """Статистика примененных аугментаций."""

    total_images: int = 0
    augmentations_applied: dict[str, int] = field(default_factory=dict)
    failed_augmentations: int = 0


def apply_rotation(
    volume: np.ndarray,
    angle_range: tuple[float, float],
    axes: tuple[int, int] = (0, 1),
    order: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Применяет случайный поворот к объему.

    Args:
        volume: 3D или 2D массив
        angle_range: Диапазон углов (min, max) в градусах
        axes: Оси для вращения
        order: Порядок интерполяции
        rng: Генератор случайных чисел

    Returns:
        Повернутый массив
    """
    from scipy import ndimage

    if rng is None:
        rng = np.random.default_rng()

    angle = rng.uniform(angle_range[0], angle_range[1])

    if volume.ndim == 2:
        rotated = ndimage.rotate(volume, angle, axes=(0, 1), order=order, reshape=False)
    elif volume.ndim == 3:
        rotated = ndimage.rotate(volume, angle, axes=axes, order=order, reshape=False)
    else:
        raise ValueError(f"Ожидается 2D или 3D массив, получено {volume.ndim}D")

    return rotated.astype(volume.dtype)  # type: ignore[no-any-return]


def apply_flip(
    volume: np.ndarray,
    axis: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Применяет отражение по указанной оси.

    Args:
        volume: Входной массив
        axis: Ось для отражения (0=x, 1=y, 2=z)
        rng: Генератор случайных чисел

    Returns:
        Отраженный массив
    """
    if axis >= volume.ndim:
        raise ValueError(f"Ось {axis} вне диапазона для массива размерности {volume.ndim}")

    return np.flip(volume, axis=axis)


def apply_elastic_deformation(
    volume: np.ndarray,
    alpha: float = 30.0,
    sigma: float = 5.0,
    order: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Применяет эластичную деформацию.

    Args:
        volume: Входной массив
        alpha: Сила деформации
        sigma: Гладкость поля деформации
        order: Порядок интерполяции
        rng: Генератор случайных чисел

    Returns:
        Деформированный массив
    """
    from scipy import ndimage

    if rng is None:
        rng = np.random.default_rng()

    shape = volume.shape

    # Генерация случайного поля смещений
    dx = ndimage.gaussian_filter(rng.normal(size=shape), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter(rng.normal(size=shape), sigma, mode="constant", cval=0) * alpha

    if volume.ndim == 3:
        dz = ndimage.gaussian_filter(rng.normal(size=shape), sigma, mode="constant", cval=0) * alpha
        coordinates = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        indices: tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray] = (
            coordinates[0] + dx,
            coordinates[1] + dy,
            coordinates[2] + dz,
        )
    else:  # 2D
        coordinates = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            indexing="ij",
        )
        indices_2d: tuple[np.ndarray, np.ndarray] = (
            coordinates[0] + dx,
            coordinates[1] + dy,
        )
        indices = indices_2d  # type: ignore[assignment]

    # Применение деформации
    deformed = ndimage.map_coordinates(volume, indices, order=order, mode="nearest")

    return deformed.astype(volume.dtype)  # type: ignore[no-any-return]


def apply_gaussian_noise(
    volume: np.ndarray,
    mean: float = 0.0,
    std: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Добавляет гауссовский шум.

    Args:
        volume: Входной массив
        mean: Среднее значение шума
        std: Стандартное отклонение шума
        rng: Генератор случайных чисел

    Returns:
        Зашумленный массив
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(mean, std, size=volume.shape)
    noisy = volume + noise

    return noisy.astype(volume.dtype)


def apply_salt_pepper_noise(
    volume: np.ndarray,
    amount: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Добавляет шум "соль и перец".

    Args:
        volume: Входной массив (ожидается в диапазоне [0, 1] или [0, 255])
        amount: Доля зашумленных пикселей
        rng: Генератор случайных чисел

    Returns:
        Зашумленный массив
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = volume.copy()
    total_pixels = volume.size
    num_noisy = int(total_pixels * amount)

    # Координаты зашумленных пикселей
    coords = [rng.choice(dim_size, size=num_noisy) for dim_size in volume.shape]

    # Черные и белые пиксели
    salt_coords = rng.choice([0, 1], size=num_noisy)
    salt_indices = tuple(coords[i][salt_coords == 1] for i in range(volume.ndim))
    pepper_indices = tuple(coords[i][salt_coords == 0] for i in range(volume.ndim))

    # Определение диапазона значений
    if volume.dtype == np.float32 or volume.dtype == np.float64:
        max_val = 1.0
    else:
        max_val = np.iinfo(volume.dtype).max

    noisy[salt_indices] = max_val
    noisy[pepper_indices] = 0

    return noisy


def apply_zoom(
    volume: np.ndarray,
    zoom_range: tuple[float, float],
    order: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Применяет случайное масштабирование.

    Args:
        volume: Входной массив
        zoom_range: Диапазон масштаба (min, max)
        order: Порядок интерполяции
        rng: Генератор случайных чисел

    Returns:
        Масштабированный массив
    """
    from scipy import ndimage

    if rng is None:
        rng = np.random.default_rng()

    zoom_factor = rng.uniform(zoom_range[0], zoom_range[1])

    if volume.ndim == 2:
        zoomed = ndimage.zoom(volume, zoom_factor, order=order)
    elif volume.ndim == 3:
        zoomed = ndimage.zoom(volume, zoom_factor, order=order)
    else:
        raise ValueError(f"Ожидается 2D или 3D массив, получено {volume.ndim}D")

    # Обрезка/дополнение до исходного размера
    if zoomed.shape != volume.shape:
        cropped = np.zeros_like(volume)
        slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(zoomed.shape, volume.shape))
        cropped_slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(volume.shape, zoomed.shape))
        cropped[slices] = zoomed[cropped_slices]
        return cropped

    return zoomed.astype(volume.dtype)  # type: ignore[no-any-return]


def apply_shift(
    volume: np.ndarray,
    shift_range: tuple[float, float],
    order: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Применяет случайный сдвиг.

    Args:
        volume: Входной массив
        shift_range: Диапазон сдвига как доля от размера
        order: Порядок интерполяции
        rng: Генератор случайных чисел

    Returns:
        Сдвинутый массив
    """
    from scipy import ndimage

    if rng is None:
        rng = np.random.default_rng()

    shift_factors = [rng.uniform(shift_range[0], shift_range[1]) for _ in range(volume.ndim)]
    shift_pixels = [int(factor * dim) for factor, dim in zip(shift_factors, volume.shape)]

    shifted = ndimage.shift(volume, shift=shift_pixels, order=order, mode="nearest")

    return shifted.astype(volume.dtype)  # type: ignore[no-any-return]


def apply_brightness_contrast(
    volume: np.ndarray,
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Применяет случайную яркость и контраст.

    Args:
        volume: Входной массив
        brightness_range: Диапазон множителя яркости
        contrast_range: Диапазон множителя контраста
        rng: Генератор случайных чисел

    Returns:
        Модифицированный массив
    """
    if rng is None:
        rng = np.random.default_rng()

    result = volume.astype(np.float32)

    # Яркость
    brightness_factor = rng.uniform(brightness_range[0], brightness_range[1])
    result = result * brightness_factor

    # Контраст
    contrast_factor = rng.uniform(contrast_range[0], contrast_range[1])
    mean_val = result.mean()
    result = mean_val + (result - mean_val) * contrast_factor

    return result.astype(volume.dtype)  # type: ignore[no-any-return]


def apply_augmentation(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: Optional[AugmentationConfig] = None,
    augmentation_types: Optional[list[AugmentationType]] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Применяет набор аугментаций к изображению и маске.

    Args:
        volume: Входное изображение
        mask: Опциональная маска сегментации
        config: Конфигурация аугментаций
        augmentation_types: Список типов аугментаций для применения
        seed: Random seed

    Returns:
        Кортеж (аугментированное изображение, аугментированная маска)
    """
    if config is None:
        config = AugmentationConfig()

    if seed is not None:
        rng = np.random.default_rng(seed)
    elif config.random_seed is not None:
        rng = np.random.default_rng(config.random_seed)
    else:
        rng = np.random.default_rng()

    augmented_volume = volume.copy()
    augmented_mask = mask.copy() if mask is not None else None

    # Определение порядка применения аугментаций
    if augmentation_types is None:
        augmentation_types = list(AugmentationType)

    for aug_type in augmentation_types:
        try:
            if aug_type == AugmentationType.ROTATION and rng.random() < config.rotation_prob:
                augmented_volume = apply_rotation(
                    augmented_volume,
                    config.rotation_range,
                    order=config.interpolation_order,
                    rng=rng,
                )
                if augmented_mask is not None:
                    augmented_mask = apply_rotation(
                        augmented_mask,
                        config.rotation_range,
                        axes=(0, 1),
                        order=0,  # Nearest для масок
                        rng=rng,
                    )

            elif aug_type == AugmentationType.FLIP:
                # Горизонтальный флип
                if rng.random() < config.flip_horizontal_prob:
                    augmented_volume = apply_flip(augmented_volume, axis=1, rng=rng)
                    if augmented_mask is not None:
                        augmented_mask = apply_flip(augmented_mask, axis=1, rng=rng)

                # Вертикальный флип
                if rng.random() < config.flip_vertical_prob:
                    augmented_volume = apply_flip(augmented_volume, axis=0, rng=rng)
                    if augmented_mask is not None:
                        augmented_mask = apply_flip(augmented_mask, axis=0, rng=rng)

                # Аксиальный флип (для 3D)
                if augmented_volume.ndim == 3 and rng.random() < config.flip_axial_prob:
                    augmented_volume = apply_flip(augmented_volume, axis=2, rng=rng)
                    if augmented_mask is not None:
                        augmented_mask = apply_flip(augmented_mask, axis=2, rng=rng)

            elif aug_type == AugmentationType.ELASTIC and rng.random() < config.elastic_prob:
                augmented_volume = apply_elastic_deformation(
                    augmented_volume,
                    alpha=config.elastic_alpha,
                    sigma=config.elastic_sigma,
                    order=config.interpolation_order,
                    rng=rng,
                )
                if augmented_mask is not None:
                    augmented_mask = apply_elastic_deformation(
                        augmented_mask,
                        alpha=config.elastic_alpha,
                        sigma=config.elastic_sigma,
                        order=0,  # Nearest для масок
                        rng=rng,
                    )

            elif aug_type == AugmentationType.GAUSSIAN_NOISE and rng.random() < config.gaussian_noise_prob:
                augmented_volume = apply_gaussian_noise(
                    augmented_volume,
                    mean=config.gaussian_noise_mean,
                    std=config.gaussian_noise_std,
                    rng=rng,
                )

            elif aug_type == AugmentationType.SALT_PEPPER_NOISE and rng.random() < config.salt_pepper_prob:
                augmented_volume = apply_salt_pepper_noise(
                    augmented_volume,
                    amount=config.salt_pepper_amount,
                    rng=rng,
                )

            elif aug_type == AugmentationType.ZOOM and rng.random() < config.zoom_prob:
                augmented_volume = apply_zoom(
                    augmented_volume,
                    config.zoom_range,
                    order=config.interpolation_order,
                    rng=rng,
                )
                if augmented_mask is not None:
                    augmented_mask = apply_zoom(
                        augmented_mask,
                        config.zoom_range,
                        order=0,
                        rng=rng,
                    )

            elif aug_type == AugmentationType.SHIFT and rng.random() < config.shift_prob:
                augmented_volume = apply_shift(
                    augmented_volume,
                    config.shift_range,
                    order=config.interpolation_order,
                    rng=rng,
                )
                if augmented_mask is not None:
                    augmented_mask = apply_shift(
                        augmented_mask,
                        config.shift_range,
                        order=0,
                        rng=rng,
                    )

            elif aug_type == AugmentationType.BRIGHTNESS and rng.random() < config.intensity_prob:
                augmented_volume = apply_brightness_contrast(
                    augmented_volume,
                    config.brightness_range,
                    config.contrast_range,
                    rng=rng,
                )

        except Exception as e:
            logger.warning(f"Ошибка при применении аугментации {aug_type}: {e}")

    return augmented_volume, augmented_mask


def generate_augmented_dataset(
    input_dir: Path,
    output_dir: Path,
    config: Optional[AugmentationConfig] = None,
    num_augmentations: int = 5,
    save_masks: bool = True,
) -> AugmentationStats:
    """
    Генерирует аугментированный датасет из входных данных.

    Args:
        input_dir: Папка с исходными данными
        output_dir: Папка для аугментированных данных
        config: Конфигурация аугментаций
        num_augmentations: Количество аугментаций на изображение
        save_masks: Сохранять ли маски

    Returns:
        Статистика аугментации
    """
    from PIL import Image

    stats = AugmentationStats()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = AugmentationConfig()

    # Поиск изображений
    image_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]
    image_files: list[Path] = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"**/*{ext}"))

    # Поиск масок
    mask_files = {}
    if save_masks:
        for img_file in image_files:
            mask_name = img_file.stem.replace("_image", "").replace("_img", "") + "_mask.png"
            potential_mask = input_path / mask_name
            if potential_mask.exists():
                mask_files[img_file] = potential_mask

    stats.total_images = len(image_files)

    for img_file in image_files:
        try:
            # Чтение изображения
            img = np.array(Image.open(img_file))
            if img.ndim == 3 and img.shape[2] == 3:
                img = img.mean(axis=2)  # Конвертация в grayscale

            mask = None
            if save_masks and img_file in mask_files:
                mask = np.array(Image.open(mask_files[img_file]))

            # Генерация аугментаций
            for i in range(num_augmentations):
                seed = np.random.randint(0, 2**31)
                aug_img, aug_mask = apply_augmentation(
                    img,
                    mask=mask,
                    config=config,
                    seed=seed,
                )

                # Сохранение
                aug_filename = f"{img_file.stem}_aug{i}{img_file.suffix}"
                aug_img_path = output_path / aug_filename

                # Нормализация к uint8
                if aug_img.dtype == np.float32 or aug_img.dtype == np.float64:
                    aug_img = np.clip(aug_img * 255, 0, 255).astype(np.uint8)

                Image.fromarray(aug_img, mode="L").save(aug_img_path)

                if save_masks and aug_mask is not None:
                    aug_mask_path = output_path / f"{img_file.stem}_aug{i}_mask.png"
                    if aug_mask.dtype == np.float32 or aug_mask.dtype == np.float64:
                        aug_mask = (aug_mask > 0.5).astype(np.uint8) * 255
                    Image.fromarray(aug_mask, mode="L").save(aug_mask_path)

                # Обновление статистики
                stats.augmentations_applied["total"] = stats.augmentations_applied.get("total", 0) + 1

        except Exception as e:
            logger.error(f"Ошибка обработки файла {img_file}: {e}")
            stats.failed_augmentations += 1

    return stats


__all__ = [
    "AugmentationType",
    "AugmentationConfig",
    "AugmentationStats",
    "apply_rotation",
    "apply_flip",
    "apply_elastic_deformation",
    "apply_gaussian_noise",
    "apply_salt_pepper_noise",
    "apply_zoom",
    "apply_shift",
    "apply_brightness_contrast",
    "apply_augmentation",
    "generate_augmented_dataset",
]
