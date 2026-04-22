"""Тесты модуля обработки изображений."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcmmetatest.image_processor import (
    AugmentationConfig,
    PreprocessingPipelineConfig,
    apply_augmentations,
    convert_to_image_slices,
    convert_to_png,
    crop_to_nonzero,
    crop_to_roi,
    preprocess_volume_pipeline,
    resample_volume,
)


def _make_test_volume() -> np.ndarray:
    """Создает простой тестовый объем [D, H, W]."""
    return np.stack([np.full((8, 3), fill, dtype=np.float32) for fill in np.linspace(0.1, 0.9, 8)], axis=0)


@pytest.mark.parametrize(
    ("image_format", "expected_suffix"),
    [
        ("png", ".png"),
        ("jpg", ".jpg"),
        ("jpeg", ".jpg"),
        ("tiff", ".tiff"),
    ],
)
def test_convert_to_image_slices_supports_target_formats(tmp_path: Path, image_format: str, expected_suffix: str):
    """Проверяет экспорт срезов в PNG/JPG/TIFF."""
    volume = _make_test_volume()

    saved_files = convert_to_image_slices(volume, tmp_path, image_format=image_format)

    assert len(saved_files) == volume.shape[0]
    assert all(p.exists() for p in saved_files)
    assert all(p.suffix == expected_suffix for p in saved_files)


def test_convert_to_image_slices_rejects_unsupported_format(tmp_path: Path):
    """Проверяет валидацию неподдерживаемого формата."""
    volume = _make_test_volume()

    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        convert_to_image_slices(volume, tmp_path, image_format="bmp")


def test_convert_to_png_keeps_backward_compatibility(tmp_path: Path):
    """Проверяет, что convert_to_png по-прежнему сохраняет PNG-файлы."""
    volume = _make_test_volume()

    saved_files = convert_to_png(volume, tmp_path)

    assert len(saved_files) == volume.shape[0]
    assert all(p.suffix == ".png" for p in saved_files)


def test_apply_augmentations_rotates_and_flips_image_and_mask():
    """Проверяет, что геометрические аугментации синхронны для image/mask."""
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    config = AugmentationConfig(rotate_k=1, flip_horizontal=True)
    aug_image, aug_mask = apply_augmentations(image, config, mask)

    expected_image = np.array([[4, 2], [3, 1]], dtype=np.float32)
    expected_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    assert np.array_equal(aug_image, expected_image)
    assert aug_mask is not None
    assert np.array_equal(aug_mask, expected_mask)


def test_apply_augmentations_adds_noise_with_reproducible_seed():
    """Проверяет воспроизводимое добавление гауссова шума."""
    image = np.zeros((2, 2), dtype=np.float32)
    config = AugmentationConfig(add_gaussian_noise=True, noise_std=0.5, random_seed=7)

    aug_image_1, _ = apply_augmentations(image, config)
    aug_image_2, _ = apply_augmentations(image, config)

    assert not np.array_equal(aug_image_1, image)
    assert np.allclose(aug_image_1, aug_image_2)


def test_apply_augmentations_validates_mask_shape():
    """Проверяет валидацию размера маски."""
    image = np.zeros((4, 4), dtype=np.float32)
    wrong_mask = np.zeros((3, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="Размер маски"):
        apply_augmentations(image, AugmentationConfig(), wrong_mask)


def test_crop_to_nonzero_returns_tight_bbox_with_margin():
    """Проверяет кроппинг ненулевой области с margin."""
    volume = np.zeros((6, 8, 10), dtype=np.float32)
    volume[2:4, 3:5, 4:7] = 1.0

    cropped, bbox = crop_to_nonzero(volume, threshold=0.5, margin=1)

    assert bbox == (slice(1, 5), slice(2, 6), slice(3, 8))
    assert cropped.shape == (4, 4, 5)
    assert cropped.max() == 1.0


def test_crop_to_nonzero_returns_full_array_when_empty():
    """Если значимых пикселей нет, возвращается исходный массив."""
    image = np.zeros((5, 5), dtype=np.float32)
    cropped, bbox = crop_to_nonzero(image)

    assert np.array_equal(cropped, image)
    assert bbox == (slice(0, 5), slice(0, 5))


def test_crop_to_roi_crops_image_and_mask():
    """ROI-кроп должен одинаково обрезать image и mask."""
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    mask = (image > 10).astype(np.uint8)
    roi = (slice(1, 4), slice(2, 5))

    cropped_image, cropped_mask = crop_to_roi(image, roi, mask)

    assert cropped_image.shape == (3, 3)
    assert cropped_mask is not None
    assert np.array_equal(cropped_mask, mask[roi])
    assert np.array_equal(cropped_image, image[roi])


def test_resample_volume_changes_shape_according_to_spacing():
    """Проверяет пересчет shape при изменении spacing."""
    volume = np.ones((4, 6, 8), dtype=np.float32)
    current_spacing = (2.0, 1.0, 1.0)
    target_spacing = (1.0, 1.0, 1.0)

    resampled, out_spacing = resample_volume(volume, current_spacing, target_spacing)

    assert resampled.shape == (8, 6, 8)
    assert out_spacing == target_spacing


def test_resample_volume_mask_mode_uses_nearest_neighbor():
    """Для масок должен использоваться nearest-neighbor без дробных значений."""
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[1, 1, 1] = 3

    resampled, _ = resample_volume(mask, (2.0, 2.0, 2.0), (1.0, 1.0, 1.0), is_mask=True)

    assert resampled.dtype == np.uint8
    assert set(np.unique(resampled)) <= {0, 3}


def test_resample_volume_validates_input():
    """Проверяет валидацию входных параметров ресемплинга."""
    with pytest.raises(ValueError, match="3D объем"):
        resample_volume(np.zeros((10, 10), dtype=np.float32), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="Spacing должен быть положительным"):
        resample_volume(np.zeros((2, 2, 2), dtype=np.float32), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0))


def test_preprocess_volume_pipeline_combines_steps():
    """Проверяет единый pipeline: normalize -> resample -> crop -> augment."""
    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[:, 1:3, 1:3] = 10
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[:, 1:3, 1:3] = 1

    config = PreprocessingPipelineConfig(
        normalization_method="minmax",
        enable_resampling=True,
        target_spacing=(1.0, 1.0, 1.0),
        crop_nonzero=True,
        crop_threshold=0.01,
        crop_margin=0,
        enable_augmentation=True,
        augmentation=AugmentationConfig(flip_horizontal=True, random_seed=1),
    )

    processed, processed_mask, out_spacing = preprocess_volume_pipeline(
        volume=volume,
        spacing=(2.0, 1.0, 1.0),
        config=config,
        mask=mask,
    )

    assert out_spacing == (1.0, 1.0, 1.0)
    assert processed.shape[0] == 4  # по Z после ресемплинга x2
    assert processed_mask is not None
    assert processed_mask.shape == processed.shape
