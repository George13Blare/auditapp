"""
Модуль визуализации медицинских изображений и разметки.
Поддерживает DICOM, NIfTI, NRRD, RTSTRUCT, SEG и другие форматы.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class ImageVolume:
    """Унифицированное представление 3D изображения."""

    data: npt.NDArray[np.float32]
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, ...] = field(default_factory=lambda: (1, 0, 0, 0, 1, 0, 0, 0, 1))
    modality: str = "CT"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationSet:
    """Унифицированное представление разметки."""

    masks: Optional[npt.NDArray[np.uint8]] = None  # [Z, Y, X, Classes] или [Z, Y, X]
    contours: List[Dict[str, Any]] = field(default_factory=list)  # Для RTSTRUCT
    keypoints: List[Dict[str, Any]] = field(default_factory=list)  # Для JSON
    bboxes: List[Dict[str, Any]] = field(default_factory=list)  # Для CSV
    class_names: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_image_volume(path: Union[str, Path]) -> ImageVolume:
    """
    Загружает изображение из различных форматов в единый формат ImageVolume.

    Поддерживаемые форматы:
    - DICOM серии (папка с .dcm файлами)
    - NIfTI (.nii, .nii.gz)
    - NRRD (.nrrd)
    - MetaImage (.mhd, .mha)
    """
    path = Path(path)

    if path.is_dir():
        return _load_dicom_series(path)

    suffix = path.suffix.lower()
    if suffix == ".gz":
        suffix = Path(path.stem).suffix.lower()

    if suffix in [".nii", ".nii.gz"]:
        return _load_nifti(path)
    elif suffix == ".nrrd":
        return _load_nrrd(path)
    elif suffix in [".mhd", ".mha"]:
        return _load_metaimage(path)
    else:
        # Попытка загрузить как DICOM файл
        try:
            return _load_dicom_single(path)
        except Exception as e:
            raise ValueError(f"Неподдерживаемый формат файла: {path}") from e


def _load_dicom_series(folder_path: Path) -> ImageVolume:
    """Загрузка серии DICOM файлов."""
    import pydicom

    dcm_files = list(folder_path.glob("*.dcm")) + list(folder_path.glob("*.DCM"))
    if not dcm_files:
        # Рекурсивный поиск
        dcm_files = list(folder_path.rglob("*.dcm")) + list(folder_path.rglob("*.DCM"))

    if not dcm_files:
        raise ValueError(f"DICOM файлы не найдены в {folder_path}")

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except Exception as e:
            logger.warning(f"Ошибка чтения {f}: {e}")

    if not slices:
        raise ValueError("Не удалось прочитать ни одного DICOM файла")

    # Сортировка по позиции среза
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        logger.warning("ImagePositionPatient не найден, сортировка по имени файла")
        slices.sort(key=lambda s: str(s.filename) if s.filename else "")

    # Сбор данных
    pixel_arrays = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        # Применение Rescale Slope/Intercept
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        arr = cast(npt.NDArray[np.float32], arr * slope + intercept)
        pixel_arrays.append(arr)

    volume = np.stack(pixel_arrays, axis=0)

    # Извлечение метаданных
    first = slices[0]
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    spacing = (
        float(pixel_spacing[0]),
        float(pixel_spacing[1]),
        float(getattr(first, "SliceThickness", 1.0)),
    )
    origin = (0.0, 0.0, 0.0)
    if hasattr(first, "ImagePositionPatient"):
        origin = (
            float(first.ImagePositionPatient[0]),
            float(first.ImagePositionPatient[1]),
            float(first.ImagePositionPatient[2]),
        )

    modality = getattr(first, "Modality", "OT")

    return ImageVolume(
        data=volume,
        spacing=spacing,
        origin=origin,
        modality=modality,
        metadata={"patient_id": getattr(first, "PatientID", ""), "study_uid": getattr(first, "StudyInstanceUID", "")},
    )


def _load_dicom_single(path: Path) -> ImageVolume:
    """Загрузка одиночного DICOM файла."""
    import pydicom

    ds = pydicom.dcmread(path, force=True)

    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = cast(npt.NDArray[np.float32], arr * slope + intercept)

    # Для 2D добавляем измерение среза
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    spacing = (
        float(pixel_spacing[0]),
        float(pixel_spacing[1]),
        float(getattr(ds, "SliceThickness", 1.0)),
    )

    return ImageVolume(
        data=arr, spacing=spacing, modality=getattr(ds, "Modality", "OT"), metadata={"filename": path.name}
    )


def _load_nifti(path: Path) -> ImageVolume:
    """Загрузка NIfTI файла."""
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("Установите nibabel: pip install nibabel") from e

    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    affine = img.affine

    # Извлечение spacing из аффинной матрицы
    spacing = (
        float(np.linalg.norm(affine[:3, 0])),
        float(np.linalg.norm(affine[:3, 1])),
        float(np.linalg.norm(affine[:3, 2])),
    )
    origin: Tuple[float, float, float] = tuple(float(x) for x in affine[:3, 3])  # type: ignore[assignment]

    return ImageVolume(data=data, spacing=spacing, origin=origin, metadata={"affine": affine})


def _load_nrrd(path: Path) -> ImageVolume:
    """Загрузка NRRD файла."""
    try:
        import nrrd
    except ImportError as e:
        raise ImportError("Установите pynrrd: pip install pynrrd") from e

    data, header = nrrd.read(str(path))
    data = data.astype(np.float32)

    spacing_vals = header.get("space directions", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    spacing_arr = np.array(spacing_vals).diagonal()
    spacing = (float(spacing_arr[0]), float(spacing_arr[1]), float(spacing_arr[2]))
    origin_vals = header.get("space origin", [0, 0, 0])
    origin = (float(origin_vals[0]), float(origin_vals[1]), float(origin_vals[2]))

    return ImageVolume(data=data, spacing=spacing, origin=origin, metadata=header)


def _load_metaimage(path: Path) -> ImageVolume:
    """Загрузка MetaImage (.mhd/.mha)."""
    try:
        import SimpleITK as sitk  # noqa: N813
    except ImportError as e:
        raise ImportError("Установите SimpleITK: pip install SimpleITK") from e

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    img = reader.Execute()

    data = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()

    return ImageVolume(data=data, spacing=spacing, origin=origin, metadata={"direction": img.GetDirection()})


def load_annotations(
    image_volume: ImageVolume, annotation_paths: List[Union[str, Path]], ref_image_path: Optional[Path] = None
) -> AnnotationSet:
    """
    Загружает разметку из различных источников.

    Args:
        image_volume: Референсное изображение для привязки размеров
        annotation_paths: Список путей к файлам разметки
        ref_image_path: Путь к референсному DICOM (для RTSTRUCT)

    Returns:
        AnnotationSet с объединенной разметкой
    """
    ann_set = AnnotationSet()

    for p in annotation_paths:
        path = Path(p)
        if not path.exists():
            logger.warning(f"Файл разметки не найден: {path}")
            continue

        suffix = path.suffix.lower()

        if suffix in [".dcm", ".dicom"]:
            # Попытка определить тип: SEG или RTSTRUCT
            try:
                import pydicom

                ds = pydicom.dcmread(path, force=True)
                modality = getattr(ds, "Modality", "")

                if modality == "SEG":
                    _load_dicom_seg(ds, ann_set)
                elif modality == "RTSTRUCT":
                    if ref_image_path:
                        _load_rtstruct(ds, ref_image_path, ann_set)
                    else:
                        logger.warning("Для RTSTRUCT нужен ref_image_path")
                else:
                    logger.warning(f"Неизвестный тип DICOM: {modality}")
            except Exception as e:
                logger.error(f"Ошибка чтения DICOM {path}: {e}")

        elif suffix in [".nii", ".nii.gz", ".nrrd"]:
            _load_mask_volume(path, ann_set, image_volume.data.shape)

        elif suffix == ".json":
            _load_keypoints_json(path, ann_set)

        elif suffix == ".csv":
            _load_bboxes_csv(path, ann_set)

    return ann_set


def _load_dicom_seg(ds: Any, ann_set: AnnotationSet) -> None:
    """Загрузка DICOM SEG."""

    try:
        # Получение количества сегментов
        _ = len(ds.SegmentSequence)

        # Чтение пиксельных данных (упрощенно, требуется полная реализация для фреймов)
        # В реальной реализации нужно обрабатывать PerFrameFunctionalGroupsSequence
        if hasattr(ds, "pixel_array"):
            mask_data = ds.pixel_array.astype(np.uint8)
            if mask_data.ndim == 2:
                mask_data = mask_data[np.newaxis, ...]
            ann_set.masks = mask_data

        # Извлечение имен классов
        for i, seg in enumerate(ds.SegmentSequence):
            name = getattr(seg, "SegmentLabel", f"Class_{i}")
            ann_set.class_names[i] = name

    except Exception as e:
        logger.error(f"Ошибка парсинга SEG: {e}")


def _load_rtstruct(ds: Any, ref_image_path: Path, ann_set: AnnotationSet) -> None:
    """Загрузка DICOM RTSTRUCT с интерполяцией контуров."""
    import pydicom

    try:
        # Загрузка референсного изображения для геометрии
        _ = pydicom.dcmread(ref_image_path, force=True)

        # Парсинг ROI Contour Sequence
        if hasattr(ds, "ROIContourSequence"):
            for roi in ds.ROIContourSequence:
                roi_name = getattr(roi, "ROIObservationLabel", "Unknown")
                color = getattr(roi, "ROIDisplayColor", [255, 255, 255])

                contours = []
                if hasattr(roi, "ContourSequence"):
                    for contour in roi.ContourSequence:
                        points = contour.ContourData
                        # Преобразование в координаты [x, y, z]
                        pts = np.array(points).reshape(-1, 3)
                        contours.append(
                            {
                                "points": pts.tolist(),
                                "z": (
                                    float(contour.ContourSlabThickness)
                                    if hasattr(contour, "ContourSlabThickness")
                                    else pts[0, 2]
                                ),
                            }
                        )

                ann_set.contours.append({"name": roi_name, "color": color, "contours": contours})
                ann_set.class_names[len(ann_set.class_names)] = roi_name

    except Exception as e:
        logger.error(f"Ошибка парсинга RTSTRUCT: {e}")


def _load_mask_volume(path: Path, ann_set: AnnotationSet, target_shape: Tuple) -> None:
    """Загрузка маски из NIfTI/NRRD."""
    vol = load_image_volume(path)
    data = vol.data.astype(np.uint8)

    # Ресемплинг к целевой форме (очень упрощенно)
    if data.shape != target_shape:
        logger.warning("Размеры маски не совпадают с изображением, требуется ресемплинг")
        # Здесь должна быть логика ресемплинга

    ann_set.masks = data


def _load_keypoints_json(path: Path, ann_set: AnnotationSet) -> None:
    """Загрузка ключевых точек из JSON (COCO-style)."""
    import json

    with open(path) as f:
        data = json.load(f)

    # Упрощенный парсинг
    if "annotations" in data:
        for ann in data["annotations"]:
            ann_set.keypoints.append(
                {
                    "category": ann.get("category_id", 0),
                    "keypoints": ann.get("keypoints", []),
                    "bbox": ann.get("bbox", []),
                }
            )

    if "categories" in data:
        for cat in data["categories"]:
            ann_set.class_names[cat["id"]] = cat["name"]


def _load_bboxes_csv(path: Path, ann_set: AnnotationSet) -> None:
    """Загрузка bounding boxes из CSV."""
    import csv

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ann_set.bboxes.append(
                {
                    "filename": row.get("filename", ""),
                    "x_min": float(row.get("x_min", 0)),
                    "y_min": float(row.get("y_min", 0)),
                    "x_max": float(row.get("x_max", 0)),
                    "y_max": float(row.get("y_max", 0)),
                    "class": row.get("class", "unknown"),
                }
            )


def render_slice(volume: npt.NDArray[np.float32], axis: int = 0, index: int = 0) -> npt.NDArray[np.float32]:
    """Извлекает 2D срез из 3D объема."""
    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    elif axis == 2:
        return volume[:, :, index]
    else:
        raise ValueError(f"Недопустимая ось: {axis}")


def compose_overlay(
    image: npt.NDArray[np.float32],
    mask: Optional[npt.NDArray[np.uint8]] = None,
    contours: Optional[List[Dict]] = None,
    bboxes: Optional[List[Dict]] = None,
    keypoints: Optional[List[Dict]] = None,
    alpha: float = 0.5,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> npt.NDArray[np.uint8]:
    """
    Создает композитное изображение с наложением разметки.

    Args:
        image: 2D массив изображения (градиентная шкала)
        mask: 2D или 3D массив маски
        contours: Список контуров для отрисовки
        bboxes: Список bounding box'ов
        keypoints: Список ключевых точек
        alpha: Прозрачность маски
        class_colors: Словарь {class_id: (R, G, B)}

    Returns:
        RGB изображение типа uint8
    """
    import matplotlib.pyplot as plt

    # Нормализация изображения для отображения
    img_norm = image.copy()
    if img_norm.max() > 0:
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

    # Конвертация в RGB
    cmap = plt.get_cmap("gray")
    rgb_img = cmap(img_norm)[..., :3] * 255
    rgb_img = cast(npt.NDArray[np.uint8], rgb_img.astype(np.uint8))

    if mask is not None:
        # Отрисовка маски
        if mask.ndim == 3:
            # Многоклассовая маска
            for cls_id in range(mask.shape[-1]):
                m = mask[..., cls_id]
                if m.sum() == 0:
                    continue
                color = class_colors.get(cls_id, (255, 0, 0)) if class_colors else (255, 0, 0)
                # Упрощенное наложение
                mask_rgba = np.zeros((*m.shape, 4), dtype=np.float32)
                mask_rgba[m > 0, :3] = [c / 255.0 for c in color]
                mask_rgba[m > 0, 3] = alpha
                rgb_img = (
                    rgb_img.astype(np.float32) * (1 - mask_rgba[..., 3:4])
                    + mask_rgba[..., :3] * 255 * mask_rgba[..., 3:4]
                ).astype(np.uint8)
        else:
            # Бинарная маска
            mask_bool: npt.NDArray[np.bool_] = mask > 0
            if mask_bool.any():
                color = (0, 255, 0)
                mask_rgba = np.zeros((*mask_bool.shape, 4), dtype=np.float32)
                mask_rgba[mask_bool, :3] = [c / 255.0 for c in color]
                mask_rgba[mask_bool, 3] = alpha
                rgb_img = (
                    rgb_img.astype(np.float32) * (1 - mask_rgba[..., 3:4])
                    + mask_rgba[..., :3] * 255 * mask_rgba[..., 3:4]
                ).astype(np.uint8)

    # Здесь можно добавить отрисовку контуров, bbox и keypoints с использованием OpenCV или PIL

    return cast(npt.NDArray[np.uint8], rgb_img)


# Заглушки для UI компонентов Streamlit (будут реализованы при интеграции)
def render_viewer_ui():
    """Рендерит интерфейс зрителя в Streamlit."""
    pass
