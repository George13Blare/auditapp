"""Сервисы intake: валидация входа и безопасные файловые операции."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetPathConfig:
    """Конфигурация проверки входного пути датасета."""

    raw_path: str


@dataclass(frozen=True)
class DatasetPathArtifact:
    """Результат intake-проверки пути."""

    is_valid: bool
    resolved_path: str | None
    message: str


@dataclass(frozen=True)
class DatasetScanConfig:
    """Параметры сканирования структуры."""

    root_path: str
    max_depth: int = 3
    max_items_per_dir: int = 50


@dataclass(frozen=True)
class DatasetScanArtifact:
    """Дерево структуры датасета для UI."""

    root: dict[str, Any] | None


@dataclass(frozen=True)
class FileOperationRequest:
    """Запрос на файловую операцию."""

    item_path: str
    new_name: str | None = None


@dataclass(frozen=True)
class FileOperationArtifact:
    """Унифицированный результат файловой операции."""

    success: bool
    message: str
    updated_path: str | None = None


def validate_dataset_path(config: DatasetPathConfig) -> DatasetPathArtifact:
    if not config.raw_path:
        return DatasetPathArtifact(False, None, "Путь не указан")

    path = Path(config.raw_path)
    if not path.exists():
        return DatasetPathArtifact(False, None, f"Путь не существует: {config.raw_path}")
    if not path.is_dir():
        return DatasetPathArtifact(False, None, f"Это не директория: {config.raw_path}")

    resolved = str(path.resolve()) if not path.is_absolute() else str(path)
    return DatasetPathArtifact(True, resolved, f"Путь проверен: {resolved}")


def scan_dataset_structure(config: DatasetScanConfig) -> DatasetScanArtifact:
    root_path = Path(config.root_path)
    if not root_path.exists():
        return DatasetScanArtifact(root=None)

    def scan_dir(dir_path: Path, current_depth: int) -> dict[str, Any] | None:
        if current_depth > config.max_depth:
            return None

        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return {"name": dir_path.name, "type": "dir", "error": "Нет доступа", "children": []}

        children: list[dict[str, Any]] = []
        for item in items[: config.max_items_per_dir]:
            if item.is_dir():
                child = scan_dir(item, current_depth + 1)
                if child:
                    children.append(child)
            else:
                children.append({"name": item.name, "type": "file", "size": item.stat().st_size})

        return {
            "name": dir_path.name,
            "type": "dir",
            "path": str(dir_path),
            "children": children,
            "total_items": len(items),
        }

    return DatasetScanArtifact(root=scan_dir(root_path, 0))


def delete_fs_item(request: FileOperationRequest) -> FileOperationArtifact:
    path = Path(request.item_path)
    if not path.exists():
        return FileOperationArtifact(False, "Файл/папка не существует")

    try:
        if path.is_file():
            path.unlink()
            return FileOperationArtifact(True, f"Файл удалён: {request.item_path}")
        if path.is_dir():
            if any(path.iterdir()):
                return FileOperationArtifact(False, "Папка не пуста. Удалите сначала содержимое.")
            path.rmdir()
            return FileOperationArtifact(True, f"Папка удалена: {request.item_path}")
    except Exception as exc:  # noqa: BLE001
        return FileOperationArtifact(False, f"Ошибка удаления: {exc!s}")

    return FileOperationArtifact(False, "Неизвестная ошибка")


def rename_fs_item(request: FileOperationRequest) -> FileOperationArtifact:
    old = Path(request.item_path)
    if not old.exists():
        return FileOperationArtifact(False, "Файл/папка не существует")
    if not request.new_name:
        return FileOperationArtifact(False, "Новое имя не указано")

    new = old.parent / request.new_name
    if new.exists():
        return FileOperationArtifact(False, "Файл/папка с таким именем уже существует")

    try:
        old.rename(new)
        return FileOperationArtifact(True, f"Переименовано в: {request.new_name}", str(new))
    except Exception as exc:  # noqa: BLE001
        return FileOperationArtifact(False, f"Ошибка переименования: {exc!s}")
