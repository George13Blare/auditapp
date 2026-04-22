"""Тесты модуля валидации датасета."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcmmetatest.validation import _find_duplicate_values, scan_dataset_anomalies


def test_find_duplicate_values_returns_only_duplicates():
    data = [
        ("uid-1", "file-a"),
        ("uid-2", "file-b"),
        ("uid-1", "file-c"),
    ]
    duplicates = _find_duplicate_values(data)

    assert "uid-1" in duplicates
    assert duplicates["uid-1"] == ["file-a", "file-c"]
    assert "uid-2" not in duplicates


def test_scan_dataset_anomalies_invalid_path():
    result = scan_dataset_anomalies("/path/that/does/not/exist")

    assert result["scanned_files"] == 0
    assert result["errors"]
