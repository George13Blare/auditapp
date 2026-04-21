"""Тесты для dcmmetatest_plus.py."""

import pytest
from pathlib import Path
import tempfile
import os


class TestHelpers:
    """Тесты вспомогательных функций."""

    def test_should_exclude_no_patterns(self):
        """Тест: без шаблонов исключений ничего не исключается."""
        from dcmmetatest_plus import should_exclude
        
        path = Path("/test/file.dcm")
        assert should_exclude(path, ()) is False
        assert should_exclude(path, tuple()) is False

    def test_should_exclude_with_pattern(self):
        """Тест: исключение по шаблону."""
        from dcmmetatest_plus import should_exclude
        
        path = Path("/test/temp/file.dcm")
        assert should_exclude(path, ("**/temp/*",)) is True
        assert should_exclude(path, ("*.bak",)) is False

    def test_yes_no_values(self):
        """Тест: значения YES_VALUES и NO_VALUES."""
        from dcmmetatest_plus import YES_VALUES, NO_VALUES
        
        assert "yes" in YES_VALUES
        assert "да" in YES_VALUES
        assert "y" in YES_VALUES
        assert "no" in NO_VALUES
        assert "нет" in NO_VALUES
        assert "n" in NO_VALUES


class TestDICOMDetection:
    """Тесты определения DICOM-файлов."""

    def test_is_dicom_file_nonexistent(self):
        """Тест: несуществующий файл не является DICOM."""
        from dcmmetatest_plus import is_dicom_file
        
        assert is_dicom_file("/nonexistent/file.dcm") is False

    def test_is_dicom_file_empty_file(self, tmp_path):
        """Тест: пустой файл не является DICOM."""
        from dcmmetatest_plus import is_dicom_file
        
        empty_file = tmp_path / "empty.dcm"
        empty_file.write_bytes(b"")
        assert is_dicom_file(str(empty_file)) is False


class TestStudyResult:
    """Тесты структуры StudyResult."""

    def test_study_result_creation(self):
        """Тест: создание StudyResult."""
        from dcmmetatest_plus import StudyResult
        
        result = StudyResult(
            study_key="test_study",
            has_label=False,
            non_anon_patients=[],
            modalities=["CT"],
            errors=[],
            study_path_rep="/test/path",
            label_sources=set(),
            series={},
            file_count=10,
            patient_ids=["P001"],
        )
        
        assert result.study_key == "test_study"
        assert result.has_label is False
        assert result.modalities == ["CT"]
        assert result.file_count == 10


class TestWorkerConfig:
    """Тесты конфигурации воркера."""

    def test_worker_config_defaults(self):
        """Тест: значения по умолчанию WorkerConfig."""
        from dcmmetatest_plus import WorkerConfig
        
        config = WorkerConfig()
        assert config.modality_filter is None
        assert config.strict is False
        assert config.exclude_patterns == ()
        assert config.detect_label_json is True

    def test_worker_config_custom(self):
        """Тест: кастомная конфигурация WorkerConfig."""
        from dcmmetatest_plus import WorkerConfig
        
        config = WorkerConfig(
            modality_filter={"CT", "MR"},
            strict=True,
            exclude_patterns=("**/temp/*",),
        )
        
        assert config.modality_filter == {"CT", "MR"}
        assert config.strict is True
        assert "**/temp/*" in config.exclude_patterns


class TestLabelDetection:
    """Тесты детекции разметки."""

    def test_label_sop_class_uids_defined(self):
        """Тест: SOP Class UID для разметки определены."""
        from dcmmetatest_plus import LABEL_SOP_CLASS_UIDS
        
        # RT Structure Set Storage
        assert "1.2.840.10008.5.1.4.1.1.481.3" in LABEL_SOP_CLASS_UIDS
        # Segmentation Storage
        assert "1.2.840.10008.5.1.4.1.1.66.4" in LABEL_SOP_CLASS_UIDS

    def test_label_series_keywords_defined(self):
        """Тест: ключевые слова для серий с разметкой определены."""
        from dcmmetatest_plus import LABEL_SERIES_KEYWORDS
        
        assert "seg" in LABEL_SERIES_KEYWORDS
        assert "mask" in LABEL_SERIES_KEYWORDS
        assert "label" in LABEL_SERIES_KEYWORDS

    def test_label_file_patterns_defined(self):
        """Тест: шаблоны файлов разметки определены."""
        from dcmmetatest_plus import LABEL_FILE_PATTERNS
        
        assert "*.nii" in LABEL_FILE_PATTERNS
        assert "*.nii.gz" in LABEL_FILE_PATTERNS
        assert "*.mha" in LABEL_FILE_PATTERNS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
