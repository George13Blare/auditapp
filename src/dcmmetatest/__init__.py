"""DCMMETATEST Plus — расширенный анализатор DICOM-датасетов."""

from .analyzer import (
    print_summary,
    run_analysis,
)
from .cli import (
    create_parser,
    interactive_setup,
    main,
)
from .detectors import (
    LABEL_FILE_PATTERNS,
    LABEL_SERIES_KEYWORDS,
    LABEL_SOP_CLASS_UIDS,
    detect_label_from_dataset,
)
from .io import (
    find_dicom_studies_by_dir,
    find_dicom_studies_by_uid,
    iter_all_files,
    load_config_file,
    process_study_dir,
    process_study_uid,
    save_report_csv,
    save_report_json,
    save_report_txt,
)
from .models import (
    AnalysisReport,
    StudyResult,
    WorkerConfig,
)
from .utils import (
    check_dicom_anonymization,
    configure_logging,
    extract_modality,
    prompt_choice,
    prompt_with_default,
    prompt_yes_no,
    read_dicom_header,
    should_exclude,
)

__version__ = "1.0.0"
__all__ = [
    "StudyResult",
    "WorkerConfig",
    "AnalysisReport",
    "configure_logging",
    "read_dicom_header",
    "should_exclude",
    "prompt_with_default",
    "prompt_yes_no",
    "prompt_choice",
    "extract_modality",
    "check_dicom_anonymization",
    "detect_label_from_dataset",
    "LABEL_SOP_CLASS_UIDS",
    "LABEL_SERIES_KEYWORDS",
    "LABEL_FILE_PATTERNS",
    "iter_all_files",
    "find_dicom_studies_by_dir",
    "find_dicom_studies_by_uid",
    "process_study_dir",
    "process_study_uid",
    "load_config_file",
    "save_report_txt",
    "save_report_csv",
    "save_report_json",
    "run_analysis",
    "print_summary",
    "create_parser",
    "interactive_setup",
    "main",
    "is_dicom_file",
    "is_label_json",
]
