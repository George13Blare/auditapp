"""DCMMETATEST Plus — расширенный анализатор DICOM-датасетов."""

from .models import (
    StudyResult,
    WorkerConfig,
    AnalysisReport,
)
from .utils import (
    configure_logging,
    read_dicom_header,
    should_exclude,
    prompt_with_default,
    prompt_yes_no,
    prompt_choice,
    extract_modality,
    check_dicom_anonymization,
)
from .detectors import (
    detect_label_from_dataset,
    LABEL_SOP_CLASS_UIDS,
    LABEL_SERIES_KEYWORDS,
    LABEL_FILE_PATTERNS,
)
from .io import (
    iter_all_files,
    find_dicom_studies_by_dir,
    find_dicom_studies_by_uid,
    process_study_dir,
    process_study_uid,
    load_config_file,
    save_report_txt,
    save_report_csv,
    save_report_json,
)
from .analyzer import (
    run_analysis,
    print_summary,
)
from .cli import (
    create_parser,
    interactive_setup,
    main,
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
