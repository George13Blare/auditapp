#!/usr/bin/env python3
"""
Streamlit приложение для визуального анализа DICOM-датасетов.

Запуск:
    streamlit run app.py

Или через модуль:
    python -m streamlit run app.py
"""

import json
import logging
from pathlib import Path

import streamlit as st

from src.dcmmetatest.image_processor import AugmentationConfig, PreprocessingPipelineConfig
from src.dcmmetatest.normalizer import NormalizationConfig, SplitConfig
from src.dcmmetatest.services import (
    AnalysisRequest,
    DatasetPathConfig,
    DatasetPreprocessRequest,
    DatasetScanConfig,
    FileOperationRequest,
    NormalizeRequest,
    PreprocessSeriesRequest,
    ReportManifestConfig,
    SplitRequest,
    build_report_manifest,
    delete_fs_item,
    rename_fs_item,
    run_analysis,
    run_dataset_preprocessing,
    run_normalize,
    run_preprocess_series,
    run_segmentation_analysis,
    run_split,
    scan_dataset_structure,
    validate_dataset_path,
)
from src.dcmmetatest.ui import (
    convert_report_to_dataframe,
    create_age_distribution_chart,
    create_label_source_bar_chart,
    create_modality_pie_chart,
    create_quality_metrics_cards,
    create_study_date_timeline,
)
from src.dcmmetatest.validation import scan_dataset_anomalies

# Настройка страницы
st.set_page_config(
    page_title="DICOM Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Заголовок
st.title("🏥 DICOM Dataset Analyzer")
st.markdown(
    """
Интерактивный инструмент для анализа DICOM-датасетов, проверки разметки и качества данных.
"""
)

# Инициализация session state
if "analysis_in_progress" not in st.session_state:
    st.session_state.analysis_in_progress = False
if "progress_value" not in st.session_state:
    st.session_state.progress_value = 0
if "progress_status" not in st.session_state:
    st.session_state.progress_status = ""
if "dataset_structure" not in st.session_state:
    st.session_state.dataset_structure = None
if "selected_folder_path" not in st.session_state:
    st.session_state.selected_folder_path = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []  # История анализов
if "structure_templates" not in st.session_state:
    st.session_state.structure_templates = {}  # Шаблоны структуры
if "schema_builder_levels" not in st.session_state:
    st.session_state.schema_builder_levels = ["patient_level", "study_level", "series_level"]
if "schema_builder_file_patterns" not in st.session_state:
    st.session_state.schema_builder_file_patterns = "*.dcm"
if "current_report" not in st.session_state:
    st.session_state.current_report = None  # Текущий отчёт для доступа из других вкладок

# Боковая панель
st.sidebar.header("⚙️ Настройки анализа")

# Ввод пути
folder_path = st.sidebar.text_input(
    "Путь к датасету",
    placeholder="/path/to/dicom/dataset",
    help="Укажите абсолютный путь к корневой директории с DICOM-файлами",
)

# Опции конфигурации
st.sidebar.subheader("Параметры")

group_by = st.sidebar.selectbox(
    "Группировка",
    ["study", "directory"],
    index=0,
    help="study: по StudyInstanceUID, directory: по папкам",
)

max_workers = st.sidebar.slider(
    "Потоков обработки",
    min_value=1,
    max_value=16,
    value=4,
    help="Количество воркеров для параллельной обработки",
)

show_progress = st.sidebar.checkbox("Показывать прогресс", value=True)

follow_symlinks = st.sidebar.checkbox("Следовать за симлинками", value=False)

only_labeled = st.sidebar.checkbox("Только размеченные", value=False)

modality_filter = st.sidebar.multiselect(
    "Фильтр по модальности",
    options=["CT", "MR", "CR", "DX", "US", "PT", "NM", "MG", "XA", "RF"],
    default=[],
    help="Оставьте пустым для всех модальностей",
)

# Кнопка запуска анализа
analyze_button = st.sidebar.button("🚀 Запустить анализ", type="primary", disabled=not folder_path)

st.sidebar.divider()
st.sidebar.subheader("🧪 Preprocessing pipeline")
preprocess_input_path = st.sidebar.text_input(
    "Путь к DICOM серии",
    value=folder_path,
    help="Папка с DICOM-срезами одной серии",
)
preprocess_output_path = st.sidebar.text_input(
    "Папка экспорта",
    value="",
    help="Куда сохранить обработанные данные",
)
preprocess_format = st.sidebar.selectbox("Формат экспорта", ["png", "jpg", "tiff", "nifti"], index=0)
preprocess_normalization = st.sidebar.selectbox("Нормализация", ["minmax", "zscore", "sigmoid"], index=0)
preprocess_resample = st.sidebar.checkbox("Ресемплинг до 1x1x1 мм", value=False)
preprocess_crop = st.sidebar.checkbox("Air/ROI кроппинг", value=False)
preprocess_augment = st.sidebar.checkbox("Аугментация (flip + noise)", value=False)
run_preprocess_button = st.sidebar.button(
    "⚙️ Запустить preprocessing",
    disabled=not preprocess_input_path or not preprocess_output_path,
)


# Зона прогресс бара (отдельная выделенная область)
progress_container = st.container()

# Preprocessing pipeline block
if run_preprocess_button:
    preprocess_path = Path(preprocess_input_path)
    export_path = Path(preprocess_output_path)

    if not preprocess_path.exists() or not preprocess_path.is_dir():
        st.error(f"❌ Некорректный путь к серии: {preprocess_path}")
    else:
        with st.spinner("Выполняется preprocessing pipeline..."):
            preprocessing_config = PreprocessingPipelineConfig(
                normalization_method=preprocess_normalization,
                target_spacing=(1.0, 1.0, 1.0) if preprocess_resample else None,
                enable_resampling=preprocess_resample,
                crop_nonzero=preprocess_crop,
                crop_threshold=0.0,
                crop_margin=1,
                enable_augmentation=preprocess_augment,
                augmentation=AugmentationConfig(flip_horizontal=True, add_gaussian_noise=preprocess_augment),
                export_format=preprocess_format,
            )
            preprocess_stats = run_preprocess_series(
                PreprocessSeriesRequest(
                    input_series_dir=str(preprocess_path),
                    output_dir=str(export_path),
                    config=preprocessing_config,
                )
            )

        if preprocess_stats["errors"]:
            st.error(f"❌ Preprocessing завершился с ошибкой: {preprocess_stats['errors'][0]}")
        else:
            st.success(
                f"✅ Preprocessing завершён. Сохранено файлов: {preprocess_stats['files_saved']} "
                f"(формат: {preprocess_stats['export_format']})"
            )

# Основная область
if analyze_button:
    # Валидация пути
    intake_artifact = validate_dataset_path(DatasetPathConfig(raw_path=folder_path))

    is_valid, path_or_error = intake_artifact.is_valid, intake_artifact.resolved_path or intake_artifact.message

    if not is_valid:
        st.error(f"❌ Ошибка: {path_or_error}")
        st.stop()

    valid_path = path_or_error
    st.session_state.selected_folder_path = valid_path
    st.success(f"✅ Путь проверен: {valid_path}")

    # Подготовка конфигурации
    config_dict = {
        "group_by": group_by,
        "max_workers": max_workers,
        "show_progress": show_progress,
        "follow_symlinks": follow_symlinks,
        "only_labeled": only_labeled,
        "modality_filter": modality_filter if modality_filter else None,
        "pool_type": "process",
        "strict": False,
        "list_empty": True,
        "max_depth": None,
        "exclude_patterns": [],
        "min_files": 0,
        "only_non_anon": False,
    }

    # Запуск анализа с индикатором прогресса в отдельной зоне
    with progress_container:
        st.markdown("### ⏳ Прогресс анализа")
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_status = st.empty()

        # Имитация прогресса (т.к. анализ может быть длительным)
        progress_status.info("🔍 Начато сканирование датасета...")

        try:
            # Асинхронный запуск с обновлением прогресса
            import threading

            analysis_complete = threading.Event()
            analysis_result = {}
            analysis_error = {}

            def run_analysis_thread():
                try:
                    result = run_analysis(AnalysisRequest(dataset_path=valid_path, config_dict=config_dict))
                    analysis_result["report"] = result
                except Exception as e:
                    analysis_error["error"] = e
                finally:
                    analysis_complete.set()

            thread = threading.Thread(target=run_analysis_thread)
            thread.start()

            # Обновление прогресс бара
            progress_value = 0
            while not analysis_complete.wait(timeout=0.5):
                progress_value = min(progress_value + 5, 90)
                progress_bar.progress(progress_value)
                progress_text.text(f"Обработка... {progress_value}%")

            thread.join()

            if analysis_error:
                raise analysis_error["error"]

            report = analysis_result["report"]
            st.session_state.current_report = report  # Сохраняем отчёт для доступа из других вкладок
            progress_bar.progress(100)
            progress_text.text("Анализ завершён!")
            progress_status.success("✅ Анализ успешно завершён")

            # Добавляем в историю анализа
            import datetime

            history_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "path": valid_path,
                "total_studies": len(report.results),
                "total_files": sum(s.file_count for s in report.results),
                "non_anon_count": len(report.non_anon_patients),
                "config": config_dict.copy(),
            }
            st.session_state.analysis_history.append(history_entry)

        except Exception as e:
            st.error(f"❌ Ошибка при анализе: {str(e)}")
            logger.exception("Analysis failed")
            progress_status.error(f"❌ Ошибка: {str(e)}")
            st.stop()

    # Отображение результатов
    st.divider()

    # Карточки метрик
    metrics = create_quality_metrics_cards(report)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📊 Исследований", metrics["total_studies"])
        st.metric("📁 Файлов", metrics["total_files"])

    with col2:
        st.metric("👥 Пациентов", metrics["unique_patients"])
        st.metric("📈 Среднее файлов/исследование", metrics["avg_files_per_study"])

    with col3:
        st.metric("🏷️ Размечено", f"{metrics['labeled_percent']}%")
        st.metric("⚠️ Неанонимизировано", f"{metrics['non_anon_percent']}%")

    with col4:
        st.metric("📂 Пустых папок", metrics["empty_folders"])
        st.metric("❌ Ошибок", metrics["errors_count"])

    with col5:
        st.metric("🔍 Неанон. файлов", metrics["non_anon_files_total"])
        st.metric("⚡ Проблем качества", metrics["quality_issues_total"])

    st.divider()

    # Вкладки с добавлением редактора структуры, конфигуратора и истории
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "📈 Графики",
            "📋 Таблица данных",
            "⚠️ Проблемы",
            "📄 Экспорт",
            "🗂️ Редактор структуры",
            "⚙️ Конфигуратор структуры",
            "🔐 Анонимизатор",
            "📜 История анализов",
            "🧪 Бета: Нормализация и сплит",
        ]
    )

    with tab1:
        st.subheader("Визуализация данных")

        col_chart1, col_chart2, col_chart3 = st.columns(3)

        with col_chart1:
            fig_modality = create_modality_pie_chart(report)
            st.plotly_chart(fig_modality, use_container_width=True)

        with col_chart2:
            fig_labels = create_label_source_bar_chart(report)
            st.plotly_chart(fig_labels, use_container_width=True)

        with col_chart3:
            fig_timeline = create_study_date_timeline(report)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)

        # Дополнительный ряд графиков
        col_chart4, col_chart5 = st.columns(2)

        with col_chart4:
            fig_age = create_age_distribution_chart(report)
            if fig_age:
                st.plotly_chart(fig_age, use_container_width=True)

    with tab2:
        st.subheader("Детальные данные по исследованиям")

        if report.results:
            df = convert_report_to_dataframe(report)

            # Фильтрация в таблице
            search_query = st.text_input("🔍 Поиск по таблице", placeholder="Patient ID, Study Key...")

            if search_query:
                mask = df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)
                df_filtered = df[mask]
            else:
                df_filtered = df

            st.write(f"Найдено исследований: {len(df_filtered)} из {len(df)}")

            # Интерактивная таблица
            st.dataframe(
                df_filtered,
                use_container_width=True,
                height=500,
                hide_index=True,
            )
        else:
            st.info("Нет данных для отображения")

    with tab3:
        st.subheader("Проблемы и ошибки")

        if report.errors:
            st.error(f"Обнаружено ошибок: {len(report.errors)}")
            for i, err in enumerate(report.errors[:20], 1):
                st.code(f"{i}. {err}", language="text")

            if len(report.errors) > 20:
                st.warning(f"... и ещё {len(report.errors) - 20} ошибок")
        else:
            st.success("✅ Ошибок не обнаружено")

        if report.empty_folders:
            st.warning(f"Пустых папок: {len(report.empty_folders)}")
            with st.expander("Показать список пустых папок"):
                for folder in report.empty_folders:
                    st.text(folder)

        st.divider()
        st.markdown("### 🧪 Бета: углублённый поиск аномалий")
        st.caption("Проверяет битые DICOM, пустые SEG-маски и дубликаты SOPInstanceUID.")

        if st.button("🔍 Запустить сканер аномалий", key="scan_anomalies_btn"):
            with st.spinner("Сканирование аномалий..."):
                anomalies = scan_dataset_anomalies(st.session_state.selected_folder_path, max_files=10000)

            if anomalies["errors"]:
                st.error(f"Ошибки сканера: {'; '.join(anomalies['errors'])}")
            else:
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                with col_a1:
                    st.metric("Проверено файлов", anomalies["scanned_files"])
                with col_a2:
                    st.metric("Битые файлы", len(anomalies["broken_files"]))
                with col_a3:
                    st.metric("Пустые SEG", len(anomalies["empty_seg_masks"]))
                with col_a4:
                    st.metric("Дубликаты SOP UID", len(anomalies["duplicate_sop_instance_uid"]))

                if anomalies["broken_files"]:
                    with st.expander("Показать битые файлы"):
                        for p in anomalies["broken_files"][:100]:
                            st.text(p)

                if anomalies["empty_seg_masks"]:
                    with st.expander("Показать пустые SEG-маски"):
                        for p in anomalies["empty_seg_masks"][:100]:
                            st.text(p)

                if anomalies["duplicate_sop_instance_uid"]:
                    with st.expander("Показать дубликаты SOPInstanceUID"):
                        st.json(anomalies["duplicate_sop_instance_uid"])

    with tab4:
        st.subheader("Экспорт отчёта")

        st.markdown(
            """
        Выберите формат для экспорта результатов анализа.
        """
        )

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            if st.button("📄 CSV", use_container_width=True):
                if report.results:
                    df = convert_report_to_dataframe(report)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Скачать CSV",
                        data=csv,
                        file_name="dicom_analysis_report.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Нет данных для экспорта")

        with col_exp2:
            if st.button("📝 JSON", use_container_width=True):
                manifest = build_report_manifest(report, metrics, ReportManifestConfig())
                st.download_button(
                    label="Скачать JSON",
                    data=manifest.json_payload,
                    file_name="dicom_analysis_report.json",
                    mime="application/json",
                )

        with col_exp3:
            if st.button("📋 TXT", use_container_width=True):
                manifest = build_report_manifest(report, metrics, ReportManifestConfig())
                st.download_button(
                    label="Скачать TXT",
                    data=manifest.text_payload,
                    file_name="dicom_analysis_report.txt",
                    mime="text/plain",
                )

    # Вкладка редактора структуры датасета
    with tab5:
        st.subheader("🗂️ Редактор структуры датасета")
        st.markdown(
            """
        Инструмент для просмотра и редактирования структуры вашего DICOM-датасета.

        **Возможности:**
        - Просмотр древовидной структуры
        - Переименование файлов и папок
        - Удаление пустых папок и файлов
        """
        )

        if st.session_state.selected_folder_path:
            # Кнопка обновления структуры
            if st.button("🔄 Обновить структуру", key="refresh_structure"):
                st.session_state.dataset_structure = scan_dataset_structure(
                    DatasetScanConfig(root_path=st.session_state.selected_folder_path)
                ).root
                st.rerun()

            # Сканирование структуры если ещё не сделано
            if st.session_state.dataset_structure is None:
                with st.spinner("Сканирование структуры..."):
                    st.session_state.dataset_structure = scan_dataset_structure(
                        DatasetScanConfig(root_path=st.session_state.selected_folder_path)
                    ).root

            if st.session_state.dataset_structure:
                # Функция для рекурсивного отображения дерева
                def render_tree(node, level=0, parent_key=""):
                    "  " * level
                    node_key = f"{parent_key}/{node['name']}" if parent_key else node["name"]

                    if node["type"] == "dir":
                        # Отображение папки
                        col_icon, col_name, col_actions = st.columns([0.1, 0.7, 0.2])

                        with col_icon:
                            st.write("📁")

                        with col_name:
                            st.markdown(f"**{node['name']}**")
                            if "total_items" in node:
                                st.caption(f"Элементов: {node['total_items']}")

                        with col_actions:
                            # Переименование
                            new_name = st.text_input(
                                "Переименовать",
                                value=node["name"],
                                key=f"rename_{node_key}",
                                placeholder="Новое имя",
                                label_visibility="collapsed",
                            )
                            if st.button("✏️", key=f"btn_rename_{node_key}", help="Переименовать"):
                                if new_name and new_name != node["name"]:
                                    rename_result = rename_fs_item(
                                        FileOperationRequest(item_path=node.get("path", ""), new_name=new_name)
                                    )
                                    success, msg = rename_result.success, rename_result.message
                                    if success:
                                        st.success(msg)
                                        st.session_state.dataset_structure = None  # Сброс кэша
                                        st.rerun()
                                    else:
                                        st.error(msg)

                            # Удаление (только пустые папки)
                            if st.button("🗑️", key=f"btn_delete_{node_key}", help="Удалить пустую папку"):
                                delete_result = delete_fs_item(FileOperationRequest(item_path=node.get("path", "")))
                                success, msg = delete_result.success, delete_result.message
                                if success:
                                    st.success(msg)
                                    st.session_state.dataset_structure = None  # Сброс кэша
                                    st.rerun()
                                else:
                                    st.error(msg)

                        # Рекурсивный рендеринг дочерних элементов
                        if "children" in node and node["children"]:
                            with st.expander(f"Открыть папку {node['name']}", expanded=(level < 1)):
                                for child in node["children"]:
                                    render_tree(child, level + 1, node_key)

                    else:
                        # Отображение файла
                        col_icon, col_name, col_size, col_actions = st.columns([0.1, 0.6, 0.15, 0.15])

                        with col_icon:
                            st.write("📄")

                        with col_name:
                            st.write(node["name"])

                        with col_size:
                            size_kb = node.get("size", 0) / 1024
                            if size_kb < 1024:
                                st.caption(f"{size_kb:.1f} KB")
                            else:
                                st.caption(f"{size_kb/1024:.1f} MB")

                        with col_actions:
                            # Переименование файла
                            new_name = st.text_input(
                                "Имя",
                                value=node["name"],
                                key=f"rename_{node_key}",
                                placeholder="Новое имя",
                                label_visibility="collapsed",
                            )
                            if st.button("✏️", key=f"btn_rename_{node_key}", help="Переименовать"):
                                if new_name and new_name != node["name"]:
                                    # Для файлов нужен путь
                                    file_path = str(Path(st.session_state.selected_folder_path) / node_key)
                                    rename_result = rename_fs_item(
                                        FileOperationRequest(item_path=file_path, new_name=new_name)
                                    )
                                    success, msg = rename_result.success, rename_result.message
                                    if success:
                                        st.success(msg)
                                        st.session_state.dataset_structure = None  # Сброс кэша
                                        st.rerun()
                                    else:
                                        st.error(msg)

                            # Удаление файла
                            if st.button("🗑️", key=f"btn_delete_{node_key}", help="Удалить файл"):
                                file_path = str(Path(st.session_state.selected_folder_path) / node_key)
                                delete_result = delete_fs_item(FileOperationRequest(item_path=file_path))
                                success, msg = delete_result.success, delete_result.message
                                if success:
                                    st.success(msg)
                                    st.session_state.dataset_structure = None  # Сброс кэша
                                    st.rerun()
                                else:
                                    st.error(msg)

                # Рендеринг дерева начиная с корня
                render_tree(st.session_state.dataset_structure)
            else:
                st.warning("Не удалось сканировать структуру датасета")
        else:
            st.info("Сначала запустите анализ датасета")

    # Вкладка конфигуратора структуры (редактор шаблонов)
    with tab6:
        st.subheader("⚙️ Конфигуратор структуры датасета")
        st.markdown(
            """
        Инструмент для создания и управления шаблонами ожидаемой структуры DICOM-датасета.

        **Возможности:**
        - Создание шаблонов структуры для разных типов исследований
        - Валидация реального датасета по шаблону
        - Сохранение и загрузка шаблонов в JSON
        - Автоматическое выявление отклонений от шаблона
        """
        )

        col_tmpl1, col_tmpl2 = st.columns([2, 1])

        with col_tmpl1:
            # Список существующих шаблонов
            st.markdown("### 📋 Существующие шаблоны")
            if st.session_state.structure_templates:
                for template_name in st.session_state.structure_templates.keys():
                    st.write(f"📁 {template_name}")
            else:
                st.info("Нет сохранённых шаблонов")

            st.divider()

            # Создание нового шаблона
            st.markdown("### ➕ Создать новый шаблон")
            new_template_name = st.text_input("Название шаблона", placeholder="например: CT_Chest_Standard")
            schema_mode = st.radio(
                "Режим создания схемы",
                ["🧱 Конструктор", "🧾 JSON вручную"],
                horizontal=True,
                key="schema_mode",
            )

            def build_template_from_constructor(level_names: list[str], file_patterns: list[str]) -> dict:
                nested: dict | list[str] = file_patterns
                for level_name in reversed(level_names):
                    nested = {level_name: nested}
                return {"root": nested}

            if schema_mode == "🧱 Конструктор":
                st.markdown("#### Конструктор схемы")
                st.caption("Добавьте уровни структуры и шаблоны файлов без ручного редактирования JSON.")

                cols_builder = st.columns([1, 1, 2])
                with cols_builder[0]:
                    if st.button("➕ Уровень", key="builder_add_level"):
                        st.session_state.schema_builder_levels.append(
                            f"level_{len(st.session_state.schema_builder_levels) + 1}"
                        )
                with cols_builder[1]:
                    if st.button("➖ Уровень", key="builder_remove_level"):
                        if len(st.session_state.schema_builder_levels) > 1:
                            st.session_state.schema_builder_levels.pop()

                for idx, default_level_name in enumerate(st.session_state.schema_builder_levels):
                    st.session_state.schema_builder_levels[idx] = st.text_input(
                        f"Уровень {idx + 1}",
                        value=default_level_name,
                        key=f"builder_level_{idx}",
                        help="Например: patient, study, series",
                    )

                patterns_raw = st.text_input(
                    "Шаблоны файлов (через запятую)",
                    value=st.session_state.schema_builder_file_patterns,
                    key="builder_file_patterns",
                    help="Например: *.dcm, *.nii.gz",
                )
                st.session_state.schema_builder_file_patterns = patterns_raw
                file_patterns = [p.strip() for p in patterns_raw.split(",") if p.strip()]
                if not file_patterns:
                    file_patterns = ["*.dcm"]

                generated_template = build_template_from_constructor(
                    st.session_state.schema_builder_levels,
                    file_patterns,
                )
                import json

                template_structure = json.dumps(generated_template, indent=2, ensure_ascii=False)
                st.code(template_structure, language="json")
            else:
                template_structure = st.text_area(
                    "Структура шаблона (JSON формат)",
                    value="""{
  "root": {
    "patient_level": {
      "study_level": {
        "series_level": ["*.dcm"]
      }
    }
  }
}""",
                    height=200,
                    help="Опишите ожидаемую структуру в формате JSON",
                )

            if st.button("💾 Сохранить шаблон"):
                if new_template_name and template_structure:
                    try:
                        import json

                        parsed_template = json.loads(template_structure)
                        st.session_state.structure_templates[new_template_name] = parsed_template
                        st.success(f"Шаблон '{new_template_name}' сохранён!")
                    except json.JSONDecodeError as e:
                        st.error(f"Ошибка JSON: {e}")
                else:
                    st.warning("Заполните название и структуру шаблона")

        with col_tmpl2:
            # Валидация по шаблону
            st.markdown("### ✅ Валидация датасета")
            if st.session_state.structure_templates:
                selected_template = st.selectbox(
                    "Выберите шаблон", options=list(st.session_state.structure_templates.keys())
                )

                if st.button("🔍 Проверить соответствие"):
                    if st.session_state.current_report and st.session_state.selected_folder_path:
                        # Валидация структуры датасета по шаблону
                        template = st.session_state.structure_templates[selected_template]

                        def validate_structure_against_template(
                            actual_structure: dict, template: dict, path_prefix: str = ""
                        ) -> tuple[list[str], list[str]]:
                            """Сравнивает реальную структуру с шаблоном.

                            Возвращает списки соответствий и отклонений.
                            """
                            matches = []
                            deviations = []

                            def check_node(actual: dict | None, tmpl_node: dict | list, current_path: str):
                                if isinstance(tmpl_node, dict):
                                    if actual is None or not isinstance(actual, dict):
                                        deviations.append(f"❌ Ожидается папка: {current_path}")
                                        return

                                    for key, value in tmpl_node.items():
                                        new_path = f"{current_path}/{key}" if current_path else key

                                        # Поиск соответствия в реальной структуре
                                        if actual.get("children"):
                                            found_child = None
                                            for child in actual["children"]:
                                                if child.get("name") == key or child.get("type") == "dir":
                                                    found_child = child
                                                    break

                                            if found_child:
                                                check_node(found_child, value, new_path)
                                            else:
                                                deviations.append(f"❌ Не найдено: {new_path}")
                                        else:
                                            deviations.append(f"❌ Нет дочерних элементов в: {current_path}")

                                elif isinstance(tmpl_node, list):
                                    # Это уровень файлов (например, ["*.dcm"])
                                    if actual is None:
                                        deviations.append(f"❌ Ожидается файл/папка: {current_path}")
                                        return

                                    # Проверка наличия файлов по маске
                                    for pattern in tmpl_node:
                                        if pattern.startswith("*."):
                                            ext = pattern[1:]  # например ".dcm"
                                            # Проверяем, есть ли файлы с таким расширением
                                            if actual.get("children"):
                                                has_matching = any(
                                                    c.get("type") == "file"
                                                    and c.get("name", "").endswith(ext.replace("*.", "."))
                                                    for c in actual["children"]
                                                )
                                                if has_matching:
                                                    matches.append(f"✅ Найдены файлы {pattern} в {current_path}")
                                                else:
                                                    deviations.append(f"⚠️ Не найдены файлы {pattern} в {current_path}")
                                            else:
                                                deviations.append(f"❌ Нет файлов в: {current_path}")

                            check_node(actual_structure, template, path_prefix)
                            return matches, deviations

                        # Сканирование текущей структуры
                        with st.spinner("🔍 Проверка структуры..."):
                            actual_structure = scan_dataset_structure(
                                DatasetScanConfig(root_path=st.session_state.selected_folder_path, max_depth=5)
                            ).root

                            if actual_structure:
                                matches, deviations = validate_structure_against_template(actual_structure, template)

                                if deviations:
                                    st.error(f"Найдено отклонений: {len(deviations)}")
                                    for dev in deviations[:20]:
                                        st.text(dev)
                                    if len(deviations) > 20:
                                        st.warning(f"... и ещё {len(deviations) - 20} отклонений")
                                else:
                                    st.success("✅ Структура полностью соответствует шаблону!")

                                if matches:
                                    with st.expander(f"✅ Соответствия ({len(matches)})"):
                                        for match in matches:
                                            st.text(match)
                            else:
                                st.error("Не удалось просканировать структуру датасета")
                    else:
                        st.warning("Сначала запустите анализ датасета")
            else:
                st.info("Сначала создайте шаблон")

            st.divider()

            # Экспорт/импорт шаблонов
            st.markdown("### 📤 Экспорт/Импорт")
            if st.session_state.structure_templates:
                if st.button("📥 Скачать все шаблоны"):
                    import json

                    templates_json = json.dumps(st.session_state.structure_templates, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Скачать JSON",
                        data=templates_json,
                        file_name="structure_templates.json",
                        mime="application/json",
                    )

            uploaded_templates = st.file_uploader(
                "📤 Загрузить шаблоны", type=["json"], help="Загрузите ранее сохранённые шаблоны"
            )
            if uploaded_templates:
                try:
                    import json

                    loaded = json.load(uploaded_templates)
                    if isinstance(loaded, dict):
                        st.session_state.structure_templates.update(loaded)
                        st.success(f"Загружено шаблонов: {len(loaded)}")
                    else:
                        st.error("Неверный формат JSON")
                except Exception as e:
                    st.error(f"Ошибка загрузки: {e}")

    # Вкладка анонимизатора DICOM
    with tab7:
        st.subheader("🔐 Анонимизатор DICOM-данных")
        st.markdown(
            """
        Инструмент для безопасной анонимизации DICOM-файлов с сохранением целостности исследований.

        **Возможности:**
        - Удаление персональных данных пациентов (PatientName, PatientID, даты и т.д.)
        - Псевдоанонимизация с сохранением возможности связывания данных
        - Сохранение маппинга оригинальных и анонимизированных значений
        - Гибкая настройка уровня анонимизации

        **Уровни анонимизации:**
        - **Basic**: Базовая анонимизация по стандарту DICOM PS3.15
        - **Full**: Полная анонимизация всех идентифицирующих полей
        """
        )

        if st.session_state.selected_folder_path:
            from src.dcmmetatest.anonymizer import AnonymizationConfig, run_anonymization

            st.markdown("### Настройки анонимизации")

            col_anon1, col_anon2 = st.columns(2)

            with col_anon1:
                anon_level = st.selectbox(
                    "Уровень анонимизации", options=["basic", "full"], index=0, help="Выберите уровень анонимизации"
                )

                preserve_uids = st.checkbox(
                    "Сохранить UID для целостности",
                    value=True,
                    help="Сохраняет StudyInstanceUID и SeriesInstanceUID для связи файлов",
                )

                create_mapping = st.checkbox(
                    "Создать файл маппинга",
                    value=True,
                    help="Создаёт JSON файл с соответствием оригинальных и анонимизированных значений",
                )

            with col_anon2:
                output_dir = st.text_input(
                    "Выходная директория",
                    placeholder="/path/to/anonymized/output",
                    help="Путь для сохранения анонимизированных данных",
                )

                dry_run = st.checkbox(
                    "Тестовый режим (без записи)", value=False, help="Запуск без фактической записи файлов"
                )

            if st.button("🚀 Запустить анонимизацию", type="primary"):
                if not output_dir and not dry_run:
                    st.error("Укажите выходную директорию или включите тестовый режим")
                else:
                    config = AnonymizationConfig(
                        level=anon_level,
                        preserve_study_integrity=preserve_uids,
                        create_mapping_file=create_mapping,
                        output_dir=output_dir if output_dir else None,
                        dry_run=dry_run,
                    )

                    try:
                        with st.spinner("Анонимизация..."):
                            stats, mapping = run_anonymization(
                                st.session_state.selected_folder_path,
                                output_dir if output_dir else "/tmp/anonymized",
                                config,
                            )

                        st.success("✅ Анонимизация завершена!")

                        col_stat1, col_stat2, col_stat3 = st.columns(3)

                        with col_stat1:
                            st.metric("Всего файлов", stats.total_files)
                        with col_stat2:
                            st.metric("Обработано", stats.processed_files)
                        with col_stat3:
                            st.metric("Ошибок", stats.failed_files)

                        if stats.tags_modified:
                            st.markdown("### Изменённые теги:")
                            st.json(stats.tags_modified)

                        if create_mapping and not dry_run and mapping:
                            st.download_button(
                                label="📥 Скачать файл маппинга",
                                data=json.dumps(mapping, indent=2, ensure_ascii=False),
                                file_name="anonymization_mapping.json",
                                mime="application/json",
                            )

                    except Exception as e:
                        st.error(f"❌ Ошибка анонимизации: {e!s}")
        else:
            st.info("Сначала запустите анализ датасета")

    # Вкладка истории анализов
    with tab8:
        st.subheader("📜 История анализов")
        st.markdown(
            """
        Журнал всех выполненных анализов датасетов в текущей сессии.

        **Информация в истории:**
        - Время выполнения анализа
        - Путь к датасету
        - Количество исследований и файлов
        - Количество неанонимизированных пациентов
        - Использованная конфигурация
        """
        )

        if st.session_state.analysis_history:
            # Отображение истории в виде таблицы
            history_data = []
            for idx, entry in enumerate(st.session_state.analysis_history, 1):
                history_data.append(
                    {
                        "№": idx,
                        "Время": entry["timestamp"][:19].replace("T", " "),
                        "Путь": entry["path"],
                        "Исследований": entry["total_studies"],
                        "Файлов": entry["total_files"],
                        "Неанон. пациентов": entry["non_anon_count"],
                    }
                )

            import pandas as pd

            history_df = pd.DataFrame(history_data)

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
                height=400,
            )

            st.divider()

            # Детали выбранного анализа
            st.markdown("### 📋 Детали анализа")
            selected_idx = st.selectbox(
                "Выберите анализ из истории",
                options=list(range(1, len(st.session_state.analysis_history) + 1)),
                format_func=lambda x: f"Анализ #{x} - {st.session_state.analysis_history[x-1]['path']}",
            )

            if selected_idx:
                selected_entry = st.session_state.analysis_history[selected_idx - 1]

                col_det1, col_det2 = st.columns(2)

                with col_det1:
                    st.markdown("**Параметры:**")
                    st.write(f"- **Время:** {selected_entry['timestamp'][:19].replace('T', ' ')}")
                    st.write(f"- **Путь:** `{selected_entry['path']}`")
                    st.write(f"- **Исследований:** {selected_entry['total_studies']}")
                    st.write(f"- **Файлов:** {selected_entry['total_files']}")
                    st.write(f"- **Неанонимизировано:** {selected_entry['non_anon_count']} пациентов")

                with col_det2:
                    st.markdown("**Конфигурация:**")
                    config = selected_entry.get("config", {})
                    st.write(f"- Группировка: `{config.get('group_by', 'study')}`")
                    st.write(f"- Потоков: `{config.get('max_workers', 4)}`")
                    st.write(f"- Только размеченные: `{config.get('only_labeled', False)}`")
                    modality = config.get("modality_filter")
                    st.write(f"- Фильтр модальности: `{modality if modality else 'Все'}`")

            # Кнопка очистки истории
            st.divider()
            if st.button("🗑️ Очистить историю", type="secondary"):
                st.session_state.analysis_history = []
                st.success("История очищена")
                st.rerun()
        else:
            st.info("История пуста. Запустите анализ датасета.")

    # Вкладка бета-функционала: нормализация и сплит
    with tab9:
        st.subheader("🧪 Бета: Нормализация и разделение датасетов")
        st.markdown(
            """
        Экспериментальный раздел для нормализации структуры DICOM-датасетов и разделения на train/val/test.

        **Возможности:**
        - 📁 Нормализация структуры (patient_study_series)
        - 🔀 Разделение на train/val/test с настройкой пропорций
        - 🏷️ Извлечение информации о масках сегментации
        - 📖 Работа со словарями классов

        ⚠️ **Это бета-версия!** Функционал находится в разработке.
        """
        )

        if st.session_state.selected_folder_path:
            # Переключатель между режимами
            beta_mode = st.radio(
                "Режим работы",
                [
                    "Нормализация",
                    "Разделение (Split)",
                    "Анализ сегментаций",
                    "Словарь классов",
                    "Препроцессинг датасета",
                ],
                horizontal=True,
            )

            # Режим нормализации
            if beta_mode == "Нормализация":
                st.markdown("### 📁 Нормализация структуры датасета")

                # Выбор типа задачи
                task_type = st.selectbox(
                    "Тип задачи",
                    ["segmentation", "classification", "detection", "slice_classification"],
                    index=0,
                    help="Выберите тип задачи ML для подготовки датасета",
                )

                col_norm1, col_norm2 = st.columns(2)

                with col_norm1:
                    norm_structure = st.selectbox(
                        "Целевая структура",
                        ["patient_study_series", "flat"],
                        index=0,
                        help="patient_study_series: patient_X/study_Y/series_Z/, flat: все файлы в одной папке",
                    )

                    rename_files = st.checkbox("Переименовывать файлы", value=True)
                    file_pattern = st.text_input(
                        "Шаблон имени файла",
                        value="{patient}_{study}_{series}_{index}.dcm",
                        disabled=not rename_files,
                        help="Доступны переменные: {patient}, {study}, {series}, {index}",
                    )

                    extract_metadata = st.checkbox("Извлекать метаданные", value=True)

                with col_norm2:
                    process_segs = st.checkbox("Обрабатывать сегментации", value=True)
                    seg_output_dir = st.text_input(
                        "Папка для сегментаций", value="segmentations", disabled=not process_segs
                    )

                    # Опции для классификации
                    if task_type == "classification":
                        st.info("🏷️ Настройки для классификации")
                        class_source = st.selectbox("Источник лейблов", ["dicom_tags", "csv", "folder_name"], index=0)
                        dicom_tag = st.text_input(
                            "DICOM тег для лейбла", value="SeriesDescription", disabled=(class_source != "dicom_tags")
                        )

                    output_norm_dir = st.text_input(
                        "Выходная директория",
                        placeholder="/path/to/normalized/output",
                        help="Путь для сохранения нормализованного датасета",
                    )

                if st.button("🚀 Запустить нормализацию", type="primary"):
                    if not output_norm_dir:
                        st.error("Укажите выходную директорию")
                    else:
                        # Импорт enum для типа задачи
                        from src.dcmmetatest.normalizer import DatasetTaskType

                        config = NormalizationConfig(
                            task_type=DatasetTaskType(task_type),
                            target_structure=norm_structure,
                            rename_files=rename_files,
                            file_pattern=file_pattern,
                            extract_metadata=extract_metadata,
                            process_segmentations=process_segs,
                            segmentation_output_dir=seg_output_dir,
                        )

                        # Добавляем параметры для классификации
                        if task_type == "classification":
                            config.classification_source = class_source
                            config.dicom_tag_for_label = dicom_tag

                        try:
                            with st.spinner("Нормализация..."):
                                stats = run_normalize(
                                    NormalizeRequest(
                                        source_dir=st.session_state.selected_folder_path,
                                        output_dir=output_norm_dir,
                                        config=config,
                                    )
                                )

                            st.success("✅ Нормализация завершена!")

                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric("Пациентов", stats.total_patients)
                                st.metric("Исследований", stats.total_studies)
                            with col_s2:
                                st.metric("Файлов обработано", stats.processed_files)
                                st.metric("Ошибок", stats.failed_files)
                            with col_s3:
                                st.metric("Сегментаций найдено", stats.segmentations_found)

                            if stats.output_structure:
                                st.json(stats.output_structure)

                        except Exception as e:
                            st.error(f"❌ Ошибка нормализации: {e!s}")

            # Режим разделения
            elif beta_mode == "Разделение (Split)":
                st.markdown("### 🔀 Разделение на Train / Validation / Test")

                col_split1, col_split2 = st.columns(2)

                with col_split1:
                    st.markdown("**Пропорции разделения:**")
                    train_ratio = st.slider("Train", 0.1, 0.9, 0.7, 0.05)
                    val_ratio = st.slider("Validation", 0.05, 0.5, 0.15, 0.05)
                    test_ratio = round(1.0 - train_ratio - val_ratio, 2)
                    st.info(f"Test ratio: {test_ratio} (автоматически)")

                    seed = st.number_input("Random Seed", value=42, help="Для воспроизводимости")

                with col_split2:
                    st.markdown("**Настройки:**")
                    stratify_by = st.selectbox(
                        "Стратификация по",
                        ["patient", "study"],
                        index=0,
                        help="patient: всё исследование идёт в один сплит, study: можно разделить",
                    )

                    create_manifest = st.checkbox("Создать манифест", value=True)

                    output_split_dir = st.text_input(
                        "Выходная директория",
                        placeholder="/path/to/split/output",
                        help="Будут созданы папки: train/, val/, test/",
                    )

                if st.button("🚀 Запустить разделение", type="primary"):
                    if not output_split_dir:
                        st.error("Укажите выходную директорию")
                    elif test_ratio <= 0:
                        st.error("Некорректные пропорции: test ratio должен быть > 0")
                    else:
                        config = SplitConfig(
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=test_ratio,
                            stratify_by=stratify_by,
                            seed=seed,
                            create_manifest=create_manifest,
                        )

                        try:
                            with st.spinner("Разделение датасета..."):
                                stats = run_split(
                                    SplitRequest(
                                        source_dir=st.session_state.selected_folder_path,
                                        output_dir=output_split_dir,
                                        config=config,
                                    )
                                )

                            st.success("✅ Разделение завершено!")

                            col_sp1, col_sp2, col_sp3 = st.columns(3)
                            with col_sp1:
                                st.metric("Train", f"{stats.train_samples} образцов\n{stats.train_patients} пациентов")
                            with col_sp2:
                                st.metric("Validation", f"{stats.val_samples} образцов\n{stats.val_patients} пациентов")
                            with col_sp3:
                                st.metric("Test", f"{stats.test_samples} образцов\n{stats.test_patients} пациентов")

                            if stats.split_manifest:
                                with st.expander("📄 Посмотреть манифест"):
                                    st.json(stats.split_manifest)

                        except Exception as e:
                            st.error(f"❌ Ошибка разделения: {e!s}")

            # Режим анализа сегментаций
            elif beta_mode == "Анализ сегментаций":
                st.markdown("### 🏷️ Анализ масок сегментации")

                if st.button("🔍 Сканировать маски сегментации"):
                    try:
                        with st.spinner("Анализ..."):
                            seg_info = run_segmentation_analysis(st.session_state.selected_folder_path)

                        if seg_info["total_masks"] > 0:
                            st.success(f"Найдено масок: {seg_info['total_masks']}")

                            col_seg1, col_seg2 = st.columns(2)
                            with col_seg1:
                                st.metric("Уникальных классов", seg_info["total_classes"])
                            with col_seg2:
                                st.metric("Масок", seg_info["total_masks"])

                            if seg_info["unique_classes"]:
                                st.markdown("**Классы:**")
                                st.write(", ".join(seg_info["unique_classes"]))

                            if seg_info["masks"]:
                                with st.expander("📋 Детали по маскам"):
                                    for mask in seg_info["masks"]:
                                        st.markdown(f"**{mask['name']}**")
                                        st.code(f"Файл: {mask['file']}")
                                        if mask["classes"]:
                                            st.write("Классы:")
                                            for cls in mask["classes"]:
                                                st.write(f"- ID {cls['id']}: {cls['name']} ({cls['description']})")
                                        st.divider()
                        else:
                            st.info("Маски сегментации не найдены")

                    except Exception as e:
                        st.error(f"❌ Ошибка анализа: {e!s}")

            # Режим словаря классов
            elif beta_mode == "Словарь классов":
                st.markdown("### 📖 Словарь классов сегментации")

                dict_tab1, dict_tab2 = st.tabs(["Загрузка", "Создание/Редактирование"])

                with dict_tab1:
                    st.markdown("**Загрузить существующий словарь**")

                    uploaded_dict = st.file_uploader(
                        "JSON файл со словарём",
                        type=["json"],
                        help='Формат: {"classes": [{"id": 1, "name": "...", ...}]}',
                    )

                    if uploaded_dict:
                        try:
                            import json

                            data = json.load(uploaded_dict)
                            st.session_state.class_dictionary = data
                            st.success(f"Загружено классов: {len(data.get('classes', []))}")

                            if "classes" in data:
                                st.markdown("**Классы:**")
                                for cls in data["classes"]:
                                    st.write(
                                        f"- ID {cls.get('id')}: **{cls.get('name')}** - {cls.get('description', '')}"
                                    )
                        except Exception as e:
                            st.error(f"Ошибка загрузки: {e}")

                with dict_tab2:
                    st.markdown("**Создать новый словарь классов**")

                    if "class_dictionary" not in st.session_state:
                        st.session_state.class_dictionary = {"classes": []}

                    # Добавление класса
                    st.markdown("➕ Добавить класс")
                    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

                    with col_c1:
                        new_class_id = st.number_input("ID", min_value=1, value=1, key="new_cls_id")
                    with col_c2:
                        new_class_name = st.text_input("Название", value="", key="new_cls_name")
                    with col_c3:
                        new_class_desc = st.text_input("Описание", value="", key="new_cls_desc")
                    with col_c4:
                        new_class_color = st.color_picker("Цвет", "#FF0000", key="new_cls_color")

                    if st.button("Добавить класс"):
                        if new_class_name:
                            hex_color = new_class_color.lstrip("#")
                            rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

                            st.session_state.class_dictionary["classes"].append(
                                {
                                    "id": new_class_id,
                                    "name": new_class_name,
                                    "description": new_class_desc,
                                    "color": list(rgb_color),
                                }
                            )
                            st.success(f"Добавлен класс: {new_class_name}")
                            st.rerun()

                    # Отображение текущих классов
                    if st.session_state.class_dictionary.get("classes"):
                        st.markdown("**Текущие классы:**")
                        for idx, cls in enumerate(st.session_state.class_dictionary["classes"]):
                            with st.container():
                                col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns([1, 2, 3, 1, 1])
                                with col_d1:
                                    st.write(f"ID: {cls.get('id')}")
                                with col_d2:
                                    color_preview = f"#{''.join(f'{c:02x}' for c in cls.get('color', [0,0,0]))}"
                                    st.write(f"{cls.get('name')} `{color_preview}`")
                                with col_d3:
                                    st.caption(cls.get("description", ""))
                                with col_d4:
                                    if st.button("✏️", key=f"edit_cls_{idx}"):
                                        pass  # TODO: редактирование
                                with col_d5:
                                    if st.button("🗑️", key=f"del_cls_{idx}"):
                                        st.session_state.class_dictionary["classes"].pop(idx)
                                        st.rerun()
                                st.divider()

                    # Сохранение словаря
                    if st.session_state.class_dictionary.get("classes"):
                        dict_json = json.dumps(st.session_state.class_dictionary, indent=2, ensure_ascii=False)

                        st.download_button(
                            label="💾 Скачать словарь (JSON)",
                            data=dict_json,
                            file_name="class_dictionary.json",
                            mime="application/json",
                        )

            # Режим пакетного препроцессинга
            elif beta_mode == "Препроцессинг датасета":
                st.markdown("### ⚙️ Препроцессинг всего датасета")
                st.caption("Применяет preprocessing pipeline ко всем найденным DICOM-сериям.")

                col_pp1, col_pp2 = st.columns(2)
                with col_pp1:
                    output_dataset_preprocess_dir = st.text_input(
                        "Выходная директория",
                        placeholder="/path/to/preprocessed/output",
                        help="Структура серий будет сохранена относительно корня входного датасета",
                    )
                    preprocess_export_format = st.selectbox(
                        "Формат экспорта",
                        ["png", "jpg", "tiff", "nifti"],
                        index=0,
                    )
                    preprocess_norm_method = st.selectbox(
                        "Нормализация интенсивности",
                        ["minmax", "zscore", "sigmoid"],
                        index=0,
                    )

                with col_pp2:
                    preprocess_resample = st.checkbox("Ресемплинг до 1x1x1 мм", value=False)
                    preprocess_crop = st.checkbox("Кроппинг foreground", value=False)
                    preprocess_augment = st.checkbox("Аугментация (flip+noise)", value=False)
                    max_series_to_process = st.number_input(
                        "Лимит серий (0 = без лимита)",
                        min_value=0,
                        value=0,
                    )

                if st.button("🚀 Запустить препроцессинг датасета", type="primary"):
                    if not output_dataset_preprocess_dir:
                        st.error("Укажите выходную директорию")
                    else:
                        with st.spinner("Препроцессинг датасета..."):
                            pp_config = PreprocessingPipelineConfig(
                                normalization_method=preprocess_norm_method,
                                enable_resampling=preprocess_resample,
                                target_spacing=(1.0, 1.0, 1.0),
                                crop_nonzero=preprocess_crop,
                                crop_threshold=0.0,
                                crop_margin=1,
                                enable_augmentation=preprocess_augment,
                                augmentation=AugmentationConfig(
                                    flip_horizontal=True,
                                    add_gaussian_noise=preprocess_augment,
                                    random_seed=42,
                                ),
                                export_format=preprocess_export_format,
                            )
                            max_series = int(max_series_to_process) if int(max_series_to_process) > 0 else None
                            pp_summary = run_dataset_preprocessing(
                                DatasetPreprocessRequest(
                                    input_root=st.session_state.selected_folder_path,
                                    output_root=output_dataset_preprocess_dir,
                                    config=pp_config,
                                    max_series=max_series,
                                )
                            )

                        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                        with col_r1:
                            st.metric("Найдено серий", pp_summary["total_series_found"])
                        with col_r2:
                            st.metric("Успешно", pp_summary["series_processed"])
                        with col_r3:
                            st.metric("С ошибками", pp_summary["series_failed"])
                        with col_r4:
                            st.metric("Сохранено файлов", pp_summary["total_files_saved"])

                        if pp_summary["errors"]:
                            with st.expander("Ошибки препроцессинга"):
                                for err in pp_summary["errors"][:100]:
                                    st.text(err)

        else:
            st.info("Сначала запустите анализ датасета")

else:
    # Стартовая страница
    st.info("👈 Введите путь к датасету в боковой панели и нажмите 'Запустить анализ'")

    st.markdown(
        """
    ### Возможности приложения:

    - **📊 Статистика**: Общее количество исследований, файлов, пациентов
    - **🏷️ Разметка**: Процент размеченных данных, источники разметки
    - **🔍 Фильтрация**: По модальности, наличию разметки, другим параметрам
    - **📈 Визуализация**: Интерактивные графики распределения
    - **📋 Детализация**: Полная таблица всех исследований с поиском
    - **⚠️ Валидация**: Выявление ошибок и проблемных файлов
    - **📄 Экспорт**: Отчёты в форматах CSV, JSON, TXT

    ### Поддерживаемые форматы:
    - DICOM файлы (.dcm, без расширения, .dicom)
    - DICOM Segmentation Objects
    - DICOM Structured Reports
    - JSON аннотации в форматах COCO, DICOM-SR
    """
    )

    # Пример структуры
    st.markdown(
        """
    ### Ожидаемая структура датасета:

    ```
    dataset/
    ├── patient_001/
    │   ├── study_001/
    │   │   ├── series_001/
    │   │   │   ├── image_001.dcm
    │   │   │   └── ...
    │   │   └── labels.json
    │   └── study_002/
    │       └── ...
    └── patient_002/
        └── ...
    ```
    """
    )

# Футер
st.divider()
st.caption(
    "DICOM Analyzer v1.0.0 | Powered by Streamlit & pydicom | " "Для вопросов и предложений обращайтесь к разработчикам"
)
