#!/usr/bin/env python3
"""
Streamlit приложение для визуального анализа DICOM-датасетов.

Запуск:
    streamlit run app.py

Или через модуль:
    python -m streamlit run app.py
"""

import logging

import streamlit as st

from src.dcmmetatest.ui import (
    cached_run_analysis,
    convert_report_to_dataframe,
    create_label_source_bar_chart,
    create_modality_pie_chart,
    create_quality_metrics_cards,
    validate_folder_path,
)

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
st.markdown("""
Интерактивный инструмент для анализа DICOM-датасетов, проверки разметки и качества данных.
""")

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

# Основная область
if analyze_button:
    # Валидация пути
    is_valid, path_or_error = validate_folder_path(folder_path)

    if not is_valid:
        st.error(f"❌ Ошибка: {path_or_error}")
        st.stop()

    valid_path = path_or_error
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

    # Запуск анализа с индикатором прогресса
    with st.spinner("🔍 Анализ датасета... Это может занять некоторое время."):
        try:
            report = cached_run_analysis(valid_path, config_dict)
        except Exception as e:
            st.error(f"❌ Ошибка при анализе: {str(e)}")
            logger.exception("Analysis failed")
            st.stop()

    # Отображение результатов
    st.divider()

    # Карточки метрик
    metrics = create_quality_metrics_cards(report)

    col1, col2, col3, col4 = st.columns(4)

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

    st.divider()

    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Графики", "📋 Таблица данных", "⚠️ Проблемы", "📄 Экспорт"])

    with tab1:
        st.subheader("Визуализация данных")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_modality = create_modality_pie_chart(report)
            st.plotly_chart(fig_modality, use_container_width=True)

        with col_chart2:
            fig_labels = create_label_source_bar_chart(report)
            st.plotly_chart(fig_labels, use_container_width=True)

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

    with tab4:
        st.subheader("Экспорт отчёта")

        st.markdown("""
        Выберите формат для экспорта результатов анализа.
        """)

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
                # Простая сериализация
                import json
                from dataclasses import asdict

                report_dict = asdict(report)
                json_str = json.dumps(report_dict, indent=2, ensure_ascii=False, default=str)
                st.download_button(
                    label="Скачать JSON",
                    data=json_str,
                    file_name="dicom_analysis_report.json",
                    mime="application/json",
                )

        with col_exp3:
            if st.button("📋 TXT", use_container_width=True):
                # Форматированный текстовый отчёт
                lines = [
                    "=" * 60,
                    "DICOM ANALYSIS REPORT",
                    "=" * 60,
                    f"Total Studies: {metrics['total_studies']}",
                    f"Total Files: {metrics['total_files']}",
                    f"Unique Patients: {metrics['unique_patients']}",
                    f"Labeled: {metrics['labeled_percent']}%",
                    f"Non-Anonymized: {metrics['non_anon_percent']}%",
                    "",
                    "Modality Stats:",
                ]
                for mod, count in report.modality_stats.items():
                    lines.append(f"  {mod}: {count}")

                if report.errors:
                    lines.append("")
                    lines.append(f"Errors ({len(report.errors)}):")
                    for err in report.errors[:10]:
                        lines.append(f"  - {err}")

                txt_content = "\n".join(lines)
                st.download_button(
                    label="Скачать TXT",
                    data=txt_content,
                    file_name="dicom_analysis_report.txt",
                    mime="text/plain",
                )

else:
    # Стартовая страница
    st.info("👈 Введите путь к датасету в боковой панели и нажмите 'Запустить анализ'")

    st.markdown("""
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
    """)

    # Пример структуры
    st.markdown("""
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
    """)

# Футер
st.divider()
st.caption(
    "DICOM Analyzer v1.0.0 | Powered by Streamlit & pydicom | " "Для вопросов и предложений обращайтесь к разработчикам"
)
