#!/usr/bin/env python3
"""
Streamlit приложение для визуального анализа DICOM-датасетов.

Запуск:
    streamlit run app.py

Или через модуль:
    python -m streamlit run app.py
"""

import logging
import os
import shutil
from pathlib import Path

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

# Функция для сканирования структуры датасета
def scan_dataset_structure(path: str, max_depth: int = 3) -> dict:
    """Сканирует структуру датасета и возвращает её в виде дерева."""
    def scan_dir(dir_path: Path, current_depth: int) -> dict | None:
        if current_depth > max_depth:
            return None
        
        try:
            items = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return {"name": dir_path.name, "type": "dir", "error": "Нет доступа", "children": []}
        
        children = []
        for item in items[:50]:  # Ограничиваем количество элементов для производительности
            if item.is_dir():
                child = scan_dir(item, current_depth + 1)
                if child:
                    children.append(child)
            else:
                children.append({
                    "name": item.name,
                    "type": "file",
                    "size": item.stat().st_size,
                })
        
        return {
            "name": dir_path.name,
            "type": "dir",
            "path": str(dir_path),
            "children": children,
            "total_items": len(items),
        }
    
    root_path = Path(path)
    if not root_path.exists():
        return None
    
    return scan_dir(root_path, 0)

# Функция для удаления файла/папки
def delete_item(item_path: str) -> tuple[bool, str]:
    """Удаляет файл или пустую папку."""
    path = Path(item_path)
    if not path.exists():
        return False, "Файл/папка не существует"
    
    try:
        if path.is_file():
            path.unlink()
            return True, f"Файл удалён: {item_path}"
        elif path.is_dir():
            if any(path.iterdir()):
                return False, "Папка не пуста. Удалите сначала содержимое."
            path.rmdir()
            return True, f"Папка удалена: {item_path}"
    except Exception as e:
        return False, f"Ошибка удаления: {e!s}"
    
    return False, "Неизвестная ошибка"

# Функция для переименования файла/папки
def rename_item(old_path: str, new_name: str) -> tuple[bool, str]:
    """Переименовывает файл или папку."""
    old = Path(old_path)
    if not old.exists():
        return False, "Файл/папка не существует"
    
    new = old.parent / new_name
    if new.exists():
        return False, "Файл/папка с таким именем уже существует"
    
    try:
        old.rename(new)
        return True, f"Переименовано в: {new_name}"
    except Exception as e:
        return False, f"Ошибка переименования: {e!s}"

# Зона прогресс бара (отдельная выделенная область)
progress_container = st.container()

# Основная область
if analyze_button:
    # Валидация пути
    is_valid, path_or_error = validate_folder_path(folder_path)

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
            import time
            
            analysis_complete = threading.Event()
            analysis_result = {}
            analysis_error = {}
            
            def run_analysis_thread():
                try:
                    result = cached_run_analysis(valid_path, config_dict)
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
            progress_bar.progress(100)
            progress_text.text("Анализ завершён!")
            progress_status.success("✅ Анализ успешно завершён")
            
        except Exception as e:
            st.error(f"❌ Ошибка при анализе: {str(e)}")
            logger.exception("Analysis failed")
            progress_status.error(f"❌ Ошибка: {str(e)}")
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

    # Вкладки с добавлением редактора структуры
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Графики", "📋 Таблица данных", "⚠️ Проблемы", "📄 Экспорт", "🗂️ Редактор структуры"])

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

    # Вкладка редактора структуры датасета
    with tab5:
        st.subheader("🗂️ Редактор структуры датасета")
        st.markdown("""
        Инструмент для просмотра и редактирования структуры вашего DICOM-датасета.
        
        **Возможности:**
        - Просмотр древовидной структуры
        - Переименование файлов и папок
        - Удаление пустых папок и файлов
        """)
        
        if st.session_state.selected_folder_path:
            # Кнопка обновления структуры
            if st.button("🔄 Обновить структуру", key="refresh_structure"):
                st.session_state.dataset_structure = scan_dataset_structure(st.session_state.selected_folder_path)
                st.rerun()
            
            # Сканирование структуры если ещё не сделано
            if st.session_state.dataset_structure is None:
                with st.spinner("Сканирование структуры..."):
                    st.session_state.dataset_structure = scan_dataset_structure(st.session_state.selected_folder_path)
            
            if st.session_state.dataset_structure:
                # Функция для рекурсивного отображения дерева
                def render_tree(node, level=0, parent_key=""):
                    indent = "  " * level
                    node_key = f"{parent_key}/{node['name']}" if parent_key else node['name']
                    
                    if node['type'] == 'dir':
                        # Отображение папки
                        col_icon, col_name, col_actions = st.columns([0.1, 0.7, 0.2])
                        
                        with col_icon:
                            st.write("📁")
                        
                        with col_name:
                            st.markdown(f"**{node['name']}**")
                            if 'total_items' in node:
                                st.caption(f"Элементов: {node['total_items']}")
                        
                        with col_actions:
                            # Переименование
                            new_name = st.text_input(
                                "Переименовать",
                                value=node['name'],
                                key=f"rename_{node_key}",
                                placeholder="Новое имя",
                                label_visibility="collapsed"
                            )
                            if st.button("✏️", key=f"btn_rename_{node_key}", help="Переименовать"):
                                if new_name and new_name != node['name']:
                                    success, msg = rename_item(node.get('path', ''), new_name)
                                    if success:
                                        st.success(msg)
                                        st.session_state.dataset_structure = None  # Сброс кэша
                                        st.rerun()
                                    else:
                                        st.error(msg)
                            
                            # Удаление (только пустые папки)
                            if st.button("🗑️", key=f"btn_delete_{node_key}", help="Удалить пустую папку"):
                                success, msg = delete_item(node.get('path', ''))
                                if success:
                                    st.success(msg)
                                    st.session_state.dataset_structure = None  # Сброс кэша
                                    st.rerun()
                                else:
                                    st.error(msg)
                        
                        # Рекурсивный рендеринг дочерних элементов
                        if 'children' in node and node['children']:
                            with st.expander(f"Открыть папку {node['name']}", expanded=(level < 1)):
                                for child in node['children']:
                                    render_tree(child, level + 1, node_key)
                    
                    else:
                        # Отображение файла
                        col_icon, col_name, col_size, col_actions = st.columns([0.1, 0.6, 0.15, 0.15])
                        
                        with col_icon:
                            st.write("📄")
                        
                        with col_name:
                            st.write(node['name'])
                        
                        with col_size:
                            size_kb = node.get('size', 0) / 1024
                            if size_kb < 1024:
                                st.caption(f"{size_kb:.1f} KB")
                            else:
                                st.caption(f"{size_kb/1024:.1f} MB")
                        
                        with col_actions:
                            # Переименование файла
                            new_name = st.text_input(
                                "Имя",
                                value=node['name'],
                                key=f"rename_{node_key}",
                                placeholder="Новое имя",
                                label_visibility="collapsed"
                            )
                            if st.button("✏️", key=f"btn_rename_{node_key}", help="Переименовать"):
                                if new_name and new_name != node['name']:
                                    # Для файлов нужен путь
                                    file_path = str(Path(st.session_state.selected_folder_path) / node_key)
                                    success, msg = rename_item(file_path, new_name)
                                    if success:
                                        st.success(msg)
                                        st.session_state.dataset_structure = None  # Сброс кэша
                                        st.rerun()
                                    else:
                                        st.error(msg)
                            
                            # Удаление файла
                            if st.button("🗑️", key=f"btn_delete_{node_key}", help="Удалить файл"):
                                file_path = str(Path(st.session_state.selected_folder_path) / node_key)
                                success, msg = delete_item(file_path)
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
