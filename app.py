#!/usr/bin/env python3
"""
Streamlit приложение для визуального анализа DICOM-датасетов.

Запуск:
    streamlit run app.py

Или через модуль:
    python -m streamlit run app.py
"""

import logging
from pathlib import Path

import streamlit as st

from src.dcmmetatest.ui import (
    cached_run_analysis,
    convert_report_to_dataframe,
    create_age_distribution_chart,
    create_label_source_bar_chart,
    create_modality_pie_chart,
    create_quality_metrics_cards,
    create_study_date_timeline,
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
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []  # История анализов
if "structure_templates" not in st.session_state:
    st.session_state.structure_templates = {}  # Шаблоны структуры
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
                children.append(
                    {
                        "name": item.name,
                        "type": "file",
                        "size": item.stat().st_size,
                    }
                )

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
                "total_files": sum(len(s.files) for s in report.results),
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "📈 Графики",
            "📋 Таблица данных",
            "⚠️ Проблемы",
            "📄 Экспорт",
            "🗂️ Редактор структуры",
            "⚙️ Конфигуратор структуры",
            "🔐 Анонимизатор",
            "📜 История анализов",
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
                                    success, msg = rename_item(node.get("path", ""), new_name)
                                    if success:
                                        st.success(msg)
                                        st.session_state.dataset_structure = None  # Сброс кэша
                                        st.rerun()
                                    else:
                                        st.error(msg)

                            # Удаление (только пустые папки)
                            if st.button("🗑️", key=f"btn_delete_{node_key}", help="Удалить пустую папку"):
                                success, msg = delete_item(node.get("path", ""))
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

    # Вкладка конфигуратора структуры (редактор шаблонов)
    with tab6:
        st.subheader("⚙️ Конфигуратор структуры датасета")
        st.markdown("""
        Инструмент для создания и управления шаблонами ожидаемой структуры DICOM-датасета.

        **Возможности:**
        - Создание шаблонов структуры для разных типов исследований
        - Валидация реального датасета по шаблону
        - Сохранение и загрузка шаблонов в JSON
        - Автоматическое выявление отклонений от шаблона
        """)

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
                                st.session_state.selected_folder_path, max_depth=5
                            )

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
        st.markdown("""
        Инструмент для безопасной анонимизации DICOM-файлов с сохранением целостности исследований.

        **Возможности:**
        - Удаление персональных данных пациентов (PatientName, PatientID, даты и т.д.)
        - Псевдоанонимизация с сохранением возможности связывания данных
        - Сохранение маппинга оригинальных и анонимизированных значений
        - Гибкая настройка уровня анонимизации

        **Уровни анонимизации:**
        - **Basic**: Базовая анонимизация по стандарту DICOM PS3.15
        - **Full**: Полная анонимизация всех идентифицирующих полей
        """)

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
        st.markdown("""
        Журнал всех выполненных анализов датасетов в текущей сессии.

        **Информация в истории:**
        - Время выполнения анализа
        - Путь к датасету
        - Количество исследований и файлов
        - Количество неанонимизированных пациентов
        - Использованная конфигурация
        """)

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
