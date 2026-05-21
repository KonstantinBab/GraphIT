# parse_class_p.py — Парсинг PDF через Ollama API (requests, без библиотеки ollama)
#
# Работает с любой vision-моделью: devstral, qwen3-vl, и т.д.
# Прямые HTTP запросы к Ollama REST API — полный контроль.

import re
import json
import base64
import requests
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from typing import List, Optional
import cv2
from PIL import Image
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class ConstructionBillOCR:
    OCR_PROMPT = """
        Ты OCR. Перепиши таблицу с изображения в markdown.

        Правила:
        1. НЕ объединяй строки. НЕ изменяй текст. НЕ пропускай строки. НЕ добавляй от себя.
        2. Каждая видимая строка на изображении = отдельная строка в таблице.
        3. Штампы и подписи внизу страницы не переписывай.

        Выделение заголовков разделов:
        Оберни текст в колонке Наименование в **жирный**, если это заголовок раздела.

        Паттерн заголовка:
        1. Цифра (1-3 знака) + точка/скобка/пробел + Текст с Заглавной Буквы
            Примеры: "1 Металлоконструкции", "2. Бурение", "12) Фундаменты"
        2. ИЛИ текст полностью ЗАГЛАВНЫМИ буквами (5+ символов)
            Примеры: "ЗЕМЛЯНЫЕ РАБОТЫ", "МОНОЛИТНЫЙ БЕТОН"

        Что НЕ выделять (обычные позиции):
        - Подпункты с точкой: "1.1 Монтаж", "2.5 Бетон"
        - Размеры и единицы: "100мм труба", "50кг мешок"
        - Технические марки: "В25 бетон", "А500С арматура"

        Примеры вывода:
        | 1 | **1 Металлоконструкции** | | |
        | 1.1 | Монтаж колонн | шт | 10 |
        | 2 | **ЗЕМЛЯНЫЕ РАБОТЫ** | | |
        | 2.1 | Разработка грунта | м3 | 100 |

        Формат таблицы:
        | № п/п | Наименование | Ед. изм. | Кол-во |
        |-------|--------------|----------|--------|
        """.strip()

    OCR_PROMPT_STAMP = """
    Ты OCR. Перепиши ТЕКСТ ИЗ НИЖНЕЙ ЧАСТИ ИЗОБРАЖЕНИЯ без изменений.

    КРИТИЧЕСКИ ВАЖНО:
    1. НЕ объединяй строки. НЕ изменяй текст. НЕ пропускай строки. НЕ добавляй от себя.
    2. Обрабатывай ТОЛЬКО НИЖНЮЮ ЧАСТЬ изображения (штамп, реквизиты).
    3. Ищи текст в правом нижнем углу.
    4. Выводи строки в том же порядке, как они расположены на изображении.
    5. Не используй markdown, не добавляй |, не делай таблицы.

    Пример вывода (точно так):
    0092.049.Р.13/1.0004.УКПГ.045.3856.1226-КМ.ВР
    Ведомость объемов строительных и монтажных работ
    Стадия   Лист   Листов
    Р        2      7
    """.strip()

    def __init__(
            self,
            pdf_path: str,
            output_xlsx: Optional[str] = None,
            xlsx_lookup_path: Optional[str] = "Копия !ВСЕ_ЧТО_ЕСТЬ2.xlsx",
            poppler_path: Optional[str] = None,
            ollama_url: str = "http://localhost:11434",
            ollama_model: str = "qwen3.5:27b",
            dpi: int = 400,
            preprocess_images: bool = True,
            ocr_prompt: Optional[str] = None,
            progress_callback=None,
            max_parallel_ocr: int = 1,
            fast_preprocess: bool = True,
            num_ctx: int = 4096,
            num_predict: int = 4096,
            # Обратная совместимость
            model_name: Optional[str] = None,
            vllm_url: Optional[str] = None,
            max_tokens: Optional[int] = None,
    ):
        self.main_file_type = ""
        self.vikr_name = ""
        self.pdf_path = pdf_path
        self.output_xlsx = output_xlsx
        self.xlsx_lookup_path = xlsx_lookup_path
        self.poppler_path = poppler_path
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = model_name or ollama_model
        self.dpi = dpi
        self.preprocess_images = preprocess_images
        self.ocr_prompt = ocr_prompt or self.OCR_PROMPT
        self._df: Optional[pd.DataFrame] = None
        self.progress_callback = progress_callback or (lambda c, t, s: None)
        self.logger = print

        self.max_parallel_ocr = max(1, max_parallel_ocr)
        self.fast_preprocess = fast_preprocess
        self.num_ctx = num_ctx
        self.num_predict = max_tokens or num_predict
        self._log_lock = Lock()

    # ═══════════════════════════════════════════════════════════════
    # IMAGE PREPROCESSING
    # ═══════════════════════════════════════════════════════════════

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        if not self.preprocess_images:
            return img

        arr = np.array(img)
        if len(arr.shape) == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        if not self.fast_preprocess:
            gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced)

    def _image_to_base64(self, img: Image.Image) -> str:
        processed = self._preprocess_image(img)
        buf = BytesIO()
        processed.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ═══════════════════════════════════════════════════════════════
    # OCR — через requests к Ollama REST API
    # ═══════════════════════════════════════════════════════════════

    def _ocr_from_image(self, img: Image.Image, prompt: str = None) -> str:
        """Отправляет изображение в Ollama через /api/chat и возвращает raw текст."""
        img_b64 = self._image_to_base64(img)

        # Используем переданный промпт или основной
        used_prompt = prompt if prompt is not None else self.ocr_prompt

        payload = {
            "model": self.ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": used_prompt,
                    "images": [img_b64],
                }
            ],
            "options": {
                "temperature": 0.0,
                "top_p": 0.5,
                "top_k": 10,
                "num_ctx": self.num_ctx,
                "num_predict": min(self.num_predict, 1500),  # Ограничение
            },
            "stream": False,
            "think": False,
        }

        # Отправляем запрос
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()

        data = resp.json()

        # DEBUG: логируем полный ответ (ключи)
        with self._log_lock:
            self.logger(f"  🔧 Ollama response keys: {list(data.keys())}")
            if "message" in data:
                msg = data["message"]
                self.logger(
                    f"  🔧 message type: {type(msg)}, keys: {list(msg.keys()) if isinstance(msg, dict) else 'N/A'}")
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    self.logger(f"  🔧 content length: {len(content)}, first 1000: {repr(content[:1000])}")
                    # 🔥 Печать последних 500 символов
                    # self.logger(f"  🔍 content last 500: {repr(content[-500:])}")

        # Извлекаем текст
        raw = ""
        msg = data.get("message", {})
        if isinstance(msg, dict):
            raw = msg.get("content", "")
            if not raw and msg.get("thinking"):
                raw = msg["thinking"]
                with self._log_lock:
                    self.logger(f"  🔧 Взяли текст из 'thinking' ({len(raw)} символов)")
        elif isinstance(msg, str):
            raw = msg
        elif "response" in data:
            raw = data["response"]

        raw = raw.strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

        table_lines = [l for l in raw.split('\n') if l.strip().startswith('|') and '---' not in l]
        with self._log_lock:
            self.logger(f"  📊 Таблица: {len(table_lines)} строк данных (из {len(raw)} символов)")
            if len(table_lines) > 40:
                self.logger(f"  ⚠️ ВНИМАНИЕ: {len(table_lines)} строк — возможно, модель галлюцинирует!")

        lines = raw.split("\n")
        table_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("|"):
                table_start = i
                break

        if table_start > 0:
            raw = "\n".join(lines[table_start:]).strip()
            with self._log_lock:
                self.logger(f"  🔧 Обрезали рассуждения, таблица с строки {table_start}")

        if "<table" in raw.lower():
            html_start = raw.lower().find("<table")
            if html_start > 0:
                raw = raw[html_start:].strip()

        raw = re.sub(r'^```(?:markdown)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        with self._log_lock:
            self.logger(f"🔍 OCR raw (last 500 chars): {repr(raw[-500:])}")

        return raw

    def _extract_file_type(self, raw_text: str) -> str:
        """
        Извлекает шифр из сырого текста (например, 163.083.Р.0/0.1264-КМ1.ВР).
        Паттерн: 4 цифры.3 цифры. (далее любые символы).
        Ищет по всему тексту (включая штамп внизу).
        """
        lines = raw_text.strip().splitlines()

        # 🔥 Ищем по всем строкам (не только последние 10)
        for line in lines:
            line = line.strip()

            # 🔍 Паттерн: 4 цифры.3 цифры. (далее любые символы)
            # Пример: 163.083.Р.0/0.1264-КМ1.ВР
            match = re.search(r'\b\d{4}\.\d{3}[^\s\n\r]*\b', line)
            if match:
                return match.group(0)

        return ""

    def _extract_vikr_name_from_raw(self, raw_text: str) -> str:
        """
        Извлекает "Наименование ВиКР" из сырого текста первой страницы:
        - Ищет первую строку с шифром (например, '0092.049.Р.13/1.0004.УКПГ.045.3856.1226-КМ.ВР')
        - Затем ищет первую строку с **жирным** текстом после неё, но до 'Примечание:'
        - Возвращает содержимое внутри **...**
        """
        lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]

        if not lines:
            return ""

        # 🔍 Шаг 1: Найдём индекс строки с шифром (по паттерну)
        vikr_start_idx = -1
        for i, line in enumerate(lines):
            if re.search(r'\b\d{4}\.\d{3}[^\s\n\r]*\b', line):
                vikr_start_idx = i
                break

        if vikr_start_idx == -1:
            # Если шифр не найден, начинаем с начала
            vikr_start_idx = 0

        # 🔍 Шаг 2: Ищем **...** после шифра, но до 'Примечание:'
        for j in range(vikr_start_idx + 1, len(lines)):
            line = lines[j]
            if 'Примечание:' in line:
                break  # Останавливаемся перед 'Примечание:'

            # Ищем жирный текст: **текст**
            match = re.search(r'\*\*(.*?)\*\*', line)
            if match:
                # Берём первый найденный — это и есть "Наименование ВиКР"
                return match.group(1).strip()

        return ""

    def _detect_markdown_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        df['_is_section'] = False
        df['_header'] = ''
        df['_subheader'] = ''

        section_count = 0
        header_count = 0
        subheader_count = 0

        for idx in df.index:
            # 🔥 Очистка: берём текст, убираем лишние пробелы и кавычки
            name = str(df.at[idx, 'Наименование']).strip().strip('"').strip("'")

            # ═══════════════════════════════════════════════════════════════
            # ПРОВЕРКА: Текст в **двойных звёздочках**
            # ═══════════════════════════════════════════════════════════════
            if name.startswith('**') and name.endswith('**'):
                clean_name = name[2:-2].strip()

                # 🔹 Заголовок: начинается с цифры (например, "1 Металлоконструкции")
                if re.match(r'^\d', clean_name):
                    df.at[idx, '_is_section'] = True
                    df.at[idx, '_header'] = clean_name
                    df.at[idx, '_subheader'] = ''
                    df.at[idx, 'Наименование'] = clean_name
                    header_count += 1
                    section_count += 1
                    continue

                # 🔹 Подзаголовок: НЕ начинается с цифры (например, "Земляные работы")
                else:
                    df.at[idx, '_is_section'] = True
                    df.at[idx, '_subheader'] = clean_name
                    df.at[idx, '_header'] = ''
                    df.at[idx, 'Наименование'] = clean_name
                    subheader_count += 1
                    section_count += 1
                    # Логгер можно оставить, но поменять текст
                    with self._log_lock:
                        self.logger(f"  🔧 Обнаружен подзаголовок: {clean_name[:50]}...")
                    continue

        self.logger(f"🔖 Найдено разделов: {section_count} (заголовки: {header_count}, подзаголовки: {subheader_count})")
        return df

    def _extract_file_type_from_raw_stamp(self, raw_stamp_text: str) -> str:
        """
        Извлекает шифр из сырого текста, полученного по промпту штампа.
        Ищет в строках, содержащих 'Шифр' или похожие паттерны, или просто в любой строке.
        """
        if not raw_stamp_text:
            return ""

        lines = raw_stamp_text.strip().splitlines()

        for line in lines:
            line = line.strip()
            # 🔍 Паттерн: 4 цифры.3 цифры. (далее любые символы) - как в предыдущем методе
            match = re.search(r'\b\d{4}\.\d{3}[^\s\n\r]*\b', line)
            if match:
                return match.group(0)

        return ""

    def _ocr_page(self, page: Image.Image, page_number: int = 1) -> pd.DataFrame:
        # --- ПЕРВЫЙ ПРОХОД: Основная таблица ---
        raw_main = self._ocr_from_image(page, prompt=self.ocr_prompt)

        if not raw_main:
            raise ValueError("Empty response from Ollama on main table")

        df_main = self._parse_ocr_output(raw_main)
        # df_main = self._detect_markdown_sections(df_main)

        for col in ['_is_section', '_header', '_subheader']:
            if col not in df_main.columns:
                if col == '_is_section':
                    df_main[col] = False
                else:
                    df_main[col] = ''

        # --- ВТОРОЙ ПРОХОД: Штамп (только для первой страницы) ---
        file_type = ""
        if page_number == 1:  # 🔥 Условие: только первая страница
            raw_stamp = self._ocr_from_image(page, prompt=self.OCR_PROMPT_STAMP)
            # 🔥 Извлекаем шифр из вывода штампа
            file_type = self._extract_file_type(raw_main)
            if file_type:
                self.logger(f"  🏷️ Найден шифр (из основного текста): {file_type}")

            # 🔥 Извлекаем "Наименование ВиКР" из основного raw_main (не из штампа!)
            vikr_name = self._extract_vikr_name_from_raw(raw_main)
            if vikr_name:
                self.vikr_name = vikr_name
                self.logger(f"  🏷️ Найдено Наименование ВиКР: {vikr_name}")

        # 🔥 Добавляем шифр как атрибут всей страницы
        # df_main['_page_file_type'] = file_type

        return df_main

    def _extract_project_number(self, full_code: str) -> str:
        """
        Извлекает число между двумя точками перед последним тире из строки шифра.
        Пример: '0092.049.Р.13/1.0004.УКПГ.045.3856.1226-КМ.ВР' -> '3856'
        """
        if not full_code:
            return ""

        # Найдём индекс последнего тире
        last_dash_index = full_code.rfind('-')

        if last_dash_index == -1:
            # Если тире нет, вернём пустую строку
            return ""

        # Возьмём часть строки до последнего тире
        part_before_dash = full_code[:last_dash_index]

        # Найдём индекс последней точки в этой части
        last_dot_index = part_before_dash.rfind('.')

        if last_dot_index == -1:
            # Если точки перед тире нет, вернём пустую строку
            return ""

        # Найдём индекс точки перед последней точкой
        second_last_dot_index = part_before_dash.rfind('.', 0, last_dot_index)

        if second_last_dot_index == -1:
            # Если только одна точка перед тире, вернём пустую строку
            return ""

        # Извлечём подстроку между этими двумя точками
        number_between_dots = part_before_dash[second_last_dot_index + 1:last_dot_index]

        # Проверим, что это число (содержит только цифры)
        if number_between_dots.isdigit():
            return number_between_dots
        else:
            # Если между точками не число, вернём пустую строку
            return ""

    def _fill_file_type(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """
        Заполняет столбцы _file_type и _project_number для строк, где есть _header или _subheader.
        """
        if df.empty or not file_type:
            return df

        df = df.copy()

        # Извлекаем номер проекта из шифра
        project_number = self._extract_project_number(file_type)

        # Заполняем для строк с _header (разделы)
        header_mask = df['_header'].notna() & (df['_header'] != '')
        df.loc[header_mask, '_file_type'] = file_type
        df.loc[header_mask, '_project_number'] = project_number

        # Заполняем для строк с _subheader (подразделы), если у них есть родительский _header
        subheader_mask = df['_subheader'].notna() & (df['_subheader'] != '') & (df['_header'].notna()) & (
                df['_header'] != '')
        df.loc[subheader_mask, '_file_type'] = file_type
        df.loc[subheader_mask, '_project_number'] = project_number

        return df

    # ═══════════════════════════════════════════════════════════════
    # PARSING — универсальный (markdown, HTML, plain text)
    # ═══════════════════════════════════════════════════════════════

    def _parse_ocr_output(self, text: str) -> pd.DataFrame:
        lines = text.strip().splitlines()

        # HTML?
        if "<table" in text.lower() or "<tr" in text.lower():
            return self._parse_html_table(text)

        # Markdown?
        has_pipes = sum(1 for l in lines if "|" in l) > len(lines) * 0.3
        if has_pipes:
            return self._parse_markdown_lines(lines)

        # Plain text
        return self._parse_plain_text(lines)

    def _parse_html_table(self, html: str) -> pd.DataFrame:
        dfs = pd.read_html(StringIO(html))
        if not dfs:
            raise ValueError("No HTML tables found")
        df = dfs[0].fillna("").astype(str)
        df.columns = [str(c).strip() for c in df.columns]
        return self._normalize_columns(df)

    def _parse_markdown_lines(self, lines: list) -> pd.DataFrame:
        table_lines = []
        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                continue
            if not line.endswith("|"):
                line += "|"
            inner = re.sub(r"^\||\|$", "", line).strip()
            if re.match(r"^[\s\-:|]+$", inner):  # Пропускаем строку с --- | --- |
                continue
            table_lines.append(line)

        if len(table_lines) < 2:
            raise ValueError(f"Too few markdown lines ({len(table_lines)})")

        # 🔥 Извлекаем названия колонок из ПЕРВОЙ строки
        header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
        n_cols = len(header_cells)

        rows = []
        for line in table_lines[1:]:  # Пропускаем заголовок
            # 🔥 Разбиваем строку данных
            cells = [c.strip() for c in line.strip("|").split("|")]
            # 🔥 Корректируем длину
            if len(cells) > n_cols:
                cells = cells[:n_cols]
            while len(cells) < n_cols:
                cells.append("")
            rows.append(cells)

        # 🔥 Создаём DataFrame с ФАКТИЧЕСКИМИ названиями колонок
        df = pd.DataFrame(rows, columns=header_cells)
        # 🔥 Нормализуем (приведём к стандартным: № п/п, Наименование, Ед. изм., Кол-во)
        return self._normalize_columns(df)

    def _parse_plain_text(self, lines: list) -> pd.DataFrame:
        if not lines:
            raise ValueError("Empty OCR output")

        data_lines = lines[1:] if lines else []
        rows = []
        num_pattern = re.compile(r'^(\d+[\d.]*)\s+(.+)')
        unit_qty_pattern = re.compile(
            r'^(.*?)\s+(Т|т|м|М|м2|м3|мм|шт|кг|кГ|л|км|исп\.?|компл\.?|п\.м\.?)\s+([\d.,]+)\s*$'
        )

        for line in data_lines:
            line = line.strip()
            if not line:
                continue

            m_num = num_pattern.match(line)
            if m_num:
                num = m_num.group(1)
                rest = m_num.group(2).strip()
                m_uq = unit_qty_pattern.match(rest)
                if m_uq:
                    rows.append({"№ п/п": num, "Наименование": m_uq.group(1).strip(),
                                 "Ед. изм.": m_uq.group(2).strip(), "Кол-во": m_uq.group(3).strip()})
                else:
                    rows.append({"№ п/п": num, "Наименование": rest, "Ед. изм.": "", "Кол-во": ""})
            else:
                rows.append({"№ п/п": "", "Наименование": line, "Ед. изм.": "", "Кол-во": ""})

        if not rows:
            raise ValueError("No data rows parsed")

        df = pd.DataFrame(rows)
        df = df[~df.apply(lambda row: all(str(v).strip() == "" for v in row), axis=1)].reset_index(drop=True)
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Удаляем дубликаты колонок (OCR может вернуть "№ п/п" и "№ в ЛСР" и т.д.)
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

        col_mapping = self._detect_columns(df.columns.tolist())
        rename_map = {}
        used_targets = set()  # Защита от дублей при rename
        for std_name, real_name in col_mapping.items():
            if real_name and real_name in df.columns and std_name not in used_targets:
                rename_map[real_name] = std_name
                used_targets.add(std_name)
        df = df.rename(columns=rename_map)

        # Если после rename остались дубликаты — оставляем первый
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

        expected = ["№ п/п", "Наименование", "Ед. изм.", "Кол-во"]
        for col in expected:
            if col not in df.columns:
                df[col] = ""
        df = df[expected].copy()

        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        df = df[~df.apply(lambda row: all(str(v).strip() == "" for v in row), axis=1)].reset_index(drop=True)
        return df

    def _detect_columns(self, columns: List[str]) -> dict:
        mapping = {"№ п/п": None, "Наименование": None, "Ед. изм.": None, "Кол-во": None}
        cols_lower = {c: c.strip().lower() for c in columns}

        for real_name, lower_name in cols_lower.items():
            # "п/п" точнее, чем просто "№" — чтобы не захватить "№ в ЛСР"
            if any(x in lower_name for x in ["п/п", "строк"]):
                mapping["№ п/п"] = real_name
            elif "№" in lower_name and "лср" not in lower_name and mapping["№ п/п"] is None:
                mapping["№ п/п"] = real_name
            elif any(x in lower_name for x in ["наименование", "вид работ", "название"]):
                mapping["Наименование"] = real_name
            elif any(x in lower_name for x in ["ед.", "изм", "единиц"]):
                mapping["Ед. изм."] = real_name
            elif any(x in lower_name for x in ["кол", "колич", "объём", "объем"]):
                mapping["Кол-во"] = real_name

        return mapping

    # ═══════════════════════════════════════════════════════════════
    # POST-PROCESSING
    # ═══════════════════════════════════════════════════════════════

    def _merge_multiline_names(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        rows = df.to_dict("records")
        merged = []
        i = 0

        while i < len(rows):
            cur = rows[i]

            # 🔥 Если это раздел (помечен ранее) — не склеиваем
            if cur.get("_is_section", False):
                merged.append(cur.copy())
                i += 1
                continue

            cur_num = str(cur.get("№ п/п", "")).strip()
            cur_name = str(cur.get("Наименование", "")).strip()

            if cur_num:
                combined_name = cur_name
                j = i + 1
                while j < len(rows):
                    nxt = rows[j]
                    nxt_num = str(nxt.get("№ п/п", "")).strip()
                    nxt_name = str(nxt.get("Наименование", "")).strip()

                    if not nxt_num and nxt_name:
                        if re.match(r'^\d+\s+[А-ЯЁA-Z]', nxt_name):
                            break
                        combined_name += " " + nxt_name
                        j += 1
                    else:
                        break

                new_row = cur.copy()
                new_row["Наименование"] = combined_name.strip()
                for k in range(i + 1, j):
                    nxt = rows[k]
                    if not new_row.get("Ед. изм.") and nxt.get("Ед. изм."):
                        new_row["Ед. изм."] = nxt["Ед. изм."]
                    if not new_row.get("Кол-во") and nxt.get("Кол-во"):
                        new_row["Кол-во"] = nxt["Кол-во"]
                merged.append(new_row)
                i = j
            else:
                merged.append(cur)
                i += 1

        return pd.DataFrame(merged)

    def _merge_multiline_subheaders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Склеивает:
        1. Подзаголовки, разбитые на несколько строк
        2. Особый случай: раздел, разбитый на две строки (вторая — подраздел)
        """
        if df.empty:
            return df

        df = df.copy()
        rows = df.to_dict("records")
        merged = []
        i = 0

        while i < len(rows):
            cur = rows[i]
            cur_name = str(cur.get("Наименование", "")).strip()
            cur_subheader = cur.get("_subheader", "")
            cur_is_section = cur.get("_is_section", False)
            cur_num = str(cur.get("№ п/п", "")).strip()
            cur_unit = str(cur.get("Ед. изм.", "")).strip()
            cur_qty = str(cur.get("Кол-во", "")).strip()
            cur_has_data = bool(cur_unit) or bool(cur_qty)

            # 🔥 Особый случай: заголовок, за которым идёт строка-подраздел без разбивки
            if cur_is_section and cur.get("_header"):
                # Проверяем следующую строку
                if i + 1 < len(rows):
                    nxt = rows[i + 1]
                    nxt_name = str(nxt.get("Наименование", "")).strip()
                    nxt_num = str(nxt.get("№ п/п", "")).strip()
                    nxt_unit = str(nxt.get("Ед. изм.", "")).strip()
                    nxt_qty = str(nxt.get("Кол-во", "")).strip()
                    nxt_has_data = bool(nxt_unit) or bool(nxt_qty)

                    # Если следующая строка: пустой № п/п, нет данных, и похожа на подраздел
                    if not nxt_num and not nxt_has_data and nxt_name and len(nxt_name) > 5:
                        # Это подраздел, разбитый на вторую строку
                        cur["_subheader"] = nxt_name
                        # Пропускаем следующую строку
                        i += 2
                        merged.append(cur.copy())
                        continue

            # 🔥 Обычное слияние подзаголовков
            if cur_is_section and cur_subheader:
                combined_subheader = cur_name
                j = i + 1

                while j < len(rows):
                    nxt = rows[j]
                    nxt_name = str(nxt.get("Наименование", "")).strip()
                    nxt_unit = str(nxt.get("Ед. изм.", "")).strip()
                    nxt_qty = str(nxt.get("Кол-во", "")).strip()
                    nxt_has_data = bool(nxt_unit) or bool(nxt_qty)
                    nxt_is_section = nxt.get("_is_section", False)

                    if not nxt_is_section and not nxt_has_data and nxt_name:
                        if nxt_name[0].islower() or nxt_name.startswith((' ', '-', '–', '—')):
                            combined_subheader += " " + nxt_name
                            j += 1
                        else:
                            break
                    else:
                        break

                new_row = cur.copy()
                new_row["Наименование"] = combined_subheader.strip()
                new_row["_subheader"] = combined_subheader.strip()
                merged.append(new_row)
                i = j
            else:
                merged.append(cur.copy())
                i += 1

        result_df = pd.DataFrame(merged)
        self.logger(f"🔗 Склеено подзаголовков: {len(rows) - len(result_df)} строк")
        return result_df

    def _build_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        parents, levels, paths = [], [], []
        for val in df["№ п/п"]:
            s = str(val).strip()
            if not s:
                parents.append("")
                levels.append(0)
                paths.append("")
                continue

            clean = re.sub(r"[^\d.]", "", s)
            parts = [p for p in clean.split(".") if p.isdigit()]
            level = len(parts)
            parent = ".".join(parts[:-1]) if level > 1 else ""
            path = " > ".join(".".join(parts[:i + 1]) for i in range(len(parts))) if parts else ""
            parents.append(parent)
            levels.append(level)
            paths.append(path)

        df = df.copy()
        df["parent_id"] = parents
        df["level"] = levels
        df["path"] = paths
        return df

    def _generate_item_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует иерархическую нумерацию:
        - Заголовок раздела: № п/п = '' (пусто)
        - Подзаголовок: № п/п = номер_раздела.номер_подраздела (1.1, 1.2, 2.1...)
        - Обычная работа: № п/п = номер_раздела.номер_подраздела (1.1, 1.2, 2.1...)
        """
        if df.empty:
            return df

        df = df.copy()
        current_section_num = None  # Номер текущего раздела (из "Раздел 1.")
        current_subheader_num = 0  # Счётчик подразделов внутри раздела
        item_counter = 0  # Счётчик работ внутри подраздела

        for idx in df.index:
            is_section = df.at[idx, '_is_section']
            header_val = df.at[idx, '_header']
            subheader_val = df.at[idx, '_subheader']

            if is_section:
                if header_val:
                    # 🔹 Заголовок раздела — извлекаем номер из "Раздел 1."
                    match = re.search(r'Раздел\s+(\d+)', header_val, re.IGNORECASE)
                    if match:
                        current_section_num = match.group(1)
                        current_subheader_num = 0
                        item_counter = 0

                    df.at[idx, '№ п/п'] = ''  # Пусто для заголовка

                elif subheader_val:
                    # 🔹 Подзаголовок — увеличиваем счётчик подразделов
                    current_subheader_num += 1
                    item_counter = 0

                    if current_section_num:
                        df.at[idx, '№ п/п'] = f"{current_section_num}.{current_subheader_num}"
                    else:
                        df.at[idx, '№ п/п'] = str(current_subheader_num)
            else:
                # 🔹 Обычная строка работы
                if current_section_num and current_subheader_num > 0:
                    item_counter += 1
                    df.at[idx, '№ п/п'] = f"{current_section_num}.{current_subheader_num}.{item_counter}"
                elif current_section_num:
                    item_counter += 1
                    df.at[idx, '№ п/п'] = f"{current_section_num}.{item_counter}"
                # Если нет раздела — оставляем как есть или пусто

        return df

    def _propagate_headers_across_pages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Пропагирует заголовки и подзаголовки по всему документу.
        """
        if df.empty:
            return df

        df = df.copy()
        current_header = ''
        current_subheader = ''

        for idx in df.index:
            is_section = df.at[idx, '_is_section']
            header_val = df.at[idx, '_header']
            subheader_val = df.at[idx, '_subheader']

            if is_section:
                if header_val:  # Заголовок "Раздел цифра."
                    current_header = header_val
                    current_subheader = ''
                elif subheader_val:  # Подраздел
                    current_subheader = subheader_val
            else:
                # Обычная строка — пропагируем
                df.at[idx, '_header'] = current_header
                df.at[idx, '_subheader'] = current_subheader

        self.logger(f"🔗 Пропагация заголовков завершена")
        return df

    def _remove_before_first_header(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет строки до первого заголовка "Раздел цифра.".
        """
        if df.empty:
            return df

        df = df.copy()

        first_header_idx = None
        for idx in df.index:
            if df.at[idx, '_header']:  # Нашли заголовок
                first_header_idx = idx
                break

        if first_header_idx is not None:
            rows_before = df.index.get_loc(first_header_idx)
            if rows_before > 0:
                df = df.iloc[first_header_idx:].reset_index(drop=True)
                self.logger(f"🗑️ Удалено {rows_before} строк до первого заголовка")

        return df

    # ═══════════════════════════════════════════════════════════════
    # EXCEL
    # ═══════════════════════════════════════════════════════════════

    def _save_to_excel(self, df: pd.DataFrame, path: str):
        from openpyxl.styles import Font

        if df.empty:
            df = pd.DataFrame([{"№ п/п": "НЕТ ДАННЫХ", "Наименование": "", "Ед. изм.": "", "Кол-во": ""}])

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            cols_order = ["№ п/п", "Раздел", "Подраздел", "Наименование", "Ед. изм.", "Кол-во"]

            final_df = pd.DataFrame()
            for col in cols_order:
                if col == 'Раздел':
                    final_df[col] = df.get('_header', '')
                elif col == 'Подраздел':
                    final_df[col] = df.get('_subheader', '')
                elif col in df.columns:
                    final_df[col] = df[col]
                else:
                    final_df[col] = ''

            final_df.to_excel(writer, index=False, sheet_name="Ведомость")
            ws = writer.book["Ведомость"]
            ws.freeze_panes = "A2"

            ws.column_dimensions['A'].width = 8
            ws.column_dimensions['B'].width = 35
            ws.column_dimensions['C'].width = 45
            ws.column_dimensions['D'].width = 60
            ws.column_dimensions['E'].width = 10
            ws.column_dimensions['F'].width = 10

            # Выделение заголовков жирным
            bold_font = Font(bold=True)

            for row_idx, df_idx in enumerate(df.index, start=2):
                if df.at[df_idx, '_header']:  # Только заголовки "Раздел"
                    for col_idx in range(1, len(cols_order) + 1):
                        cell = ws.cell(row=row_idx, column=col_idx)
                        cell.font = bold_font

        self.logger(f"✅ Сохранено: {path}")

    def _build_structure_entry_for_row(self, header_val: str) -> tuple:
        """
        Строит один кортеж для структуры типа ('Раздел ВРX', 'Заголовок раздела', 3, -1).
        header_val: значение из колонки '_header' для конкретной строки.
        """
        # Извлекаем первую цифру из заголовка (например, '1 Металлоконструкции' -> '1')
        if not header_val:
            return None
        match = re.match(r'^(\d+)', str(header_val).strip())
        if match:
            num = match.group(1)
            # Формируем 'Раздел ВРX'
            section_label = f"Раздел ВР{num}"
            return (section_label, header_val.strip(), 3, -1)
        else:
            # Если цифра не найдена, можно вернуть общий ярлык или None
            # Пока вернём None, чтобы не добавлять в список
            return None

    def _build_summary_row(self, current_row_series) -> list:
        """
        Строит строку-сводку *для одной строки* DataFrame.
        Использует данные из *одной* строки current_row_series (pd.Series).
        Возвращает список кортежей для столбца 'structure'.
        [   ('<номер проекта>', '<наименование сооружения>', 1, 1),
            ('Марка РД', '<шифр>', 2, 1),
            ('Раздел ВР1', '<первый заголовок раздела>', 3, -1) # <-- Только если в этой строке есть _header ]
        """
        summary = []

        # 1. Номер проекта и Наименование сооружения (берём из переданной строки)
        proj_num = current_row_series.get('_project_number', '')
        proj_name = current_row_series.get('Наименование зданий, сооружений, систем и установок', '')
        if proj_num and proj_name:
            summary.append((proj_num, proj_name, 1, 1))

        # 2. Марка РД (шифр) (берём из переданной строки)
        file_type = current_row_series.get('_file_type', '')
        if file_type:
            summary.append(("Марка РД", file_type, 2, 1))

        # 3. Раздел ВРX (только если в *этой строке* есть _header)
        header_val = current_row_series.get('_header', '')
        if header_val:
            entry = self._build_structure_entry_for_row(header_val)
            if entry:
                summary.append(entry)

        return summary

    def _save_to_excel_header(self, df: pd.DataFrame, path: str):
        from openpyxl.styles import Font

        if df.empty:
            df = pd.DataFrame([{
                "№ п/п": "",
                "Наименование ВиКР": "",
                "Раздел": "НЕТ РАЗДЕЛОВ",
                "Подраздел": "",
                "Наименование работ": "",
                "Ед. изм.": "",
                "Кол-во": "",
                "Марка РД": "",
                "Номер проекта": "",
                "Наименование зданий, сооружений, систем и установок": "",
                "structure": "[]"
            }])

        # Фильтруем строки с заголовками разделов или подразделов
        header_rows = df[
            ((df['_header'].notna()) & (df['_header'] != '')) |
            ((df['_subheader'].notna()) & (df['_subheader'] != ''))
            ].copy()

        if header_rows.empty:
            header_rows = pd.DataFrame([{
                "№ п/п": "",
                "Наименование ВиКР": "",
                "Раздел": "НЕТ РАЗДЕЛОВ",
                "Подраздел": "",
                "Наименование работ": "",
                "Ед. изм.": "",
                "Кол-во": "",
                "Марка РД": "",
                "Номер проекта": "",
                "Наименование зданий, сооружений, систем и установок": "",
                "structure": "[]"
            }])
        else:
            # Формируем DataFrame с нужными колонками
            header_rows['№ п/п'] = header_rows.get('№ п/п', '')  # 🔥 Добавляем № п/п
            header_rows['Наименование ВиКР'] = self.vikr_name
            header_rows['Раздел'] = header_rows.get('_header', '')
            header_rows['Подраздел'] = header_rows.get('_subheader', '')
            header_rows['Наименование работ'] = header_rows.get('Наименование', '')
            header_rows['Ед. изм.'] = header_rows.get('Ед. изм.', '')
            header_rows['Кол-во'] = header_rows.get('Кол-во', '')
            header_rows['Марка РД'] = header_rows.get('_file_type', '')
            header_rows['Номер проекта'] = header_rows.get('_project_number', '')
            header_rows['Наименование зданий, сооружений, систем и установок'] = header_rows.get(
                'Наименование зданий, сооружений, систем и установок', '')

            # 🔥 Генерируем сводку ОТДЕЛЬНО для КАЖДОЙ строки
            structure_list_strs = []
            for _, row in header_rows.iterrows():
                summary_list = self._build_summary_row(row)  # Передаём Series (одну строку)
                structure_list_strs.append(str(summary_list))  # Преобразуем в строку

            # Присваиваем столбцу structure список строк, по одной для каждой строки header_rows
            header_rows['structure'] = structure_list_strs

        # 🔥 Обновлённый порядок столбцов: 'Наименование ВиКР' первым, затем 'Наименование работ', 'Ед. изм.', 'Кол-во' после 'Подраздел'
        cols_order = [
            '№ п/п', 'Наименование ВиКР', 'Раздел', 'Подраздел',
            'Наименование работ', 'Ед. изм.', 'Кол-во',
            'Марка РД', 'Номер проекта',
            'Наименование зданий, сооружений, систем и установок', 'structure'
        ]
        final_df = header_rows[cols_order].copy()

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            final_df.to_excel(writer, index=False, sheet_name="Заголовки")
            ws = writer.book["Заголовки"]

            # 🔥 Обновлённая ширина колонок (добавлены новые)
            # A - Наименование ВиКР, B - Раздел, C - Подраздел, D - Наименование работ, E - Ед. изм., F - Кол-во
            # G - Марка РД, H - Номер проекта, I - Наименование зданий..., J - structure
            ws.column_dimensions['A'].width = 6
            ws.column_dimensions['B'].width = 60  # 🔥 Наименование ВиКР
            ws.column_dimensions['C'].width = 30  # 🔥 Раздел
            ws.column_dimensions['D'].width = 20  # 🔥 Подраздел
            ws.column_dimensions['E'].width = 60  # 🔥 Наименование работ (раньше D было Марка РД)
            ws.column_dimensions['F'].width = 10  # 🔥 Ед. изм. (раньше E было Номер проекта)
            ws.column_dimensions['G'].width = 10  # 🔥 Кол-во (раньше F было Наименование зданий...)
            ws.column_dimensions['H'].width = 45  # 🔥 Марка РД (раньше G было structure)
            ws.column_dimensions['I'].width = 15  # 🔥 Номер проекта
            ws.column_dimensions['J'].width = 50  # 🔥 Наименование зданий...
            ws.column_dimensions['K'].width = 80  # 🔥 structure

            # Жирный шрифт для строк данных
            bold_font = Font(bold=True)
            for row_idx, _ in enumerate(final_df.index, start=2):  # Начинаем с 2 (первая строка - заголовки Excel)
                # Применяем жирный шрифт ко всем ячейкам в строке данных
                for col_idx in range(1, len(cols_order) + 1):  # 1-based индексация для openpyxl
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell.font = bold_font

        self.logger(f"✅ Сохранено: {len(final_df)} строк в {path}")

    # ═══════════════════════════════════════════════════════════════
    # CORE
    # ═══════════════════════════════════════════════════════════════

    def _process_single_page(self, page_idx: int, page: Image.Image, total_pages: int) -> Optional[pd.DataFrame]:
        try:
            with self._log_lock:
                self.logger(f"📄 Страница {page_idx + 1}/{total_pages} — OCR...")
            # 🔥 ВЫЗЫВАЕМ _ocr_page, а не _ocr_from_image напрямую
            df_page = self._ocr_page(page, page_number=page_idx + 1)
            df_page["page"] = page_idx + 1
            with self._log_lock:
                self.logger(f"  ✅ Страница {page_idx + 1}: {len(df_page)} строк")
            return df_page
        except Exception as e:
            with self._log_lock:
                self.logger(f"  ⚠️ Ошибка на странице {page_idx + 1}: {e}")
            return None

    def _load_lookup_table(self) -> Optional[pd.DataFrame]:
        """
        Загружает лист 'Перечень сооружений (ОССР)' из xlsx_lookup_path.
        Подготавливает столбцы 'Диапазон кодов, код' и 'Наименование зданий, сооружений, систем и установок'.
        """
        if not self.xlsx_lookup_path:
            self.logger("⚠️ Путь к xlsx_lookup_path не задан, пропускаем загрузку таблицы поиска.")
            return None

        try:
            # Загружаем только нужный лист
            lookup_df = pd.read_excel(self.xlsx_lookup_path, sheet_name='Перечень сооружений (ОССР)')
            self.logger(f"✅ Загружен лист 'Перечень сооружений (ОССР)' из {self.xlsx_lookup_path}, {len(lookup_df)} строк.")
        except FileNotFoundError:
            self.logger(f"❌ Файл {self.xlsx_lookup_path} не найден.")
            return None
        except Exception as e:
            self.logger(f"❌ Ошибка при загрузке {self.xlsx_lookup_path}: {e}")
            return None

        # Проверим, что требуемые столбцы существуют
        required_cols = ['Диапазон кодов, код', 'Наименование зданий, сооружений, систем и установок']
        missing_cols = [col for col in required_cols if col not in lookup_df.columns]
        if missing_cols:
            self.logger(f"❌ В листе 'Перечень сооружений (ОССР)' отсутствуют столбцы: {missing_cols}")
            return None

        return lookup_df

    def _get_unique_project_name(self, df: pd.DataFrame, lookup_df: pd.DataFrame) -> str:
        """
        Извлекает уникальный номер проекта из df и находит соответствующее наименование в lookup_df.
        Возвращает наименование или пустую строку.
        """
        if lookup_df is None:
            self.logger("⚠️ lookup_df отсутствует, невозможно получить наименование.")
            return ""

        # 🔍 Извлекаем уникальные значения _project_number, исключая NaN/None/не-строки/не-цифры
        unique_proj_nums = df['_project_number'].dropna()
        unique_proj_nums = unique_proj_nums[unique_proj_nums.apply(lambda x: isinstance(x, str) and x.isdigit())]
        unique_proj_nums = unique_proj_nums.unique()

        if len(unique_proj_nums) == 0:
            self.logger("  ⚠️ _project_number не найден или не содержит чисел, пропускаем сопоставление.")
            return ""
        elif len(unique_proj_nums) > 1:
            self.logger(f"  ⚠️ Найдено несколько уникальных _project_number: {unique_proj_nums}. Используем первое.")

        # Берём первое (и, как ожидается, единственное) уникальное числовое значение
        proj_num_str = unique_proj_nums[0]
        proj_num_int = int(proj_num_str)

        # 🔍 Поиск в lookup_df
        matched_name = self._find_name_by_number(proj_num_int, lookup_df)
        if matched_name:
            self.logger(f"  🏷️ Найдено наименование для номера проекта '{proj_num_str}': {matched_name}...")
        else:
            self.logger(f"  ⚠️ Наименование для номера проекта '{proj_num_str}' не найдено.")

        return matched_name

    def _map_project_number_to_name(self, df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Сопоставляет _project_number с наименованием из lookup_df.
        Ищет _project_number в столбце 'Диапазон кодов, код' (который может содержать диапазоны).
        Добавляет столбец 'Наименование зданий, сооружений, систем и установок'.
        """
        if df.empty:
            self.logger("⚠️ DataFrame пуст, пропускаем сопоставление.")
            df = df.copy()
            df['Наименование зданий, сооружений, систем и установок'] = ""
            return df

        # 🔥 НОВОЕ: Получаем наименование ОДИН РАЗ
        matched_name = self._get_unique_project_name(df, lookup_df)

        df = df.copy()
        # 🔥 Заполняем весь столбец одним значением
        df['Наименование зданий, сооружений, систем и установок'] = matched_name

        return df

    def _find_name_by_number(self, number: int, lookup_df: pd.DataFrame) -> str:
        """
        Ищет наименование по числу в столбце 'Диапазон кодов, код'.
        Совпадение только если в ячейке находится ТОЛЬКО одно число, равное 'number'.
        Диапазоны (например, '1234-5678') игнорируются.
        Возвращает 'Наименование...' если найдено, иначе ''.
        """
        # Итерируемся по строкам lookup_df
        for _, row in lookup_df.iterrows():
            range_str = str(row['Диапазон кодов, код']).strip()
            name = row['Наименование зданий, сооружений, систем и установок']

            # Проверяем, содержит ли строка символ '-'
            if '-' in range_str:
                # Это диапазон или строка с тире, игнорируем её
                continue

            # Проверяем, содержит ли строка запятые (несколько значений)
            if ',' in range_str:
                # Это список значений, разбиваем и проверяем каждое
                parts = [part.strip() for part in range_str.split(',')]
                for part in parts:
                    part = part.strip()
                    try:
                        if int(part) == number:
                            return name
                    except ValueError:
                        # Если часть не является числом, пропускаем
                        continue
            else:
                # Это одиночная строка, проверяем, является ли она числом
                try:
                    if int(range_str) == number:
                        return name
                except ValueError:
                    # Если строка не является числом, пропускаем
                    continue

        return ""  # Не найдено

    def process(self) -> pd.DataFrame:
        self.logger(f"📄 PDF: {self.pdf_path} (DPI={self.dpi})")
        self.logger(f"⚡ Ollama: {self.ollama_url}, model={self.ollama_model}, parallel={self.max_parallel_ocr}")

        try:
            pages = convert_from_path(self.pdf_path, dpi=self.dpi, poppler_path=self.poppler_path)
        except Exception as e:
            raise RuntimeError(f"pdf2image failed: {e}")

        total = len(pages)
        self.logger(f"📄 Страниц: {total}")

        all_dfs = []

        # 🔥 Извлекаем шифр из ПЕРВОЙ страницы отдельно, до основного цикла
        if pages:
            first_page_img = pages[0]
            # Второй проход для первой страницы
            raw_stamp = self._ocr_from_image(first_page_img, prompt=self.OCR_PROMPT_STAMP)
            main_file_type = self._extract_file_type_from_raw_stamp(raw_stamp)
            if main_file_type:
                self.main_file_type = main_file_type  # 🔥 Сохраняем в атрибут
                self.logger(f"🏷️ Найден основной шифр из штампа первой страницы: {self.main_file_type}")
            else:
                self.logger(f"⚠️ Шифр не найден в штампе первой страницы.")

        # Основной цикл обработки страниц (только первый проход)
        if self.max_parallel_ocr <= 1:
            for i, page in enumerate(pages):
                self.progress_callback(i + 1, total, "📄 OCR")
                df_page = self._process_single_page(i, page, total)
                if df_page is not None:
                    all_dfs.append(df_page)
                    with self._log_lock:
                        self.logger(f"  📊 Страница {i + 1}: {len(df_page)} строк")
        else:
            # ... параллельная обработка ...
            all_dfs_dict = {}
            completed = 0
            lock = Lock()

            with ThreadPoolExecutor(max_workers=self.max_parallel_ocr) as executor:
                futures = {executor.submit(self._process_single_page, i, page, total): i
                           for i, page in enumerate(pages)}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        df_page = future.result()
                        if df_page is not None:
                            with lock:
                                all_dfs_dict[idx] = df_page
                    except Exception as e:
                        self.logger(f"  💥 Страница {idx + 1}: {e}")
                    with lock:
                        completed += 1
                    self.progress_callback(completed, total, "📄 OCR")

            all_dfs = [all_dfs_dict[i] for i in sorted(all_dfs_dict.keys())]

        self.logger(f"✅ OCR завершён: собрано {len(all_dfs)} страниц")

        if not all_dfs:
            self._df = pd.DataFrame()
        else:
            self.logger(f"🔧 Начало структуризации...")

            # 1. Объединяем все страницы (только основные таблицы)
            self._df = pd.concat(all_dfs, ignore_index=True)
            self.logger(f"   📊 Объединено: {len(self._df)} строк")
            self.logger(f"   📊 Колонки self._df до детекта: {list(self._df.columns)}")

            # 2. 🔥 Детект разделов (колонка _is_section создаётся здесь)
            self._df = self._detect_markdown_sections(self._df)

            # 3. Логирование статистики по разделам (теперь это безопасно)
            section_count = self._df['_is_section'].sum()
            header_count = len(self._df[self._df['_header'].notna() & (self._df['_header'] != '')])
            self.logger(f"   🔖 Найдено разделов: {section_count} (заголовки: {header_count})")

            # 4. Слияние подзаголовков
            self._df = self._merge_multiline_subheaders(self._df)

            # 5. Генерация иерархической нумерации
            self._df = self._generate_item_numbers(self._df)
            self._df = self._merge_multiline_names(self._df)

            # 6. Пропагация заголовков
            self._df = self._propagate_headers_across_pages(self._df)

            # 7. Построение иерархии
            self._df = self._build_hierarchy(self._df)

            # 8. Удаление преамбулы
            self._df = self._remove_before_first_header(self._df)

            # 🔥 9. Заполняем шифр в разделах и подразделах (после пропагации)
            # Используем self.main_file_type, который был извлечён ранее
            self._df = self._fill_file_type(self._df, self.main_file_type)

            lookup_df = self._load_lookup_table()

            # 🔥 13. Сопоставляем номер проекта с наименованием
            self._df = self._map_project_number_to_name(self._df, lookup_df)

            self.logger(f"✅ Структуризация завершена: {len(self._df)} строк")

        # Выгрузка модели (базовая)
        try:
            requests.post(f"{self.ollama_url}/api/generate",
                          json={"model": self.ollama_model, "prompt": "", "keep_alive": 0}, timeout=10)
            self.logger(f"🧹 {self.ollama_model} выгружена из VRAM")
        except:
            pass

        self.logger(f"✅ Итого: {len(self._df)} строк")
        return self._df

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return self._df.copy() if self._df is not None else None

    def save_to_excel(self, output_path: Optional[str] = None):
        path = output_path or self.output_xlsx
        if not path:
            raise ValueError("output_path не задан")
        if self._df is None:
            raise RuntimeError("DataFrame не создан. Вызовите .process()")

        # 🔥 Выберите нужный метод:
        # self._save_to_excel(self._df, path)           # <- полная ведомость
        self._save_to_excel_header(self._df, path)  # <- только заголовки, подразделы и шифр