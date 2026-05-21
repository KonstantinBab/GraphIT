# section_detector.py — Определение разделов в строительных ведомостях
#
# Использует:
#   1. Паттерны (нет ед.изм., нет кол-во)
#   2. Морфологический анализ (pymorphy2) — нет отглагольных существительных
#   3. Длина текста (разделы короткие)
#   4. Отсутствие технических деталей (ГОСТ, диаметры, марки)
#
# pip install pymorphy2

import re
import pymorphy3 as pymorphy2

morph = pymorphy2.MorphAnalyzer()

# Суффиксы отглагольных существительных
VERBAL_NOUN_SUFFIXES = (
    "ка", "ние", "ие", "тие", "ство", "овка", "ировка", "ение", "ание",
    "ёвка", "ация", "яция", "изация", "ировка", "монтаж",
)

# Конкретные отглагольные существительные которые встречаются в работах
VERBAL_NOUNS = {
    "засыпка", "разработка", "устройство", "монтаж", "демонтаж",
    "установка", "укладка", "нанесение", "покрытие", "бетонирование",
    "армирование", "сварка", "окраска", "грунтовка", "изготовление",
    "разборка", "сборка", "погружение", "забивка", "вдавливание",
    "испытание", "промывка", "продувка", "прокладка", "перекладка",
    "крепление", "утепление", "гидроизоляция", "теплоизоляция",
    "штукатурка", "облицовка", "отделка", "заливка", "выравнивание",
    "уплотнение", "трамбовка", "планировка", "выемка", "насыпка",
    "обратная", "перемещение", "транспортировка", "подача",
    "очистка", "обработка", "пропитка", "защита", "огнезащита",
    "антикоррозийная", "стоимость",
}

# Технические паттерны (если есть — точно не раздел)
TECH_PATTERNS = re.compile(
    r'ГОСТ|ТУ\s*\d|[Øø∅]\s*\d|диаметр|толщин|класс[а-я]*\s+[А-ЯЁABC]|'
    r'марк[аи]\s+|С\d{3}|В\d{1,3}|М\d{2,4}|F\d{2,4}|W\d|'
    r'\d+\s*мм|\d+\s*мкм|\d+\s*см|\d+,\d+|'
    r'из\s+стали|из\s+бетона|из\s+раствора',
    re.IGNORECASE
)

# Паттерн: раздел начинается с номера (опционально) + текст
SECTION_NUM_PATTERN = re.compile(r'^\*{0,3}\s*(\d{1,2})\s+(.+?)\s*\*{0,3}$')
SECTION_PLAIN_PATTERN = re.compile(r'^\*{0,3}\s*([А-ЯЁA-Z][а-яёА-ЯЁa-zA-Z\s\-()]+?)\s*\*{0,3}$')


def is_verbal_noun(word: str) -> bool:
    """Проверяет является ли слово отглагольным существительным."""
    word_lower = word.lower().strip()

    # Прямое совпадение со словарём
    if word_lower in VERBAL_NOUNS:
        return True

    # Проверка суффиксов
    for suffix in VERBAL_NOUN_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            # Дополнительно проверяем через pymorphy2 — это существительное?
            parsed = morph.parse(word_lower)[0]
            if parsed.tag.POS == 'NOUN':
                return True

    return False


def has_verbal_nouns(text: str) -> bool:
    """
    Проверяет начинается ли текст с отглагольного существительного.
    Только ПЕРВОЕ значимое слово проверяется.

    "Монтаж ростверков" → True (первое слово "монтаж" — действие)
    "Антикоррозионные покрытия" → False (первое слово "антикоррозионные" — прилагательное)
    "Свайное основание" → False (первое слово "свайное" — прилагательное)
    """
    # Убираем номер раздела в начале ("1 Металлоконструкции" → "Металлоконструкции")
    clean = re.sub(r'^\d+\s+', '', text.strip())

    words = re.findall(r'[а-яёА-ЯЁ]+', clean)
    if not words:
        return False

    first_word = words[0].lower()

    # Проверяем первое слово — если оно отглагольное сущ., это работа, не раздел
    if first_word in VERBAL_NOUNS:
        return True

    # Проверяем через pymorphy2 — первое слово существительное с "действенным" суффиксом?
    parsed = morph.parse(first_word)[0]
    if parsed.tag.POS == 'NOUN':
        for suffix in ("ка", "ние", "ие", "тие", "овка", "ировка", "ение", "ание"):
            if first_word.endswith(suffix) and len(first_word) > len(suffix) + 2:
                return True

    return False


def has_verbs(text: str) -> bool:
    """Проверяет есть ли глаголы или деепричастия."""
    words = re.findall(r'[а-яёА-ЯЁ]+', text)
    verb_tags = {'VERB', 'INFN', 'GRND'}
    for word in words:
        parsed = morph.parse(word)[0]
        if parsed.tag.POS in verb_tags:
            return True
    return False


def has_tech_details(text: str) -> bool:
    """Проверяет есть ли технические детали (ГОСТ, диаметры, марки)."""
    return bool(TECH_PATTERNS.search(text))


def word_count(text: str) -> int:
    """Количество значимых слов."""
    words = re.findall(r'[а-яёА-ЯЁa-zA-Z]+', text)
    return len(words)


def clean_section_name(text: str) -> str:
    """Очищает название раздела от ** и лишних пробелов."""
    text = re.sub(r'\*+', '', text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def is_section(name: str, unit: str = "", qty: str = "") -> bool:
    """
    Определяет является ли строка заголовком раздела.

    Возвращает True если:
    - Нет ед.изм. и кол-во
    - Нет отглагольных существительных
    - Нет глаголов
    - Нет технических деталей
    - Текст короткий (до 7 слов без скобок)
    """
    # Если есть ед.изм. или кол-во — точно не раздел
    unit = str(unit).strip()
    qty = str(qty).strip()
    if unit and unit.lower() not in ("", "nan", "-", "—"):
        return False
    if qty and qty.lower() not in ("", "nan", "-", "—", "0"):
        return False

    name_clean = clean_section_name(name)

    if not name_clean:
        return False

    # Слишком длинный — не раздел
    # Но учитываем скобки — "Земляные работы (Фундамент на свайном основании)" допустимо
    name_no_parens = re.sub(r'\([^)]*\)', '', name_clean).strip()
    if word_count(name_no_parens) > 7:
        return False

    # Слишком короткий (1 символ) — мусор
    if len(name_clean) < 3:
        return False

    # Технические детали — не раздел
    if has_tech_details(name_clean):
        return False

    # Глаголы — не раздел
    if has_verbs(name_clean):
        return False

    # Отглагольные существительные — не раздел
    # (но "Антикоррозионные покрытия" — ок, "покрытия" множественное число от "покрытие")
    # Проверяем основные слова (не в скобках)
    if has_verbal_nouns(name_no_parens):
        return False

    # Паттерн: номер + название (типичный раздел)
    if SECTION_NUM_PATTERN.match(name.strip()):
        return True

    # Паттерн: просто название с заглавной
    if SECTION_PLAIN_PATTERN.match(name.strip()):
        return True

    # Всё заглавными — почти наверняка раздел
    if name_clean.upper() == name_clean and len(name_clean) > 3:
        return True

    return False


def detect_sections(df, name_col="Наименование", unit_col="Ед. изм.", qty_col="Кол-во"):
    """
    Проставляет колонку 'is_section' в DataFrame.

    Returns: DataFrame с добавленной колонкой 'is_section' (bool)
    """
    import pandas as pd

    df = df.copy()
    results = []

    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        unit = str(row.get(unit_col, "")).strip()
        qty = str(row.get(qty_col, "")).strip()
        results.append(is_section(name, unit, qty))

    df["is_section"] = results
    return df


def add_section_column(df, name_col="Наименование", unit_col="Ед. изм.", qty_col="Кол-во"):
    """
    Определяет разделы и проставляет колонку 'section'.
    Каждая строка получает название текущего раздела.
    Строки до первого раздела удаляются.
    """
    import pandas as pd

    df = detect_sections(df, name_col, unit_col, qty_col)

    # Проставляем section
    df["section"] = ""
    current_section = ""
    first_section_found = False
    rows_to_keep = []

    for idx, row in df.iterrows():
        if row["is_section"]:
            current_section = clean_section_name(str(row[name_col]))
            first_section_found = True

        if first_section_found:
            df.at[idx, "section"] = current_section
            rows_to_keep.append(idx)

    # Удаляем строки до первого раздела
    df = df.loc[rows_to_keep].reset_index(drop=True)

    return df


# ═══ ТЕСТИРОВАНИЕ ═══
if __name__ == "__main__":
    test_cases = [
        # (текст, ед.изм, кол-во, ожидание)
        ("1 Металлоконструкции", "", "", True),
        ("2 Площадка обслуживания", "", "", True),
        ("2 ПЛОЩАДКА ОБСЛУЖИВАНИЯ", "", "", True),
        ("** 2 ФУНДАМЕНТЫ **", "", "", True),
        ("Земляные работы", "", "", True),
        ("Антикоррозионные покрытия", "", "", True),
        ("Свайное основание", "", "", True),
        ("Прочие работы", "", "", True),
        ("Земляные работы (Фундамент на свайном основании)", "", "", True),
        ("СВАЙНЫЕ РАБОТЫ", "", "", True),
        ("3 Конструкции козырька", "", "", True),
        # НЕ разделы:
        ("Монтаж ростверков", "Т", "0,571", False),
        ("Засыпка пазух фундаментов", "м3", "10", False),
        ("Устройство покрытия толщиной 5 см", "м2", "100", False),
        ("Стоимость м/к из г/к сортового проката", "Т", "0,18", False),
        ("стали С255 ГОСТ 27772-2021", "", "", False),  # продолжение
        ("при высоте до 6м", "", "", False),  # продолжение
        ("- электродами типа Э46А ГОСТ 9467-75", "", "", False),
        ("Монтаж балок и каркаса козырька", "", "", False),  # отглагольное
        ("Устройство антикоррозионного покрытия", "м2", "276", False),
        ("Нанесение грунтовки", "", "", False),  # отглагольное
    ]

    print("═" * 60)
    print("ТЕСТ ОПРЕДЕЛЕНИЯ РАЗДЕЛОВ")
    print("═" * 60)

    passed = 0
    for text, unit, qty, expected in test_cases:
        result = is_section(text, unit, qty)
        ok = "✅" if result == expected else "❌"
        if result != expected:
            print(f"  {ok} '{text}' → {result} (ожидалось {expected})")
        else:
            passed += 1

    failed = len(test_cases) - passed
    print(f"\n  Пройдено: {passed}/{len(test_cases)}")
    if failed:
        print(f"  ❌ Ошибок: {failed}")
    else:
        print("  ✅ Все тесты пройдены!")