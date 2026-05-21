# classifier.py — Гибридный классификатор Материал/Работа
#
# Жёсткие правила (перекрывают модель) + ruBERT-base для сложных случаев
#
# Использование:
#   classifier = WorkMaterialClassifier("path/to/model")
#   result = classifier.classify("Устройство свай", "м3", "Свайное основание")
#   # → "work"

import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional

# ══════════════════════════════════════════════════════════════
# ЖЁСТКИЕ ПРАВИЛА
# ══════════════════════════════════════════════════════════════

# Слова-маркеры → ВСЕГДА Работа (отглагольные существительные = действие)
ALWAYS_WORK_KEYWORDS = [
    "устройство", "монтаж", "демонтаж", "установка", "укладка",
    "засыпка", "разработка", "бурение", "забивка", "вдавливание",
    "нанесение", "удаление", "окрашивание", "обезжиривание",
    "сварка", "армирование", "бетонирование", "штукатурка",
    "облицовка", "отделка", "заливка", "выравнивание",
    "уплотнение", "трамбовка", "планировка", "погружение",
    "испытание", "промывка", "продувка", "прокладка",
    "крепление", "утепление", "гидроизоляция", "теплоизоляция",
    "очистка", "обработка", "пропитка", "разборка", "сборка",
    "перемещение", "транспортировка", "выемка", "насыпка",
]

# Фразы-маркеры → ВСЕГДА Работа
ALWAYS_WORK_PHRASES = [
    "арматур",       # "каркас из арматуры", "арматура класса"
    "каркас из",     # "каркас из арматуры", "каркас пространственный"
]

# Слова-маркеры → ВСЕГДА Материал
ALWAYS_MATERIAL_KEYWORDS = [
    "стоимость", "исключение затрат", "стоимостные параметры",
]


def apply_rules(name: str) -> Optional[str]:
    """
    Применяет жёсткие правила.
    Возвращает "work", "material" или None (если правила не сработали).
    """
    name_lower = name.lower().strip()

    # Материал — проверяем первым (приоритет)
    for kw in ALWAYS_MATERIAL_KEYWORDS:
        if kw in name_lower:
            return "material"

    # Работа — проверяем все слова в наименовании
    words = re.split(r'[\s\-,;()]+', name_lower)
    for word in words:
        for kw in ALWAYS_WORK_KEYWORDS:
            if word.startswith(kw) or word == kw:
                return "work"

    # Работа — по фразам (в любом месте)
    for phrase in ALWAYS_WORK_PHRASES:
        if phrase in name_lower:
            return "work"

    return None  # правила не сработали → решает модель


# ══════════════════════════════════════════════════════════════
# КЛАССИФИКАТОР
# ══════════════════════════════════════════════════════════════

class WorkMaterialClassifier:
    """
    Гибридный классификатор: правила + ruBERT-base.
    Правила перекрывают модель для очевидных случаев.
    """

    ID2LABEL = {0: "material", 1: "work"}

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"🏷️ Загрузка классификатора: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"   ✅ Классификатор на {self.device}")

    def classify(self, name: str, unit: str = "", section: str = "", typ_vr: str = "Н") -> str:
        """
        Классифицирует строку.
        Возвращает "work" или "material".
        """
        # 1. Жёсткие правила (отключены — только модель)
        # rule_result = apply_rules(name)
        # if rule_result is not None:
        #     return rule_result

        # 2. Модель
        text = (
            f"Наименование: {name}. "
            f"Единица: {unit}. "
            f"Тип ВР: {typ_vr}. "
            f"Раздел: {section}"
        )

        inputs = self.tokenizer(
            text, max_length=256, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        pred = torch.argmax(logits, dim=1).item()
        return self.ID2LABEL[pred]

    def classify_batch(self, names, units, sections, typ_vrs=None) -> list:
        """Классифицирует батч строк."""
        results = []
        for i, name in enumerate(names):
            unit = units[i] if i < len(units) else ""
            section = sections[i] if i < len(sections) else ""
            typ_vr = typ_vrs[i] if typ_vrs and i < len(typ_vrs) else "Н"
            results.append(self.classify(name, unit, section, typ_vr))
        return results

    def cleanup(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
        print("🧹 Классификатор выгружен")


# ══════════════════════════════════════════════════════════════
# ТЕСТ
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        # (наименование, ожидание)
        ("Устройство железобетонных буронабивных свай", "work"),
        ("Монтаж ростверков", "work"),
        ("Засыпка пазух фундаментов", "work"),
        ("Каркас из арматуры класса Ас500С", "work"),
        ("Каркас из арматура класса А240 ГОСТ 34028", "work"),
        ("Арматура класса А500С", "work"),
        ("Нанесение антикоррозионного покрытия", "work"),
        ("Удаление ржавчины", "work"),
        ("Окрашивание поверхности", "work"),
        ("Обезжиривание металлоконструкций", "work"),
        ("Бурение скважин глубиной до 10м", "work"),
        ("Установка сваи Св1", "work"),
        ("Стоимость свай из труб ГОСТ 10704-91", "material"),
        ("Стоимостные параметры:", "material"),
        ("Исключение затрат на электродуговую сварку", "material"),
    ]

    print("═" * 50)
    print("ТЕСТ ПРАВИЛ (без модели)")
    print("═" * 50)

    passed = 0
    for name, expected in test_cases:
        result = apply_rules(name)
        if result is None:
            result = "???(модель)"
        ok = "✅" if result == expected else "❌"
        if result != expected:
            print(f"  {ok} '{name[:50]}' → {result} (ожидалось {expected})")
        else:
            passed += 1

    print(f"\n  Пройдено: {passed}/{len(test_cases)}")