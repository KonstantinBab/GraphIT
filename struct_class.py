# struct_class.py — v5: Классификация Работа/Материал (QLoRA)
# Путь к адаптеру приходит из pipeline. Путь к базе можно указать здесь.

import os

# =====================================================================
# НАСТРОЙКИ (Измените пути под вашу систему)
# =====================================================================
# ВАЖНО: HF_HOME должен быть установлен ДО импорта transformers
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "../train_selector/hug_models"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

import pandas as pd
import torch
from pathlib import Path
from threading import Lock
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import concurrent.futures
# Путь к базовой модели Qwen (если она скачана локально).
# Если оставить None, модель попытается загрузиться из кэша HF или интернета.
# Пример: r"C:\Models\Qwen2.5-1.5B-Instruct"
DEFAULT_BASE_MODEL_PATH = None

# Путь к адаптеру по умолчанию (если не передан из pipeline)
# Пример: r"classifier_llm\final"
DEFAULT_ADAPTER_PATH = r"classifier_llm\final"


class GESNClassifier:
    """
    Классификатор Работа/Материал на базе QLoRA (Qwen2.5-1.5B-Instruct).
    """

    PARSE_EXTRA_COLUMNS = [
        'Наименование ВиКР',
        'Марка РД',
        'Номер проекта',
        'Наименование зданий, сооружений, систем и установок',
        'structure',
    ]

    SYSTEM_PROMPT = """Ты эксперт по строительным сметам. Определи тип позиции: Работа или Материал.

Правила:
- Арматура — всегда Работа
- "Стоимостные параметры", "Исключение затрат" — Материал
- Если есть действие (монтаж, устройство, укладка) — Работа
- Если это только поставка материала без работы — Материал

Ответь ОДНИМ словом: Работа или Материал"""

    def __init__(
            self,
            model_path: Optional[str] = None,
            base_model: Optional[str] = None,
            max_workers: int = 4,
            device: Optional[str] = None,
            progress_callback=None,
            use_4bit: bool = True,
    ):
        """
        Args:
            model_path: Путь к LoRA-адаптеру (папка final/). Приходит из pipeline.
            base_model: Путь к базовой модели Qwen (локально) или имя на HF.
            max_workers: Потоков для классификации.
            device: 'cuda', 'cpu', 'mps'.
            use_4bit: Квантование для экономии VRAM.
        """
        # 1. Определяем путь к адаптеру (LoRA)
        self.adapter_path = Path(model_path) if model_path else Path(DEFAULT_ADAPTER_PATH)

        # 2. Определяем путь к базовой модели
        # Приоритет: Аргумент -> Настройка в файле -> Имя на HF
        if base_model:
            self.base_model_source = base_model
        elif DEFAULT_BASE_MODEL_PATH and Path(DEFAULT_BASE_MODEL_PATH).exists():
            self.base_model_source = DEFAULT_BASE_MODEL_PATH
        else:
            self.base_model_source = "Qwen/Qwen2.5-3B-Instruct"

        self.max_workers = max_workers
        self.progress_callback = progress_callback or (lambda c, t, s: None)
        self.use_4bit = use_4bit
        self._print_lock = Lock()

        # Устройство
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = None
        self.model = None
        self._load_model()

    def _safe_print(self, msg: str):
        with self._print_lock:
            print(msg)

    def _load_model(self):
        """Загружает токенизатор и модель."""
        if not self.adapter_path.exists():
            raise FileNotFoundError(f"❌ Адаптер не найден: {self.adapter_path}")

        # Резолвим HF-имя в локальный путь к snapshot
        if Path(self.base_model_source).exists():
            self._safe_print(f"📦 База (локально): {self.base_model_source}")
        else:
            # Ищем модель в локальном кэше HF
            hf_home = os.environ.get("HF_HOME", "")
            model_dir_name = "models--" + self.base_model_source.replace("/", "--")
            snapshot_dir = Path(hf_home) / model_dir_name / "snapshots"
            if snapshot_dir.exists():
                # Берём первый (обычно единственный) snapshot
                snapshots = [d for d in snapshot_dir.iterdir() if d.is_dir()]
                if snapshots:
                    self.base_model_source = str(snapshots[0])
                    self._safe_print(f"📦 База (кэш): {self.base_model_source}")
                else:
                    raise FileNotFoundError(f"❌ Нет snapshots в {snapshot_dir}")
            else:
                raise FileNotFoundError(
                    f"❌ Модель не найдена ни локально, ни в кэше HF.\n"
                    f"   Искали: {self.base_model_source}\n"
                    f"   HF_HOME: {hf_home}\n"
                    f"   Snapshot dir: {snapshot_dir}"
                )
        local_files_only = True

        self._safe_print(f"🔌 Адаптер: {self.adapter_path}")
        self._safe_print(f"🖥️  Устройство: {self.device}")

        # Токенизатор
        self._safe_print("⏳ Загрузка токенизатора...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_source,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._safe_print("✅ Токенизатор загружен")

        # Квантование
        bnb_config = None
        if self.use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Базовая модель
        self._safe_print("⏳ Загрузка базовой модели (4-bit)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_source,
            quantization_config=bnb_config if bnb_config else None,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            dtype=torch.bfloat16 if self.device == "cuda" else None,
            local_files_only=local_files_only,
        )
        self._safe_print("✅ Базовая модель загружена")

        # Применяем LoRA-адаптер
        self._safe_print("⏳ Загрузка LoRA-адаптера...")
        self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
        self.model.eval()
        self._safe_print("✅ LoRA применена")

        if self.device != "cuda":
            self.model.to(self.device)

        self._safe_print(f"✅ Модель готова")

    def _format_input(self, name: str, unit: str, section: str) -> str:
        return (
            f"Наименование: {name}\n"
            f"Ед.изм.: {unit}\n"
            f"Раздел: {section}"
        )

    def _classify_single(self, name: str, unit: str, section: str) -> str:
        if self.model is None or self.tokenizer is None:
            return 'material'

        try:
            user_msg = self._format_input(name, unit, section)
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            answer_lower = answer.lower()
            if "работ" in answer_lower:
                return 'work'
            elif "материал" in answer_lower:
                return 'material'
            else:
                return 'material'

        except Exception as e:
            self._safe_print(f"❌ Ошибка инференса: {e}")
            return 'material'

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ['Наименование', 'Ед. изм.', 'Кол-во', 'Раздел', 'Подраздел', *self.PARSE_EXTRA_COLUMNS]:
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].fillna('').astype(str).str.strip()

        if df['Наименование'].eq('').all() and 'Наименование работ' in df.columns:
            df['Наименование'] = df['Наименование работ']
        else:
            df['Наименование'] = df['Наименование'].where(
                df['Наименование'].astype(bool),
                df['Наименование работ'],
            )

        df['Наименование работ'] = df['Наименование работ'].where(
            df['Наименование работ'].astype(bool),
            df['Наименование'],
        )

        if '№ п/п' not in df.columns:
            df['№ п/п'] = range(1, len(df) + 1)
        if 'Тип' not in df.columns:
            df['Тип'] = ''
        return df

    def classify_work_material(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df['Ед. изм.'].str.strip().astype(bool)) | (df['Кол-во'].str.strip().astype(bool))
        indices = df[mask].index.tolist()
        total = len(indices)

        if total == 0:
            self._safe_print("ℹ️ Нет строк для классификации")
            return df

        self._safe_print(f"🧠 Классификация {total} строк (workers={self.max_workers})...")

        results = {}
        completed = [0]
        lock = Lock()

        def process_one(idx):
            row = df.loc[idx]
            name = str(row['Наименование'])
            unit = str(row['Ед. изм.'])
            section = str(row.get('section', '') or row.get('Раздел', ''))
            typ = self._classify_single(name, unit, section)

            with lock:
                results[idx] = typ
                completed[0] += 1
                self.progress_callback(completed[0], total, "🧠 Классификация")
                if completed[0] % 10 == 0 or completed[0] == total:
                    self._safe_print(f"  ✅ {completed[0]}/{total}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_one, idx) for idx in indices]
            concurrent.futures.wait(futures)

        for idx, typ in results.items():
            df.at[idx, 'Тип'] = typ

        w = (df['Тип'] == 'work').sum()
        m = (df['Тип'] == 'material').sum()
        self._safe_print(f"✅ Классификация завершена: work={w}, material={m}")
        return df

    def reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        target_order = [
            '№ п/п', 'Раздел', 'Подраздел', 'Наименование', 'Ед. изм.', 'Кол-во', 'Тип'
        ]
        existing = [c for c in target_order if c in df.columns]
        parse_tail = [c for c in self.PARSE_EXTRA_COLUMNS if c in df.columns]
        extra = [c for c in df.columns if c not in target_order and c not in parse_tail]
        return df[existing + extra + parse_tail]

    def process(self, df_input: pd.DataFrame) -> pd.DataFrame:
        self._safe_print("\n" + "=" * 70)
        self._safe_print("КЛАССИФИКАЦИЯ: Работа / Материал")
        self._safe_print("=" * 70 + "\n")

        df = self.preprocess_df(df_input)
        df = self.classify_work_material(df)
        df = self.reorder_columns(df)

        self._safe_print("\n✅ Обработка завершена")
        return df

    def cleanup(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._safe_print("🧹 Память модели очищена")

    @staticmethod
    def load_from_excel(path: str) -> pd.DataFrame:
        return pd.read_excel(path)

    @staticmethod
    def save_to_excel(df: pd.DataFrame, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=False)
        print(f"💾 Сохранено: {path}")


if __name__ == "__main__":
    # Тестовый запуск
    classifier = GESNClassifier(
        model_path=DEFAULT_ADAPTER_PATH,
        base_model=DEFAULT_BASE_MODEL_PATH,
        max_workers=1
    )
    print("✅ Тест инициализации пройден")
    classifier.cleanup()
