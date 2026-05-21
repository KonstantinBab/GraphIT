# matcher_new.py — Матчер: BERTA bi-encoder + LLM Selector
#
# Пайплайн для каждого запроса:
#   1. BERTA encode → вектор
#   2. GPU-поиск top-K в Qdrant / GPU-индексе
#   3. LLM Selector выбирает лучшего кандидата (1-5 или 0)
#   4. Поиск кода работы по справочнику

import os
import re
import numpy as np

# Принудительно offline — не лезем в HuggingFace Hub
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from qdrant_client import QdrantClient
from tqdm import tqdm


SELECTOR_SYSTEM_PROMPT = """Ты — эксперт по строительной документации и сметам.

Тебе дан запрос (наименование работы из сметы) и список кандидатов из справочника ВиКР.
Каждый кандидат имеет: номер, код, наименование, единицу измерения и состав работ.

Задача: выбери ОДИН наиболее подходящий кандидат.

Правила:
- Сравнивай смысл работы, не только слова
- Тип работы не должен отличаться(буронабивные останутся буронабивными и не могут быть забивными)
- Учитывай единицу измерения — она может соответствовать
- Состав работ помогает понять, что именно включает ВиКР
- Если ни один кандидат не подходит, ответь "0"

Отвечай ТОЛЬКО номером кандидата (1-5) или 0. Без пояснений."""


class Matcher:
    PARSE_EXTRA_COLUMNS = [
        "Наименование ВиКР",
        "Марка РД",
        "Номер проекта",
        "Наименование зданий, сооружений, систем и установок",
        "structure",
    ]

    def __init__(
        self,
        embedder_path: str,
        selector_path: str,
        selector_base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "vikr",
        top_k: int = 8,
        device: Optional[str] = None,
        progress_callback=None,
        use_gpu_search: bool = True,
        batch_size: int = 16,
    ):
        self.progress_callback = progress_callback or (lambda c, t, s: None)
        self.top_k = top_k
        self.default_batch_size = batch_size
        self.use_gpu_search = use_gpu_search

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # === 1. Bi-encoder (BERTA) ===
        print("🔍 Загрузка BERTA embedder...")
        self.embedder = SentenceTransformer(embedder_path, device=device)
        print(f"   ✅ BERTA загружена (dim={self.embedder.get_sentence_embedding_dimension()})")

        # === 2. LLM Selector (base + LoRA, bf16) ===
        print(f"🤖 Загрузка LLM Selector (bf16 + LoRA)...")
        print(f"   Базовая модель: {selector_base_model}")
        print(f"   LoRA адаптер: {selector_path}")

        hug_cache = str(Path(selector_path).parent.parent / "hug_models")
        print(f"   Кэш моделей: {hug_cache}")
        os.environ["HF_HOME"] = hug_cache

        # Резолвим HF-имя в локальный snapshot, чтобы transformers не лез в сеть
        if not Path(selector_base_model).exists():
            model_dir_name = "models--" + selector_base_model.replace("/", "--")
            snapshot_dir = Path(hug_cache) / model_dir_name / "snapshots"
            if snapshot_dir.exists():
                snapshots = [d for d in snapshot_dir.iterdir() if d.is_dir()]
                if snapshots:
                    selector_base_model = str(snapshots[0])
                    print(f"   📦 Резолвлено в snapshot: {selector_base_model}")
                else:
                    raise FileNotFoundError(f"❌ Нет snapshots в {snapshot_dir}")
            else:
                raise FileNotFoundError(
                    f"❌ Модель не найдена в кэше: {snapshot_dir}\n"
                    f"   Скачайте её локально или укажите путь."
                )

        self.selector_tokenizer = AutoTokenizer.from_pretrained(
            selector_base_model,
            cache_dir=hug_cache,
            local_files_only=True,
            trust_remote_code=False,
        )
        if self.selector_tokenizer.pad_token is None:
            self.selector_tokenizer.pad_token = self.selector_tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            selector_base_model,
            cache_dir=hug_cache,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.selector_model = PeftModel.from_pretrained(base_model, selector_path)
        self.selector_model.eval()
        print("   ✅ LLM Selector загружен (bf16 + LoRA)")

        # === 3. Qdrant ===
        print("🔌 Подключение к Qdrant...")
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name

        info = self.qdrant_client.get_collection(collection_name)
        print(f"   ✅ Коллекция '{collection_name}': {info.points_count} точек")

        # === 4. GPU-индекс ===
        self._gpu_vectors = None
        self._gpu_payloads = None
        self._gpu_codes = None

        if use_gpu_search:
            self._build_gpu_index()

        # === 5. Хэш-маппинг кодов ===
        self._code_lookup = None

        print("✅ Matcher инициализирован")

    def _build_gpu_index(self):
        """Загружает все векторы из Qdrant в GPU."""
        print("🚀 Загрузка векторов в GPU...")
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        total = collection_info.points_count

        if total == 0:
            self.use_gpu_search = False
            return

        all_vectors, all_payloads, all_codes = [], [], []
        offset = None

        while True:
            points, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000, offset=offset,
                with_payload=True, with_vectors=True
            )
            if not points:
                break

            for p in points:
                vec = p.vector
                if isinstance(vec, dict):
                    vec = list(vec.values())[0]
                all_vectors.append(vec)
                all_payloads.append(p.payload)
                all_codes.append(p.payload.get("kod_raboty", ""))

            if next_offset is None:
                break
            offset = next_offset

        vectors_np = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        vectors_np = vectors_np / np.maximum(norms, 1e-8)

        self._gpu_vectors = torch.from_numpy(vectors_np).to(self._device)
        self._gpu_payloads = all_payloads
        self._gpu_codes = all_codes

        mem_mb = self._gpu_vectors.element_size() * self._gpu_vectors.nelement() / 1024 / 1024
        print(f"   ✅ GPU-индекс: {len(all_payloads)} векторов, {mem_mb:.1f} MB")

    def _prepare_code_lookup(self, codes_df: pd.DataFrame):
        if self._code_lookup is not None:
            return

        self._code_lookup = {}
        for _, row in codes_df.iterrows():
            name = str(row.get("Наименование", "")).strip().lower()
            unit = str(row.get("Единица измерения", "")).strip().lower()
            code = str(row.get("Код работы", "")).strip()
            if name and code:
                self._code_lookup[(name, unit)] = code

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        """Нормализация единицы измерения для сравнения."""
        if unit is None:
            return ""

        unit = str(unit).strip().lower().replace("ё", "е")
        unit = re.sub(r"\s+", " ", unit)

        # Убираем точку в конце у 'шт.'
        if unit.rstrip(".") == "шт":
            unit = "шт"

        return unit

    def _build_error_message(
        self,
        chosen_text: Optional[str],
        vr_unit: str,
        embedding_score: Union[float, int, None],
    ) -> str:
        errors = []

        # 1. Работа не выбрана
        if not chosen_text:
            errors.append("Работа не выбрана")

        # 2. Низкая уверенность модели
        try:
            score_val = float(embedding_score)
            if score_val < 0.5:
                errors.append("Низкая уверенность модели")
        except Exception:
            pass

        # 3. Не совпадает единица измерения
        if chosen_text:
            parts = [p.strip() for p in str(chosen_text).split("|", 2)]
            matched_unit = parts[1] if len(parts) > 1 else ""

            if self._normalize_unit(vr_unit) != self._normalize_unit(matched_unit):
                errors.append("Единица измерения не совпадает")

        return "; ".join(errors)

    def search_candidates_batch(self, queries: List[str]) -> List[List[Dict]]:
        """Батчевый поиск кандидатов."""
        embeddings = self.embedder.encode(
            queries, convert_to_tensor=True,
            show_progress_bar=False, normalize_embeddings=True,
        )

        if self.use_gpu_search and self._gpu_vectors is not None:
            query_emb = embeddings.to(self._device).float()
            scores = torch.mm(query_emb, self._gpu_vectors.T)
            top_scores, top_indices = torch.topk(scores, k=min(self.top_k, scores.shape[1]), dim=1)

            results = []
            for i in range(len(queries)):
                candidates = []
                for j in range(top_scores.shape[1]):
                    idx = int(top_indices[i, j].item())
                    payload = self._gpu_payloads[idx]
                    candidates.append({
                        "text": self._format_candidate_text(payload),
                        "payload": payload,
                        "code": self._gpu_codes[idx],
                        "embedding_score": float(top_scores[i, j].item()),
                    })
                results.append(candidates)
            return results

        # Qdrant fallback
        results = []
        embeddings_np = embeddings.cpu().numpy()
        for emb in embeddings_np:
            hits = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=emb.tolist(), limit=self.top_k,
            ).points
            candidates = []
            for hit in hits:
                candidates.append({
                    "text": self._format_candidate_text(hit.payload),
                    "payload": hit.payload,
                    "code": str(hit.payload.get("kod_raboty", "")),
                    "embedding_score": float(hit.score),
                })
            results.append(candidates)
        return results

    def _format_candidate_text(self, payload: dict) -> str:
        name = str(payload.get("naименование", "") or "")
        ed_izm = str(payload.get("ed_izm", "") or "")
        sostav = str(payload.get("sostav_rabot", "") or "")
        if len(sostav) > 300:
            sostav = sostav[:300] + "..."
        return f"{name} | {ed_izm} | {sostav}"

    @staticmethod
    def _format_candidate_for_selector(num: int, payload: dict, score: float) -> str:
        name = payload.get("naименование", "")
        code = payload.get("kod_raboty", "")
        ed_izm = payload.get("ed_izm", "")
        sostav = payload.get("sostav_rabot", "")
        if len(sostav) > 300:
            sostav = sostav[:300] + "..."
        return (
            f"Кандидат {num}:\n"
            f"  Код: {code}\n"
            f"  Наименование: {name}\n"
            f"  Ед.изм.: {ed_izm}\n"
            f"  Состав работ: {sostav}\n"
            f"  Сходство: {score:.4f}"
        )

    def _llm_select(self, query: str, candidates: List[Dict]) -> int:
        """Вызывает LLM selector, возвращает номер кандидата (1-N) или 0."""
        candidates_text = []
        for i, c in enumerate(candidates, 1):
            candidates_text.append(
                self._format_candidate_for_selector(i, c["payload"], c["embedding_score"])
            )

        cand_block = "\n\n".join(candidates_text)
        user_msg = f"Запрос: {query}\n\nКандидаты:\n\n{cand_block}\n\nКакой кандидат лучше всего подходит?"

        messages = [
            {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        input_text = self.selector_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.selector_tokenizer(input_text, return_tensors="pt").to(self.selector_model.device)

        with torch.no_grad():
            outputs = self.selector_model.generate(
                **inputs, max_new_tokens=8, do_sample=False, temperature=1.0,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.selector_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        match = re.search(r'[0-5]', answer)
        return int(match.group()) if match else 0

    def select_batch(self, queries: List[str], candidates_batch: List[List[Dict]]) -> List[Dict]:
        """LLM Selector для батча запросов."""
        results = []

        for query, candidates in zip(queries, candidates_batch):
            if not candidates:
                results.append({
                    "chosen_text": None, "chosen_code": "",
                    "chosen_index": -1, "selector_choice": 0,
                    "embedding_score": 0.0,
                })
                continue

            selected_num = self._llm_select(query, candidates)

            if 1 <= selected_num <= len(candidates):
                best = candidates[selected_num - 1]
                results.append({
                    "chosen_text": best["text"],
                    "chosen_code": best["code"],
                    "chosen_index": selected_num - 1,
                    "selector_choice": selected_num,
                    "embedding_score": best["embedding_score"],
                })
            else:
                results.append({
                    "chosen_text": None, "chosen_code": "",
                    "chosen_index": -1, "selector_choice": 0,
                    "embedding_score": candidates[0]["embedding_score"] if candidates else 0.0,
                })

        return results

    def find_work_code(self, matched_text: Optional[str], unit: str, codes_df: pd.DataFrame) -> Optional[str]:
        if not matched_text:
            return None

        self._prepare_code_lookup(codes_df)

        parts = [p.strip() for p in matched_text.split("|", 2)]
        name = parts[0]
        vikr_unit = parts[1] if len(parts) > 1 else str(unit).strip()

        key = (name.lower(), vikr_unit.lower())
        if key in self._code_lookup:
            return self._code_lookup[key]

        # Partial match fallback
        for (k_name, k_unit), code in self._code_lookup.items():
            if name.lower()[:20] in k_name:
                return code

        return None

    @staticmethod
    def _fmt_num(val) -> str:
        """Форматирует № п/п: 1.0 → '1', сохраняя строки и нечисловое."""
        if val is None:
            return ""
        if isinstance(val, float):
            if pd.isna(val):
                return ""
            if val.is_integer():
                return str(int(val))
            return str(val)
        if isinstance(val, int):
            return str(val)
        s = str(val).strip()
        if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
            return s[:-2]
        return s

    def _extract_passthrough_fields(self, row: pd.Series) -> Dict[str, str]:
        return {
            "Наименование ВиКР": str(row.get("Наименование ВиКР", "") or ""),
            "Наименование работ": str(
                row.get("Наименование работ", "") or row.get("Наименование", "") or ""
            ),
            "Марка РД": str(row.get("Марка РД", "") or ""),
            "Номер проекта": str(row.get("Номер проекта", "") or ""),
            "Наименование зданий, сооружений, систем и установок": str(
                row.get("Наименование зданий, сооружений, систем и установок", "") or ""
            ),
            "structure": str(row.get("structure", "") or ""),
        }

    def load_vr_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        df = pd.read_excel(file_path)

        # НЕ фильтруем — оставляем и работы и материалы в исходном порядке

        df['Кол-во'] = (
            df['Кол-во'].astype(str)
            .str.replace(' ', '').str.replace(',', '.')
            .replace({'nan': '0', '-': '0', '': '0', 'None': '0'})
        )
        df['Кол-во'] = pd.to_numeric(df['Кол-во'], errors='coerce').fillna(0)

        # Нормализуем имя колонки раздела → "section"
        if "Раздел" in df.columns and "section" not in df.columns:
            df["section"] = df["Раздел"]
        if "Подраздел" in df.columns and "subsection" not in df.columns:
            df["subsection"] = df["Подраздел"]
        if "Наименование" not in df.columns and "Наименование работ" in df.columns:
            df["Наименование"] = df["Наименование работ"]
        if "Наименование работ" not in df.columns and "Наименование" in df.columns:
            df["Наименование работ"] = df["Наименование"]

        for col in ["Ед. изм.", "section", "subsection", "Наименование", "Тип", *self.PARSE_EXTRA_COLUMNS]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str).str.strip()

        df = df.dropna(subset=["Наименование"])
        work_count = (df["Тип"] == "work").sum()
        print(f"📋 Загружено {len(df)} строк ({work_count} работ, {len(df) - work_count} материалов)")
        return df

    def match_dataframe(
        self, vr_df: pd.DataFrame,
        codes_df: Optional[pd.DataFrame] = None,
        batch_size: int = None,
    ) -> pd.DataFrame:
        batch_size = batch_size or self.default_batch_size
        total = len(vr_df)

        if "Кол-во" not in vr_df.columns:
            vr_df = vr_df.copy()
            vr_df["Кол-во"] = ""

        if codes_df is not None:
            self._prepare_code_lookup(codes_df)

        work_count = (vr_df["Тип"] == "work").sum()
        print(f"🔍 Матчинг: {total} строк ({work_count} работ), batch={batch_size}")
        results = []

        # Разделяем на работы и материалы, сохраняя индексы
        work_indices = []
        material_rows = []

        for idx, (_, row) in enumerate(vr_df.iterrows()):
            if str(row.get("Тип", "")).strip() == "work":
                work_indices.append(idx)
            else:
                material_rows.append((idx, {
                    "№ п/п": self._fmt_num(row.get("№ п/п", "")),
                    "Наименование ВР": str(row.get("Наименование", "")),
                    "Ед. изм.": str(row.get("Ед. изм.", "")),
                    "Кол-во": row.get("Кол-во", ""),
                    "Раздел": str(row.get("section", "")),
                    "Подраздел": str(row.get("subsection", "")),
                    "Тип": str(row.get("Тип", "material")),
                    "Код работы": "",
                    "Смэтченная работа ВиКР": "",
                    "Позиция в топ-5": "",
                    "Уверенность модели": "",
                    "Score эмбеддинга": "",
                    "Ошибки": "",
                    **self._extract_passthrough_fields(row),
                }))

        # Матчим только работы
        work_df = vr_df.iloc[work_indices].reset_index(drop=True)
        work_results = {}

        for start in tqdm(range(0, len(work_df), batch_size), desc="Матчинг"):
            end = min(start + batch_size, len(work_df))
            batch = work_df.iloc[start:end]

            queries = []
            for _, row in batch.iterrows():
                name = str(row.get("Наименование", "") or "").strip()
                parts = [name] if name else [""]
                unit = str(row.get("Ед. изм.", "") or "").strip()
                sec = str(row.get("section", "") or "").strip()
                if unit and unit.lower() not in ("", "nan"):
                    parts.append(unit)
                if sec and sec.lower() not in ("", "nan"):
                    parts.append(sec)
                queries.append(" | ".join(parts))

            candidates_batch = self.search_candidates_batch(queries)
            rerank_results = self.select_batch(queries, candidates_batch)

            for i, ((_, row), rerank, candidates) in enumerate(
                zip(batch.iterrows(), rerank_results, candidates_batch)
            ):
                code = rerank["chosen_code"]
                if not code and codes_df is not None:
                    code = self.find_work_code(
                        rerank["chosen_text"],
                        str(row.get("Ед. изм.", "") or ""),
                        codes_df
                    ) or ""

                error_text = self._build_error_message(
                    chosen_text=rerank["chosen_text"],
                    vr_unit=str(row.get("Ед. изм.", "") or ""),
                    embedding_score=rerank["embedding_score"],
                )

                global_idx = work_indices[start + i]
                work_results[global_idx] = {
                    "№ п/п": self._fmt_num(row.get("№ п/п", "")),
                    "Наименование ВР": str(row.get("Наименование", "")),
                    "Ед. изм.": str(row.get("Ед. изм.", "")),
                    "Кол-во": row.get("Кол-во", ""),
                    "Раздел": str(row.get("section", "")),
                    "Подраздел": str(row.get("subsection", "")),
                    "Тип": "work",
                    "Код работы": code,
                    "Смэтченная работа ВиКР": rerank["chosen_text"] or "",
                    "Позиция в топ-5": rerank["selector_choice"],
                    "Уверенность модели": rerank["embedding_score"],
                    "Score эмбеддинга": rerank["embedding_score"],
                    "Ошибки": error_text,
                    **self._extract_passthrough_fields(row),
                }

            if self.progress_callback:
                self.progress_callback(end, len(work_df), "🔍 Матчинг")

        # Собираем всё в исходном порядке
        all_results = {}
        all_results.update(work_results)
        for idx, row_data in material_rows:
            all_results[idx] = row_data

        final = [all_results[i] for i in sorted(all_results.keys())]
        return pd.DataFrame(final)

    def cleanup(self):
        try:
            del self.embedder
            del self.selector_model
            del self.selector_tokenizer
            if self._gpu_vectors is not None:
                del self._gpu_vectors
            torch.cuda.empty_cache()
            print("🧹 Модели выгружены из VRAM")
        except Exception as e:
            print(f"⚠️ Ошибка очистки: {e}")

    @staticmethod
    def analyze_results(results_df: pd.DataFrame):
        total = len(results_df)
        if total == 0:
            return

        matched = results_df[results_df["Код работы"] != ""]
        print(f"\n{'='*60}")
        print(f"📊 СТАТИСТИКА")
        print(f"{'='*60}")
        print(f"  Всего: {total}")
        print(f"  С кодом: {len(matched)} ({len(matched)/total*100:.1f}%)")

        emb = pd.to_numeric(results_df["Score эмбеддинга"], errors="coerce")
        if emb.notna().any():
            print(f"  Средний emb score: {emb.mean():.3f}")
        print(f"{'='*60}")
