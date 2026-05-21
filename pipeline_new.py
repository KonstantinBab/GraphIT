# pipeline_new.py — Обновлённый пайплайн с локальной QLoRA-моделью
#
# Этапы:
#   1. Парсинг PDF (OCR через Ollama) — parse_class_p.py
#   2. Структурирование + Классификация (GESNClassifier с Qwen2.5-1.5B-QLoRA)
#   3. Матчинг (BERT embedder + LLM Selector + Qdrant)
#   4. Агрегация (группировка по кодам ВиКР)

import pandas as pd
from pathlib import Path
from datetime import datetime
from config import PARSE_DIR, STRUCT_DIR, RESULT_DIR, FINAL_DIR, MODELS, QDRANT, OLLAMA, MATCHING


class Pipeline:
    def __init__(self, log_callback=None):
        self.log = log_callback or print
        self.progress_callback = None

    def _make_output_path(self, stage: str, input_path: Path) -> Path:
        """Генерирует путь для выходного файла этапа."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = input_path.stem
        dirs = {
            "parse": PARSE_DIR,
            "struct": STRUCT_DIR,
            "match": RESULT_DIR,
            "aggregate": FINAL_DIR
        }
        return dirs[stage] / f"{stem}_{stage}_{ts}.xlsx"

    # ══════════════════════════════════════════════════════════════
    # STAGE 1: PARSE (PDF → raw Excel)
    # ══════════════════════════════════════════════════════════════
    def stage_parse(self, pdf_path: str, **kwargs) -> Path:
        """
        Этап 1: Парсинг PDF через OCR.
        Возвращает путь к raw Excel-файлу.
        """
        from parse_class import ConstructionBillOCR

        pdf_path = Path(pdf_path)
        out_path = self._make_output_path("parse", pdf_path)

        self.log(f"📄 Этап 1: Парсинг PDF → {out_path.name}")

        ocr = ConstructionBillOCR(
            pdf_path=str(pdf_path),
            poppler_path=kwargs.get("poppler_path", OLLAMA.get("poppler_path")),
            vllm_url=kwargs.get("vllm_url", OLLAMA.get("vllm_url", "http://localhost:8080")),
            model_name=kwargs.get("ocr_model", OLLAMA.get("ocr_model", "glm-ocr")),
            dpi=OLLAMA.get("ocr_dpi", 400),
            preprocess_images=True,
            progress_callback=self.progress_callback,
            max_parallel_ocr=OLLAMA.get("max_parallel_ocr", 2),
            fast_preprocess=OLLAMA.get("fast_preprocess", True),
            max_tokens=OLLAMA.get("ocr_num_predict", 4096),
        )

        df = ocr.process()
        ocr.save_to_excel(str(out_path))
        self.log(f"   ✅ Извлечено {len(df)} строк")
        return out_path

    # ══════════════════════════════════════════════════════════════
    # STAGE 2: STRUCTURE + CLASSIFY (raw → structured with type)
    # ══════════════════════════════════════════════════════════════
    def stage_struct(self, input_xlsx: str, **kwargs) -> Path:
        """
        Этап 2: Классификация Работа/Материал через локальную QLoRA-модель.
        Возвращает путь к структурированному Excel-файлу.
        """
        from struct_class import GESNClassifier

        inp_path = Path(input_xlsx)
        out_path = self._make_output_path("struct", inp_path)

        self.log(f"🔧 Этап 2: Классификация → {out_path.name}")

        # Загружаем входные данные
        df = GESNClassifier.load_from_excel(str(inp_path))
        self.log(f"   📊 Загружено {len(df)} строк")

        # Путь к модели классификатора
        classifier_path = kwargs.get("classifier_path", MODELS.get("classifier_path"))

        if classifier_path and Path(classifier_path).exists():
            self.log(f"   🏷️  Загрузка модели: {classifier_path}")

            # Инициализация классификатора
            cls = GESNClassifier(
                model_path=classifier_path,
                base_model=kwargs.get("base_model", "Qwen/Qwen2.5-3B-Instruct"),
                max_workers=kwargs.get("max_workers", 4),
                use_4bit=kwargs.get("use_4bit", True),
                progress_callback=self.progress_callback,
            )

            # Обработка
            df_out = cls.process(df)

            # Статистика
            work_count = (df_out["Тип"] == "work").sum()
            mat_count = (df_out["Тип"] == "material").sum()
            self.log(f"   ✅ Работ: {work_count}, Материалов: {mat_count}")

            # Очистка VRAM перед следующим этапом
            cls.cleanup()
        else:
            self.log(f"   ⚠️  Классификатор не найден: {classifier_path}")
            self.log(f"   ⚠️  Пропускаем классификацию, сохраняем как есть")
            df_out = df
            # Добавляем пустую колонку Тип для совместимости
            if 'Тип' not in df_out.columns:
                df_out['Тип'] = ''

        # Сохранение результата
        GESNClassifier.save_to_excel(df_out, str(out_path))
        self.log(f"   ✅ Сохранено: {len(df_out)} строк")
        return out_path

    # ══════════════════════════════════════════════════════════════
    # STAGE 3: MATCH (structured → matched via BERTA + LLM Selector)
    # ══════════════════════════════════════════════════════════════
    def stage_match(self, input_xlsx: str, **kwargs) -> Path:
        """
        Этап 3: Матчинг позиций с кодами ВиКР.
        Возвращает путь к файлу с результатами матчинга.
        """
        from matcher_new import Matcher

        inp_path = Path(input_xlsx)
        out_path = self._make_output_path("match", inp_path)

        self.log(f"🔍 Этап 3: Матчинг → {out_path.name}")

        # Загружаем справочник кодов
        codes_file = kwargs.get("codes_file", MODELS.get("codes_file"))
        if not codes_file or not Path(codes_file).exists():
            self.log(f"   ❌ Файл кодов не найден: {codes_file}")
            raise FileNotFoundError(f"Файл кодов не найден: {codes_file}")

        codes_df = pd.read_excel(codes_file)
        self.log(f"   📚 Кодов ВиКР загружено: {len(codes_df)}")

        # Инициализация матчера
        matcher = Matcher(
            embedder_path=kwargs.get("embedder_path", MODELS.get("embedder_path")),
            selector_path=kwargs.get("selector_path", MODELS.get("selector_path")),
            selector_base_model=kwargs.get("selector_base_model", MODELS.get("selector_base_model", "Qwen/Qwen2.5-7B-Instruct")),
            qdrant_host=QDRANT.get("host", "localhost"),
            qdrant_port=QDRANT.get("port", 6333),
            collection_name=kwargs.get("collection_name", QDRANT.get("collection_name", "gesn_codes")),
            top_k=MATCHING.get("top_k", 5),
            use_gpu_search=MATCHING.get("use_gpu_search", False),
            batch_size=MATCHING.get("batch_size", 32),
            progress_callback=self.progress_callback,
        )

        # Загрузка и матчинг
        vr_df = matcher.load_vr_data(inp_path)
        self.log(f"   📊 Позиций для матчинга: {len(vr_df)}")

        results_df = matcher.match_dataframe(vr_df, codes_df)

        # Сохраняем с заливкой и числовым форматом Кол-во
        from aggregator_class import FinalAggregator
        FinalAggregator.save_df_to_excel_with_highlights(results_df, out_path)

        matcher.analyze_results(results_df)
        matcher.cleanup()

        self.log(f"   ✅ Смэтчено: {len(results_df)} работ")
        return out_path

    # ══════════════════════════════════════════════════════════════
    # STAGE 4: AGGREGATE (matched → final grouped)
    # ══════════════════════════════════════════════════════════════
    def stage_aggregate(self, input_xlsx: str, **kwargs) -> Path:
        """
        Этап 4: Агрегация результатов по кодам ВиКР.
        Возвращает путь к финальному файлу.
        """
        from aggregator_class import FinalAggregator

        inp_path = Path(input_xlsx)
        out_path = self._make_output_path("aggregate", inp_path)

        self.log(f"📊 Этап 4: Агрегация → {out_path.name}")

        aggregator = FinalAggregator(
            progress_callback=self.progress_callback,
            log_callback=lambda msg: self.log(f"   {msg}"),
        )

        final_df = aggregator.aggregate(inp_path)
        aggregator.save_to_excel(final_df, out_path)

        self.log(f"   ✅ Агрегировано: {len(final_df)} уникальных кодов ВиКР")
        return out_path

    # ══════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ══════════════════════════════════════════════════════════════
    def run_full_pipeline(self, pdf_path: str, **kwargs) -> dict:
        """
        Запускает полный конвейер обработки:
        PDF → OCR → Классификация → Матчинг → Агрегация
        """
        self.log("🚀 Запуск полного конвейера...")
        self.log("=" * 70)

        try:
            # Этап 1: Парсинг
            raw_xlsx = self.stage_parse(pdf_path, **kwargs)
            self.log(f"   ✅ Этап 1 завершён: {raw_xlsx.name}\n")

            # Этап 2: Классификация
            struct_xlsx = self.stage_struct(raw_xlsx, **kwargs)
            self.log(f"   ✅ Этап 2 завершён: {struct_xlsx.name}\n")

            # Этап 3: Матчинг
            matched_xlsx = self.stage_match(struct_xlsx, **kwargs)
            self.log(f"   ✅ Этап 3 завершён: {matched_xlsx.name}\n")

            # Этап 4: Агрегация
            final_xlsx = self.stage_aggregate(matched_xlsx, **kwargs)
            self.log(f"   ✅ Этап 4 завершён: {final_xlsx.name}\n")

            self.log("=" * 70)
            self.log("🎉 Конвейер завершён!")
            self.log(f"📁 Финальный результат: {final_xlsx}")

            return {
                "stage1_parse": raw_xlsx,
                "stage2_struct": struct_xlsx,
                "stage3_match": matched_xlsx,
                "stage4_aggregate": final_xlsx,
            }

        except Exception as e:
            self.log(f"❌ Ошибка в конвейере: {e}")
            import traceback
            self.log(traceback.format_exc())
            raise

    # ══════════════════════════════════════════════════════════════
    # SINGLE STAGE RUNNERS (для отладки отдельных этапов)
    # ══════════════════════════════════════════════════════════════
    def run_stage_struct_only(self, input_xlsx: str, **kwargs) -> Path:
        """Запускает только этап классификации (для тестов)."""
        return self.stage_struct(input_xlsx, **kwargs)

    def run_stage_match_only(self, input_xlsx: str, **kwargs) -> Path:
        """Запускает только этап матчинга (для тестов)."""
        return self.stage_match(input_xlsx, **kwargs)


# =====================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =====================================================================
if __name__ == "__main__":
    import sys
    import json

    # Загрузка конфигурации (опционально)
    config_file = "config.json"
    config = {}
    if Path(config_file).exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"📋 Конфигурация загружена: {config_file}")

    # Путь к PDF (из аргументов или конфига)
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else config.get("input_pdf", "input.pdf")

    # Параметры классификатора
    classifier_params = {
        "classifier_path": config.get("models", {}).get("classifier_path", "classifier_llm/final"),
        "max_workers": config.get("classifier_workers", 4),
        "use_4bit": config.get("classifier_4bit", True),
    }

    print(f"\n🚀 Запуск пайплайна...")
    print(f"   PDF: {pdf_path}")
    print(f"   Классификатор: {classifier_params['classifier_path']}")
    print(f"   Workers: {classifier_params['max_workers']}\n")

    pipeline = Pipeline(log_callback=print)

    try:
        results = pipeline.run_full_pipeline(
            pdf_path,
            **classifier_params
        )

        print(f"\n✅ Все этапы завершены успешно!")
        print(f"📁 Финальный файл: {results['stage4_aggregate']}")

    except Exception as e:
        print(f"\n❌ Пайплайн прерван: {e}")
        sys.exit(1)