# config.py — Конфигурация пайплайна с новыми моделями
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def env_path(name: str, default: Path | str) -> str:
    """Возвращает путь из переменной окружения или локальный дефолт."""
    return str(Path(os.getenv(name, str(default))).expanduser())


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

# ── Директории результатов ─────────────────────────────────────────────
PARSE_DIR = ROOT / "parse_files"
STRUCT_DIR = ROOT / "struct_files"
RESULT_DIR = ROOT / "matched_files"
FINAL_DIR = ROOT / "final_files"
LOG_DIR = ROOT / "logs"

for d in [PARSE_DIR, STRUCT_DIR, RESULT_DIR, FINAL_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

# ── Пути к моделям ─────────────────────────────────────────────────────
MODELS = {
    # Bi-encoder (BERTA Optuna best)
    "embedder_path": env_path("GRAPHIT_EMBEDDER_PATH", ROOT.parent / "augmentation" / "berta_finetuned_v6_second" / "final"),

    # LLM Selector (Qwen2.5-7B + LoRA adapter, bf16)
    "selector_path": env_path("GRAPHIT_SELECTOR_PATH", ROOT.parent / "train_selector" / "selector_finetuned_7b_v6" / "final"),
    "selector_base_model": os.getenv("GRAPHIT_SELECTOR_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),

    # Классификатор Материал/Работа (rubert-tiny2)
    "classifier_path": env_path("GRAPHIT_CLASSIFIER_PATH", ROOT.parent / "train_classification" / "classifier_llm_v7" / "final"),

    # Справочник кодов ВиКР
    "codes_file": env_path("GRAPHIT_CODES_FILE", ROOT / "docs" / "vikr_full.xlsx"),


}

# ── Qdrant ──────────────────────────────────────────────────────────────
QDRANT = {
    "host": os.getenv("GRAPHIT_QDRANT_HOST", "localhost"),
    "port": env_int("GRAPHIT_QDRANT_PORT", 6333),
    "collection_name": os.getenv("GRAPHIT_QDRANT_COLLECTION", "vikr_v6"),
}

# ── OCR / LLM ──────────────────────────────────────────────────────────
OLLAMA = {
    # OCR через Ollama
    "vllm_url": os.getenv("GRAPHIT_OLLAMA_URL", "http://localhost:11434"),
    "ocr_model": os.getenv("GRAPHIT_OCR_MODEL", "qwen3.5:27b"),
    # Структурирование (классификация работ/материалов)
    "merge_model": os.getenv("GRAPHIT_MERGE_MODEL", "gemma3:4b-it-fp16"),
    "poppler_path": os.getenv("GRAPHIT_POPPLER_PATH") or None,  # None = системный poppler из PATH
    "max_parallel_ocr": env_int("GRAPHIT_MAX_PARALLEL_OCR", 1),
    "fast_preprocess": env_bool("GRAPHIT_FAST_PREPROCESS", True),
    "ocr_dpi": env_int("GRAPHIT_OCR_DPI", 250),
    "ocr_num_predict": env_int("GRAPHIT_OCR_NUM_PREDICT", 4096),
}

# ── Матчинг ─────────────────────────────────────────────────────────────
MATCHING = {
    "top_k": env_int("GRAPHIT_TOP_K", 5),
    "batch_size": env_int("GRAPHIT_BATCH_SIZE", 16),
    "use_gpu_search": env_bool("GRAPHIT_USE_GPU_SEARCH", True),
}
