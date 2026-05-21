# app_gradio.py — Веб-интерфейс в стилистике ГрафИТ
#
# Корпоративный сине-белый дизайн, логотип, анимации
# Запуск: python app_gradio.py

import gradio as gr
import os
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import PARSE_DIR, STRUCT_DIR, RESULT_DIR, FINAL_DIR, MODELS, QDRANT, OLLAMA, MATCHING

# ── Логирование ─────────────────────────────────────────────────────────
LOG_LINES = []
TIMER_START = [None]  # mutable container for global start time
STAGE_TIMES = {}      # stage_name → elapsed seconds

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    elapsed = _format_elapsed()
    LOG_LINES.append(f"[{ts}] [{elapsed}] {msg}")
    if len(LOG_LINES) > 500:
        LOG_LINES.pop(0)

def _format_elapsed():
    """Возвращает время с начала процесса в формате MM:SS."""
    if TIMER_START[0] is None:
        return "00:00"
    import time
    elapsed = int(time.time() - TIMER_START[0])
    m, s = divmod(elapsed, 60)
    return f"{m:02d}:{s:02d}"

def get_log():
    return "\n".join(LOG_LINES[-80:])

def get_timer_display():
    """Возвращает строку с таймером и временем по этапам."""
    elapsed = _format_elapsed()
    parts = [f"⏱ Общее время: {elapsed}"]
    for stage, secs in STAGE_TIMES.items():
        m, s = divmod(int(secs), 60)
        parts.append(f"  {stage}: {m:02d}:{s:02d}")
    return "\n".join(parts)

def clear_log():
    import time
    LOG_LINES.clear()
    STAGE_TIMES.clear()
    TIMER_START[0] = time.time()
    return ""

# ── Этапы пайплайна ────────────────────────────────────────────────────
def run_stage(mode, input_file, progress=gr.Progress()):
    import time
    from pipeline_new import Pipeline

    if input_file is None:
        return None, "❌ Загрузите файл", get_log(), get_timer_display()

    clear_log()
    log(f"▶ Запуск: {mode}")

    pipeline = Pipeline(log_callback=log)
    pipeline.progress_callback = lambda c, t, s: progress(c / t if t > 0 else 0, desc=f"{s}: {c}/{t}")

    try:
        path = input_file.name if hasattr(input_file, 'name') else str(input_file)

        if mode == "full":
            t0 = time.time()
            r = pipeline.run_full_pipeline(path)
            STAGE_TIMES["Полный конвейер"] = time.time() - t0
            out = r["stage4"]
            status = f"✅ Конвейер завершён\n\n📄 Парсинг: {r['stage1'].name}\n🔧 Структура: {r['stage2'].name}\n🔍 Матчинг: {r['stage3'].name}\n📊 Результат: {r['stage4'].name}"
        elif mode == "parse":
            t0 = time.time()
            out = pipeline.stage_parse(path)
            STAGE_TIMES["Парсинг"] = time.time() - t0
            status = f"✅ Парсинг: {out.name}"
        elif mode == "struct":
            t0 = time.time()
            out = pipeline.stage_struct(path)
            STAGE_TIMES["Структурирование"] = time.time() - t0
            status = f"✅ Структурирование: {out.name}"
        elif mode == "match":
            t0 = time.time()
            out = pipeline.stage_match(path)
            STAGE_TIMES["Матчинг"] = time.time() - t0
            status = f"✅ Матчинг: {out.name}"
        elif mode == "aggregate":
            t0 = time.time()
            out = pipeline.stage_aggregate(path)
            STAGE_TIMES["Агрегация"] = time.time() - t0
            status = f"✅ Агрегация: {out.name}"
        else:
            return None, f"❌ Неизвестный режим: {mode}", get_log(), get_timer_display()

        log(status)
        return str(out), status, get_log(), get_timer_display()

    except Exception as e:
        log(f"❌ {e}")
        log(traceback.format_exc())
        return None, f"❌ Ошибка: {e}", get_log(), get_timer_display()


def run_full(f, p=gr.Progress()): return run_stage("full", f, p)
def run_parse(f, p=gr.Progress()): return run_stage("parse", f, p)
def run_struct(f, p=gr.Progress()): return run_stage("struct", f, p)
def run_match(f, p=gr.Progress()): return run_stage("match", f, p)
def run_aggregate(f, p=gr.Progress()): return run_stage("aggregate", f, p)


# ── Интерактивный поиск ─────────────────────────────────────────────────
_matcher = None

def init_search():
    global _matcher
    if _matcher: return "✅ Модели загружены"
    try:
        from matcher_new import Matcher
        _matcher = Matcher(
            embedder_path=MODELS["embedder_path"],
            selector_path=MODELS["selector_path"],
            selector_base_model=MODELS.get("selector_base_model", "Qwen/Qwen2.5-7B-Instruct"),
            qdrant_host=QDRANT["host"], qdrant_port=QDRANT["port"],
            collection_name=QDRANT["collection_name"],
            top_k=MATCHING["top_k"], use_gpu_search=MATCHING["use_gpu_search"],
        )
        return "✅ BERTA + LLM Selector загружены"
    except Exception as e:
        return f"❌ {e}"

def search(query):
    if not _matcher: return "⚠ Сначала загрузите модели"
    if not query.strip(): return ""

    cands = _matcher.search_candidates_batch([query])[0]
    rr = _matcher.select_batch([query], [cands])[0]

    out = [f"🔎 {query}\n"]
    out.append(f"✅ Выбор: [{rr['selector_choice']}]  Код: {rr['chosen_code']}")
    out.append(f"   {(rr['chosen_text'] or '')[:120]}\n")
    out.append("─" * 60)
    for i, c in enumerate(cands, 1):
        m = " ◄" if i == rr['chosen_index'] + 1 else ""
        out.append(f"  [{i}] emb={c['embedding_score']:.4f} | {c['code']}{m}")
        out.append(f"      {c['text'][:100]}")
    return "\n".join(out)


# ── CSS ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Commissioner:wght@300;400;500;600;700&display=swap');

:root {
    --gp-blue: #0078D4;
    --gp-blue-dark: #005A9E;
    --gp-blue-deeper: #003D6B;
    --gp-blue-light: #E8F4FD;
    --gp-white: #FFFFFF;
    --gp-gray-50: #F8FAFB;
    --gp-gray-100: #EEF2F6;
    --gp-gray-200: #D5DDE5;
    --gp-gray-500: #6B7A8D;
    --gp-gray-700: #374151;
    --gp-gray-900: #111827;
    --gp-accent: #00A3E0;
    --gp-success: #10B981;
    --gp-error: #EF4444;
}

.gradio-container {
    max-width: 1280px !important;
    font-family: 'Commissioner', sans-serif !important;
    background: var(--gp-gray-50) !important;
}

/* ═══ HEADER ═══ */
.gp-header {
    background: linear-gradient(135deg, var(--gp-blue-deeper) 0%, var(--gp-blue-dark) 40%, var(--gp-blue) 100%);
    border-radius: 0 0 20px 20px;
    padding: 32px 40px;
    margin: -16px -16px 24px -16px;
    position: relative;
    overflow: hidden;
    animation: headerSlide 0.6s ease-out;
}
@keyframes headerSlide {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
.gp-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.gp-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,163,224,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.gp-header .logo-row {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
    position: relative;
    z-index: 1;
}
.gp-header .logo-row img {
    height: 36px;
    filter: brightness(0) invert(1);
}
.gp-header .brand-mark {
    color: white;
    font-size: 24px;
    font-weight: 700;
    line-height: 1;
}
.gp-header .logo-text {
    color: rgba(255,255,255,0.7);
    font-size: 13px;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.gp-header h1 {
    color: white !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    position: relative;
    z-index: 1;
    letter-spacing: -0.3px;
}
.gp-header .subtitle {
    color: rgba(255,255,255,0.75);
    font-size: 14px;
    margin-top: 8px;
    position: relative;
    z-index: 1;
    font-weight: 300;
}
.gp-header .badges {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    position: relative;
    z-index: 1;
    flex-wrap: wrap;
}
.gp-header .badge {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(8px);
    color: rgba(255,255,255,0.9);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.15);
}

/* ═══ TABS ═══ */
.tab-nav {
    border-bottom: 2px solid var(--gp-gray-200) !important;
    margin-bottom: 4px !important;
}
.tab-nav button {
    font-family: 'Commissioner', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    color: var(--gp-gray-500) !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.25s !important;
    background: transparent !important;
}
.tab-nav button:hover {
    color: var(--gp-blue) !important;
    background: var(--gp-blue-light) !important;
}
.tab-nav button.selected {
    color: var(--gp-blue-dark) !important;
    border-bottom-color: var(--gp-blue) !important;
    background: transparent !important;
}

/* ═══ BUTTONS ═══ */
.gp-btn-primary {
    background: linear-gradient(135deg, var(--gp-blue) 0%, var(--gp-blue-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Commissioner', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 8px rgba(0, 120, 212, 0.25) !important;
}
.gp-btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0, 120, 212, 0.35) !important;
}

.gp-btn-secondary {
    background: white !important;
    color: var(--gp-blue) !important;
    border: 1.5px solid var(--gp-blue) !important;
    border-radius: 10px !important;
    font-family: 'Commissioner', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
.gp-btn-secondary:hover {
    background: var(--gp-blue-light) !important;
}

/* ═══ CARDS / ACCORDION ═══ */
.gp-card {
    background: white;
    border: 1px solid var(--gp-gray-200);
    border-radius: 14px;
    padding: 20px;
    transition: box-shadow 0.3s;
}
.gp-card:hover {
    box-shadow: 0 4px 16px rgba(0, 90, 158, 0.08);
}

/* ═══ LOG AREA ═══ */
.gp-log textarea {
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 12px !important;
    background: var(--gp-gray-900) !important;
    color: #A5D6FF !important;
    border: 1px solid #2D3748 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    line-height: 1.6 !important;
}

/* ═══ INPUTS ═══ */
.gradio-container input[type="text"],
.gradio-container textarea {
    font-family: 'Commissioner', sans-serif !important;
    border-radius: 8px !important;
    border: 1.5px solid var(--gp-gray-200) !important;
    transition: border-color 0.2s !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: var(--gp-blue) !important;
    box-shadow: 0 0 0 3px rgba(0, 120, 212, 0.1) !important;
}

/* ═══ STATUS ═══ */
.gp-status textarea {
    border-left: 4px solid var(--gp-blue) !important;
    background: var(--gp-blue-light) !important;
    border-radius: 0 8px 8px 0 !important;
    font-family: 'Commissioner', sans-serif !important;
    font-size: 13px !important;
    color: var(--gp-gray-700) !important;
}

/* ═══ TIMER ═══ */
.gp-timer textarea {
    font-family: 'JetBrains Mono', 'Cascadia Code', monospace !important;
    font-size: 13px !important;
    background: linear-gradient(135deg, #EEF6FC, #E0F0FA) !important;
    color: var(--gp-blue-dark) !important;
    border: 1.5px solid var(--gp-blue) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

/* ═══ FOOTER ═══ */
.gp-footer {
    text-align: center;
    padding: 24px 16px;
    color: var(--gp-gray-500);
    font-size: 12px;
    border-top: 1px solid var(--gp-gray-200);
    margin-top: 32px;
}
.gp-footer a {
    color: var(--gp-blue);
    text-decoration: none;
}

/* ═══ ANIMATIONS ═══ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.tabitem { animation: fadeInUp 0.35s ease-out; }
"""


# ── Интерфейс ───────────────────────────────────────────────────────────
def create_app():
    with gr.Blocks(title="ГрафИТ — ВР → ВиКР", css=CSS,
                   theme=gr.themes.Base(
                       primary_hue=gr.themes.colors.blue,
                       secondary_hue=gr.themes.colors.slate,
                       neutral_hue=gr.themes.colors.gray,
                       font=gr.themes.GoogleFont("Commissioner"),
                   )) as app:

        # ═══ HEADER ═══
        gr.HTML("""
        <div class="gp-header">
            <div class="logo-row">
                <span class="brand-mark">ГрафИТ</span>
                <span class="logo-text">Интеллектуальная обработка документов</span>
            </div>
            <h1>Конвейер обработки ВР → ВиКР</h1>
            <div class="subtitle">Автоматический подбор видов и комплексов работ по ведомостям объёмов работ</div>
            <div class="badges">
                <span class="badge">BERTA Embedder</span>
                <span class="badge">LLM Selector</span>
                <span class="badge">GLM-OCR</span>
                <span class="badge">Accuracy 89.1%</span>
            </div>
        </div>
        """)

        with gr.Tabs():

            # ═══ TAB 1: ПОЛНЫЙ КОНВЕЙЕР ═══
            with gr.Tab("🔄 Полный конвейер"):
                gr.Markdown("#### Загрузите PDF с ведомостью — система выполнит все этапы автоматически")
                with gr.Row():
                    with gr.Column(scale=2):
                        full_file = gr.File(label="Загрузите PDF", file_types=[".pdf"], height=120)
                        full_btn = gr.Button("▶  Запустить конвейер", elem_classes=["gp-btn-primary"], size="lg")
                    with gr.Column(scale=3):
                        full_timer = gr.Textbox(label="⏱ Таймер", lines=3, interactive=False, elem_classes=["gp-timer"])
                        full_status = gr.Textbox(label="Статус", lines=6, interactive=False, elem_classes=["gp-status"])
                        full_output = gr.File(label="Результат", interactive=False)
                full_log = gr.Textbox(label="Журнал", lines=10, interactive=False, elem_classes=["gp-log"])
                full_btn.click(run_full, [full_file], [full_output, full_status, full_log, full_timer])

            # ═══ TAB 2: ПОЭТАПНО ═══
            with gr.Tab("🔧 По этапам"):
                gr.Markdown("#### Запуск отдельных этапов обработки")
                stages_timer = gr.Textbox(label="⏱ Таймер", lines=3, interactive=False, elem_classes=["gp-timer"])

                with gr.Accordion("📄 Этап 1 — Парсинг PDF", open=False):
                    gr.Markdown("Извлечение таблиц из PDF через LLM-OCR")
                    p_file = gr.File(label="PDF", file_types=[".pdf"])
                    p_btn = gr.Button("▶  Парсинг", elem_classes=["gp-btn-primary"])
                    p_status = gr.Textbox(label="Статус", lines=2, interactive=False, elem_classes=["gp-status"])
                    p_out = gr.File(label="Результат", interactive=False)
                    p_btn.click(run_parse, [p_file], [p_out, p_status, gr.Textbox(visible=False), stages_timer])

                with gr.Accordion("🔧 Этап 2 — Структурирование", open=False):
                    gr.Markdown("Объединение строк + классификация Материал/Работа")
                    s_file = gr.File(label="Raw Excel", file_types=[".xlsx"])
                    s_btn = gr.Button("▶  Структурировать", elem_classes=["gp-btn-primary"])
                    s_status = gr.Textbox(label="Статус", lines=2, interactive=False, elem_classes=["gp-status"])
                    s_out = gr.File(label="Результат", interactive=False)
                    s_btn.click(run_struct, [s_file], [s_out, s_status, gr.Textbox(visible=False), stages_timer])

                with gr.Accordion("🔍 Этап 3 — Матчинг", open=True):
                    gr.Markdown("BERTA → Qdrant top-5 → LLM Selector")
                    m_file = gr.File(label="Structured Excel", file_types=[".xlsx"])
                    m_btn = gr.Button("▶  Матчинг", elem_classes=["gp-btn-primary"])
                    m_status = gr.Textbox(label="Статус", lines=2, interactive=False, elem_classes=["gp-status"])
                    m_out = gr.File(label="Результат", interactive=False)
                    m_btn.click(run_match, [m_file], [m_out, m_status, gr.Textbox(visible=False), stages_timer])

                with gr.Accordion("📊 Этап 4 — Агрегация", open=False):
                    gr.Markdown("Группировка работ по кодам ВиКР")
                    a_file = gr.File(label="Matched Excel", file_types=[".xlsx"])
                    a_btn = gr.Button("▶  Агрегация", elem_classes=["gp-btn-primary"])
                    a_status = gr.Textbox(label="Статус", lines=2, interactive=False, elem_classes=["gp-status"])
                    a_out = gr.File(label="Результат", interactive=False)
                    a_btn.click(run_aggregate, [a_file], [a_out, a_status, gr.Textbox(visible=False), stages_timer])

                stage_log = gr.Textbox(label="Журнал", lines=8, interactive=False,
                                       elem_classes=["gp-log"], value=get_log, every=3)

            # ═══ TAB 3: ПОИСК ═══
            with gr.Tab("🔎 Поиск ВиКР"):
                gr.Markdown("#### Интерактивная проверка матчинга в реальном времени")

                with gr.Row():
                    init_btn = gr.Button("⬇  Загрузить модели", elem_classes=["gp-btn-secondary"])
                    init_st = gr.Textbox(label="", lines=1, interactive=False, scale=3)
                init_btn.click(init_search, outputs=[init_st])

                with gr.Row():
                    q_input = gr.Textbox(
                        label="Наименование работы",
                        placeholder="Разработка грунта экскаватором | м3 | Земляные работы",
                        lines=2, scale=4,
                    )
                    q_btn = gr.Button("Найти", elem_classes=["gp-btn-primary"], scale=1)

                q_result = gr.Textbox(label="Результаты", lines=16, interactive=False, elem_classes=["gp-log"])
                q_btn.click(search, [q_input], [q_result])
                q_input.submit(search, [q_input], [q_result])

            # ═══ TAB 4: НАСТРОЙКИ ═══
            with gr.Tab("⚙ Конфигурация"):
                gr.Markdown("#### Текущие параметры системы")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Модели ML**")
                        gr.Textbox(label="Embedder (BERTA)", value=MODELS["embedder_path"], interactive=False)
                        gr.Textbox(label="Selector (LLM)", value=MODELS["selector_path"], interactive=False)
                        gr.Textbox(label="Classifier", value=MODELS.get("classifier_path", "—"), interactive=False)
                    with gr.Column():
                        gr.Markdown("**Инфраструктура**")
                        gr.Textbox(label="Qdrant", value=f"{QDRANT['host']}:{QDRANT['port']}", interactive=False)
                        gr.Textbox(label="Коллекция", value=QDRANT["collection_name"], interactive=False)
                        gr.Textbox(label="OCR Model", value=OLLAMA.get("ocr_model", "—"), interactive=False)

                gr.Markdown(f"""
                **Параметры матчинга:** Top-K = {MATCHING['top_k']} · Batch = {MATCHING['batch_size']} · GPU = {'✅' if MATCHING['use_gpu_search'] else '❌'}
                """)

        # ═══ FOOTER ═══
        gr.HTML("""
        <div class="gp-footer">
            <div style="margin-bottom:6px;">
                <strong>ГрафИТ</strong> · Конвейер ВР → ВиКР
            </div>
            BERTA (Optuna) + LLM Selector · Recall@5 92.9%<br>
            <span>ГрафИТ</span>
        </div>
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    server_name = os.getenv("GRAPHIT_HOST", "0.0.0.0")
    server_port = int(os.getenv("GRAPHIT_PORT", "7860"))
    app.launch(server_name=server_name, server_port=server_port, share=False, inbrowser=True)
