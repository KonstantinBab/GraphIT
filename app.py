# app.py — FastAPI-сервер (замена app_gradio.py)
#
# Запуск: python app.py

import os
import sys
import uuid
import time
import json
import asyncio
import traceback
import threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import uvicorn

from config import PARSE_DIR, STRUCT_DIR, RESULT_DIR, FINAL_DIR

# ── Хранилище файлов и сессий ────────────────────────────────────────
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# file_id → {"path": Path, "name": str}
FILE_REGISTRY: dict[str, dict] = {}

# run_id → {"logs": list, "progress": dict, "stages": list, "done": bool, "error": str|None, "result_id": str|None}
RUN_REGISTRY: dict[str, dict] = {}

app = FastAPI(title="ГрафИТ — ВР → ВиКР")


# ── API: загрузка файла ──────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    content = await file.read()
    save_path.write_bytes(content)
    FILE_REGISTRY[file_id] = {"path": save_path, "name": file.filename}
    return {"file_id": file_id, "name": file.filename}


# ── API: скачивание файла ────────────────────────────────────────────
@app.get("/api/download/{file_id}")
async def download_file(file_id: str):
    info = FILE_REGISTRY.get(file_id)
    if not info or not info["path"].exists():
        raise HTTPException(404, "File not found")
    return FileResponse(
        info["path"],
        filename=info["name"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── API: запуск пайплайна через SSE ──────────────────────────────────
def _register_output(path: Path) -> str:
    """Регистрирует выходной файл и возвращает file_id."""
    fid = uuid.uuid4().hex[:12]
    FILE_REGISTRY[fid] = {"path": path, "name": path.name}
    return fid


def _run_pipeline_thread(run_id: str, mode: str, file_path: Path):
    """Выполняет пайплайн в отдельном потоке, пишет события в RUN_REGISTRY."""
    from pipeline_new import Pipeline

    run = RUN_REGISTRY[run_id]

    def log_cb(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        run["logs"].append(f"[{ts}] {msg}")

    def progress_cb(current, total, desc):
        run["progress"] = {"current": current, "total": total, "desc": desc}

    pipeline = Pipeline(log_callback=log_cb)
    pipeline.progress_callback = progress_cb

    try:
        path = str(file_path)

        if mode == "full":
            # Полный конвейер — отслеживаем каждый этап
            log_cb("Запуск полного конвейера...")

            t0 = time.time()
            log_cb("Этап 1: Парсинг PDF...")
            run["current_stage"] = 1
            raw_xlsx = pipeline.stage_parse(path)
            elapsed1 = time.time() - t0
            fid1 = _register_output(raw_xlsx)
            run["stages"].append({"stage": 1, "name": "Парсинг", "time": elapsed1, "file_id": fid1, "filename": raw_xlsx.name})
            log_cb(f"Этап 1 завершён за {elapsed1:.0f}с: {raw_xlsx.name}")

            t1 = time.time()
            log_cb("Этап 2: Структурирование...")
            run["current_stage"] = 2
            struct_xlsx = pipeline.stage_struct(str(raw_xlsx))
            elapsed2 = time.time() - t1
            fid2 = _register_output(struct_xlsx)
            run["stages"].append({"stage": 2, "name": "Структурирование", "time": elapsed2, "file_id": fid2, "filename": struct_xlsx.name})
            log_cb(f"Этап 2 завершён за {elapsed2:.0f}с: {struct_xlsx.name}")

            t2 = time.time()
            log_cb("Этап 3: Матчинг...")
            run["current_stage"] = 3
            matched_xlsx = pipeline.stage_match(str(struct_xlsx))
            elapsed3 = time.time() - t2
            fid3 = _register_output(matched_xlsx)
            run["stages"].append({"stage": 3, "name": "Матчинг", "time": elapsed3, "file_id": fid3, "filename": matched_xlsx.name})
            log_cb(f"Этап 3 завершён за {elapsed3:.0f}с: {matched_xlsx.name}")

            t3 = time.time()
            log_cb("Этап 4: Агрегация...")
            run["current_stage"] = 4
            final_xlsx = pipeline.stage_aggregate(str(matched_xlsx))
            elapsed4 = time.time() - t3
            fid4 = _register_output(final_xlsx)
            run["stages"].append({"stage": 4, "name": "Агрегация", "time": elapsed4, "file_id": fid4, "filename": final_xlsx.name})
            log_cb(f"Этап 4 завершён за {elapsed4:.0f}с: {final_xlsx.name}")

            total_time = time.time() - t0
            log_cb(f"Конвейер завершён за {total_time:.0f}с")
            run["result_id"] = fid4
            run["total_time"] = total_time

        elif mode == "parse":
            t0 = time.time()
            run["current_stage"] = 1
            out = pipeline.stage_parse(path)
            elapsed = time.time() - t0
            fid = _register_output(out)
            run["stages"].append({"stage": 1, "name": "Парсинг", "time": elapsed, "file_id": fid, "filename": out.name})
            run["result_id"] = fid
            run["total_time"] = elapsed

        elif mode == "struct":
            t0 = time.time()
            run["current_stage"] = 2
            out = pipeline.stage_struct(path)
            elapsed = time.time() - t0
            fid = _register_output(out)
            run["stages"].append({"stage": 2, "name": "Структурирование", "time": elapsed, "file_id": fid, "filename": out.name})
            run["result_id"] = fid
            run["total_time"] = elapsed

        elif mode == "match":
            t0 = time.time()
            run["current_stage"] = 3
            out = pipeline.stage_match(path)
            elapsed = time.time() - t0
            fid = _register_output(out)
            run["stages"].append({"stage": 3, "name": "Матчинг", "time": elapsed, "file_id": fid, "filename": out.name})
            run["result_id"] = fid
            run["total_time"] = elapsed

        elif mode == "aggregate":
            t0 = time.time()
            run["current_stage"] = 4
            out = pipeline.stage_aggregate(path)
            elapsed = time.time() - t0
            fid = _register_output(out)
            run["stages"].append({"stage": 4, "name": "Агрегация", "time": elapsed, "file_id": fid, "filename": out.name})
            run["result_id"] = fid
            run["total_time"] = elapsed

        run["done"] = True

    except Exception as e:
        log_cb(f"Ошибка: {e}")
        log_cb(traceback.format_exc())
        run["error"] = str(e)
        run["done"] = True


@app.get("/api/run/{mode}")
async def run_pipeline(mode: str, file_id: str):
    if mode not in ("full", "parse", "struct", "match", "aggregate"):
        raise HTTPException(400, f"Unknown mode: {mode}")

    info = FILE_REGISTRY.get(file_id)
    if not info or not info["path"].exists():
        raise HTTPException(404, "File not found")

    run_id = uuid.uuid4().hex[:12]
    RUN_REGISTRY[run_id] = {
        "logs": [],
        "progress": {"current": 0, "total": 0, "desc": ""},
        "stages": [],
        "current_stage": 0,
        "done": False,
        "error": None,
        "result_id": None,
        "total_time": 0,
    }

    # Запускаем в фоновом потоке
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(run_id, mode, info["path"]),
        daemon=True,
    )
    thread.start()

    # SSE-стрим
    async def event_stream():
        last_log_idx = 0
        last_progress = ""
        last_stage_count = 0

        while True:
            run = RUN_REGISTRY.get(run_id)
            if not run:
                break

            # Отправляем новые логи
            new_logs = run["logs"][last_log_idx:]
            for line in new_logs:
                yield f"event: log\ndata: {json.dumps({'message': line}, ensure_ascii=False)}\n\n"
            last_log_idx = len(run["logs"])

            # Отправляем прогресс
            prog = run["progress"]
            prog_key = f"{prog['current']}/{prog['total']}/{prog['desc']}"
            if prog_key != last_progress and prog["total"] > 0:
                yield f"event: progress\ndata: {json.dumps(prog, ensure_ascii=False)}\n\n"
                last_progress = prog_key

            # Отправляем завершённые этапы
            stages = run["stages"]
            if len(stages) > last_stage_count:
                for s in stages[last_stage_count:]:
                    yield f"event: stage_complete\ndata: {json.dumps(s, ensure_ascii=False)}\n\n"
                last_stage_count = len(stages)

            # Отправляем текущий этап
            if run["current_stage"] > 0:
                yield f"event: current_stage\ndata: {json.dumps({'stage': run['current_stage']}, ensure_ascii=False)}\n\n"

            # Проверяем завершение
            if run["done"]:
                if run["error"]:
                    yield f"event: error\ndata: {json.dumps({'message': run['error']}, ensure_ascii=False)}\n\n"
                else:
                    yield f"event: done\ndata: {json.dumps({'result_id': run['result_id'], 'total_time': run['total_time'], 'stages': run['stages']}, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── API: просмотр директорий результатов ──────────────────────────────
RESULT_DIRS = {
    "parse": PARSE_DIR,
    "struct": STRUCT_DIR,
    "match": RESULT_DIR,
    "aggregate": FINAL_DIR,
}

@app.get("/api/files/{folder}")
async def list_files(folder: str):
    d = RESULT_DIRS.get(folder)
    if not d:
        raise HTTPException(400, f"Unknown folder: {folder}")
    if not d.exists():
        return {"files": []}
    files = []
    for f in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.is_file() and f.suffix in (".xlsx", ".csv", ".pdf") and not f.name.startswith(("~$", ".", "_")):
            stat = f.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%d.%m.%Y %H:%M")
            files.append({
                "name": f.name,
                "folder": folder,
                "size": f"{size_kb:.1f} КБ" if size_kb < 1024 else f"{size_kb/1024:.1f} МБ",
                "modified": mtime,
            })
    return {"files": files}


@app.get("/api/files/{folder}/{filename}")
async def download_result_file(folder: str, filename: str):
    d = RESULT_DIRS.get(folder)
    if not d:
        raise HTTPException(400, f"Unknown folder: {folder}")
    file_path = d / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")
    # Безопасность: не выходим за пределы директории
    if not str(file_path.resolve()).startswith(str(d.resolve())):
        raise HTTPException(403, "Forbidden")
    return FileResponse(file_path, filename=filename)


# ── Статика и SPA ────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


if __name__ == "__main__":
    host = os.getenv("GRAPHIT_HOST", "localhost")
    port = int(os.getenv("GRAPHIT_PORT", "7860"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
