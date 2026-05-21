// app.js — Клиентская логика для конвейера ВР → ВиКР

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initFullPipeline();
    initStages();
    initResults();
});

// ═══ TABS ═══════════════════════════════════════════════════════════

function initTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
            // Auto-refresh results tab
            if (tab.dataset.tab === 'results') loadAllFolders();
        });
    });
}

// ═══ FULL PIPELINE ══════════════════════════════════════════════════

function initFullPipeline() {
    const fileInput = document.getElementById('full-file');
    const uploadArea = document.getElementById('full-upload-area');
    const placeholder = document.getElementById('full-upload-placeholder');
    const selected = document.getElementById('full-upload-selected');
    const fileName = document.getElementById('full-file-name');
    const clearBtn = document.getElementById('full-clear-btn');
    const runBtn = document.getElementById('full-run-btn');

    let uploadedFileId = null;
    let timerInterval = null;
    let timerStart = null;

    // Upload area click
    uploadArea.addEventListener('click', (e) => {
        if (e.target === clearBtn || e.target.closest('.btn-clear')) return;
        fileInput.click();
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(fileInput.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFileSelect(fileInput.files[0]);
    });

    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        uploadedFileId = null;
        fileInput.value = '';
        placeholder.hidden = false;
        selected.hidden = true;
        runBtn.disabled = true;
    }

    async function handleFileSelect(file) {
        fileName.textContent = file.name;
        placeholder.hidden = true;
        selected.hidden = false;
        runBtn.disabled = true;

        // Upload
        const fd = new FormData();
        fd.append('file', file);
        try {
            const res = await fetch('/api/upload', { method: 'POST', body: fd });
            const data = await res.json();
            uploadedFileId = data.file_id;
            runBtn.disabled = false;
        } catch (err) {
            fileName.textContent = 'Ошибка загрузки';
            console.error(err);
        }
    }

    // Run button
    runBtn.addEventListener('click', () => {
        if (!uploadedFileId) return;
        runFullPipeline(uploadedFileId);
    });

    function runFullPipeline(fileId) {
        const stepper = document.getElementById('full-stepper');
        const progress = document.getElementById('full-progress');
        const progressBar = document.getElementById('full-progress-bar');
        const progressText = document.getElementById('full-progress-text');
        const timer = document.getElementById('full-timer');
        const timerValue = document.getElementById('full-timer-value');
        const result = document.getElementById('full-result');
        const resultTime = document.getElementById('full-result-time');
        const downloadLink = document.getElementById('full-download-link');
        const log = document.getElementById('full-log');

        // Reset UI
        stepper.hidden = false;
        progress.hidden = false;
        timer.hidden = false;
        result.hidden = true;
        log.textContent = '';
        progressBar.style.width = '0%';
        progressText.textContent = '';
        runBtn.disabled = true;
        runBtn.classList.add('running');
        runBtn.textContent = 'Выполняется...';

        // Reset stepper
        stepper.querySelectorAll('.step').forEach(s => { s.classList.remove('active', 'done'); });
        stepper.querySelectorAll('.step-line').forEach(l => l.classList.remove('done'));
        stepper.querySelectorAll('.step-time').forEach(t => { t.textContent = ''; });

        // Start timer
        timerStart = Date.now();
        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - timerStart) / 1000);
            const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
            const s = String(elapsed % 60).padStart(2, '0');
            timerValue.textContent = `${m}:${s}`;
        }, 1000);

        // SSE
        const es = new EventSource(`/api/run/full?file_id=${fileId}`);

        es.addEventListener('log', (e) => {
            const data = JSON.parse(e.data);
            log.textContent += data.message + '\n';
            log.scrollTop = log.scrollHeight;
        });

        es.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            if (data.total > 0) {
                const pct = Math.round((data.current / data.total) * 100);
                progressBar.style.width = pct + '%';
                progressText.textContent = `${data.desc}: ${data.current}/${data.total}`;
            }
        });

        es.addEventListener('current_stage', (e) => {
            const data = JSON.parse(e.data);
            const stageNum = data.stage;
            // Activate current step
            stepper.querySelectorAll('.step').forEach(s => {
                const sn = parseInt(s.dataset.step);
                if (sn < stageNum) {
                    s.classList.remove('active');
                    s.classList.add('done');
                } else if (sn === stageNum) {
                    s.classList.add('active');
                    s.classList.remove('done');
                }
            });
            // Activate lines before current step
            const lines = stepper.querySelectorAll('.step-line');
            lines.forEach((l, i) => {
                if (i < stageNum - 1) l.classList.add('done');
            });
        });

        es.addEventListener('stage_complete', (e) => {
            const data = JSON.parse(e.data);
            const step = stepper.querySelector(`.step[data-step="${data.stage}"]`);
            if (step) {
                step.classList.remove('active');
                step.classList.add('done');
                const timeEl = step.querySelector('.step-time');
                if (timeEl) {
                    const m = Math.floor(data.time / 60);
                    const s = Math.round(data.time % 60);
                    timeEl.textContent = m > 0 ? `${m}м ${s}с` : `${s}с`;
                }
            }
            // Reset progress bar for next stage
            progressBar.style.width = '0%';
            progressText.textContent = '';
        });

        es.addEventListener('done', (e) => {
            const data = JSON.parse(e.data);
            es.close();
            clearInterval(timerInterval);

            // Mark all lines as done
            stepper.querySelectorAll('.step-line').forEach(l => l.classList.add('done'));

            // Show result
            result.hidden = false;
            const totalSec = Math.round(data.total_time);
            const m = Math.floor(totalSec / 60);
            const s = totalSec % 60;
            resultTime.textContent = m > 0 ? `Время: ${m}м ${s}с` : `Время: ${s}с`;
            downloadLink.href = `/api/download/${data.result_id}`;

            // Reset button
            runBtn.disabled = false;
            runBtn.classList.remove('running');
            runBtn.textContent = 'Запустить конвейер';

            progress.hidden = true;
        });

        es.addEventListener('error', (e) => {
            try {
                const data = JSON.parse(e.data);
                log.textContent += '\n❌ ' + data.message + '\n';
            } catch (_) {}
            es.close();
            clearInterval(timerInterval);
            runBtn.disabled = false;
            runBtn.classList.remove('running');
            runBtn.textContent = 'Запустить конвейер';
        });

        es.onerror = () => {
            // SSE connection closed — might be normal end or real error
            es.close();
        };
    }
}

// ═══ STAGES (По этапам) ═════════════════════════════════════════════

function initStages() {
    document.querySelectorAll('.stage-card').forEach(card => {
        const mode = card.dataset.stageMode;
        const header = card.querySelector('.stage-header');
        const body = card.querySelector('.stage-body');
        const uploadArea = card.querySelector('.upload-area');
        const fileInput = uploadArea.querySelector('input[type="file"]');
        const placeholder = uploadArea.querySelector('.upload-placeholder');
        const selected = uploadArea.querySelector('.upload-selected');
        const fileNameEl = uploadArea.querySelector('.file-name');
        const clearBtn = uploadArea.querySelector('.btn-clear');
        const runBtn = card.querySelector('.btn-primary');
        const progressSection = card.querySelector('.stage-progress');
        const progressBar = card.querySelector('.progress-bar');
        const progressText = card.querySelector('.progress-text');
        const statusEl = card.querySelector('.stage-status');
        const resultEl = card.querySelector('.stage-result');
        const downloadLink = card.querySelector('.btn-download');
        const logEl = card.querySelector('.log-output');

        let uploadedFileId = null;

        // Accordion toggle
        header.addEventListener('click', () => {
            card.classList.toggle('collapsed');
        });

        // Upload area
        uploadArea.addEventListener('click', (e) => {
            if (e.target === clearBtn || e.target.closest('.btn-clear')) return;
            fileInput.click();
        });

        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFile(fileInput.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        });

        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetStage();
        });

        function resetStage() {
            uploadedFileId = null;
            fileInput.value = '';
            placeholder.hidden = false;
            selected.hidden = true;
            runBtn.disabled = true;
            progressSection.hidden = true;
            statusEl.hidden = true;
            resultEl.hidden = true;
            logEl.hidden = true;
            logEl.textContent = '';
        }

        async function handleFile(file) {
            fileNameEl.textContent = file.name;
            placeholder.hidden = true;
            selected.hidden = false;
            runBtn.disabled = true;

            const fd = new FormData();
            fd.append('file', file);
            try {
                const res = await fetch('/api/upload', { method: 'POST', body: fd });
                const data = await res.json();
                uploadedFileId = data.file_id;
                runBtn.disabled = false;
            } catch (err) {
                fileNameEl.textContent = 'Ошибка загрузки';
            }
        }

        // Run
        runBtn.addEventListener('click', () => {
            if (!uploadedFileId) return;
            runStage(mode, uploadedFileId);
        });

        function runStage(stageMode, fileId) {
            // Reset
            progressSection.hidden = false;
            progressBar.style.width = '0%';
            progressText.textContent = '';
            statusEl.hidden = true;
            resultEl.hidden = true;
            logEl.hidden = false;
            logEl.textContent = '';
            runBtn.disabled = true;
            runBtn.classList.add('running');

            const es = new EventSource(`/api/run/${stageMode}?file_id=${fileId}`);

            es.addEventListener('log', (e) => {
                const data = JSON.parse(e.data);
                logEl.textContent += data.message + '\n';
                logEl.scrollTop = logEl.scrollHeight;
            });

            es.addEventListener('progress', (e) => {
                const data = JSON.parse(e.data);
                if (data.total > 0) {
                    const pct = Math.round((data.current / data.total) * 100);
                    progressBar.style.width = pct + '%';
                    progressText.textContent = `${data.desc}: ${data.current}/${data.total}`;
                }
            });

            es.addEventListener('done', (e) => {
                const data = JSON.parse(e.data);
                es.close();

                progressSection.hidden = true;

                // Status
                statusEl.hidden = false;
                statusEl.className = 'stage-status success';
                const totalSec = Math.round(data.total_time);
                const m = Math.floor(totalSec / 60);
                const s = totalSec % 60;
                const timeStr = m > 0 ? `${m}м ${s}с` : `${s}с`;
                statusEl.textContent = `Готово за ${timeStr}`;

                // Download
                resultEl.hidden = false;
                downloadLink.href = `/api/download/${data.result_id}`;

                runBtn.disabled = false;
                runBtn.classList.remove('running');
            });

            es.addEventListener('error', (e) => {
                let msg = 'Неизвестная ошибка';
                try {
                    const data = JSON.parse(e.data);
                    msg = data.message;
                } catch (_) {}
                es.close();

                progressSection.hidden = true;
                statusEl.hidden = false;
                statusEl.className = 'stage-status error';
                statusEl.textContent = 'Ошибка: ' + msg;

                runBtn.disabled = false;
                runBtn.classList.remove('running');
            });

            es.onerror = () => { es.close(); };
        }
    });
}

// ═══ RESULTS TAB (Директории) ═══════════════════════════════════════

function loadAllFolders() {
    document.querySelectorAll('.results-folder').forEach(folder => {
        loadFolder(folder);
    });
}

async function loadFolder(folderEl) {
    const folderName = folderEl.dataset.folder;
    const filesContainer = folderEl.querySelector('.folder-files');

    try {
        const res = await fetch(`/api/files/${folderName}`);
        const data = await res.json();

        if (!data.files || data.files.length === 0) {
            filesContainer.innerHTML = '<div class="folder-empty">Нет файлов</div>';
            return;
        }

        filesContainer.innerHTML = data.files.map(f => `
            <div class="file-row">
                <div class="file-info">
                    <span class="fname" title="${f.name}">${f.name}</span>
                    <span class="fmeta">${f.size} · ${f.modified}</span>
                </div>
                <a class="btn-dl" href="/api/files/${f.folder}/${encodeURIComponent(f.name)}">Скачать</a>
            </div>
        `).join('');
    } catch (err) {
        filesContainer.innerHTML = '<div class="folder-empty">Ошибка загрузки</div>';
    }
}

function initResults() {
    // Refresh buttons
    document.querySelectorAll('.btn-refresh').forEach(btn => {
        btn.addEventListener('click', () => {
            const folder = btn.closest('.results-folder');
            loadFolder(folder);
        });
    });
}
