// Development-only diagnostic profile review.
(function () {
    const state = {
        profiles: [],
        filtered: [],
        currentIndex: -1,
        previousIndex: null,
        current: null,
        originalImage: null,
        canvas: null,
        ctx: null,
        maskCanvas: document.createElement('canvas'),
        maskCtx: null,
        tool: 'brush',
        brushSize: 22,
        isDrawing: false,
        rectStart: null,
        lastPoint: null,
        cursorPoint: null,
        undoStack: [],
        undoLimit: 25,
        zoom: 1,
        baseScale: 1,
        zoomMin: 1,
        zoomMax: 8,
        showUnresolvedOnly: true,
    };

    function projectId() {
        return window.projectManager?.getCurrentProject?.()?.project_id ||
            localStorage.getItem('currentProjectId');
    }

    function escapeHtml(value) {
        const node = document.createElement('div');
        node.textContent = value == null ? '' : String(value);
        return node.innerHTML;
    }

    function loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    }

    function displayMessage(title, message) {
        const empty = document.getElementById('profile-empty-msg');
        const editor = document.getElementById('profile-editor');
        if (editor) editor.style.display = 'none';
        if (empty) {
            empty.innerHTML = `<h3>${escapeHtml(title)}</h3><p>${escapeHtml(message)}</p>`;
            empty.style.display = 'flex';
        }
    }

    async function loadProfileReview() {
        const pid = projectId();
        if (!pid) {
            state.profiles = [];
            renderList();
            displayMessage('No project selected', 'Select a project first.');
            return;
        }
        try {
            const response = await fetch(`/api/projects/${pid}/profiles`);
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Failed to load profiles');
            state.profiles = data.profiles || [];
            applyFilter();
            renderList();
            if (!state.filtered.length) {
                displayMessage('No profile proposals', 'Extract approved crops, then generate profile proposals.');
                return;
            }
            const next = Math.max(0, state.filtered.findIndex(item =>
                state.current && item.filename === state.current.filename));
            await selectProfile(next, { remember: false });
        } catch (error) {
            console.error('[Profiles] load failed:', error);
            displayMessage('Profile review error', error.message);
        }
    }

    function applyFilter() {
        state.showUnresolvedOnly = !!document.getElementById('profile-unresolved-only')?.checked;
        state.filtered = state.profiles.filter(item => {
            if (!state.showUnresolvedOnly) return true;
            return !['approved', 'edited', 'no_profile'].includes(item.review_status);
        });
    }

    function renderList() {
        const list = document.getElementById('profile-review-list');
        const count = document.getElementById('profile-review-count');
        if (count) count.textContent = state.filtered.length;
        if (!list) return;
        if (!state.filtered.length) {
            list.innerHTML = '<div class="empty-list">No profile crops</div>';
            return;
        }
        list.innerHTML = state.filtered.map((item, index) => {
            const active = index === state.currentIndex ? 'active' : '';
            const confidence = item.proposal?.confidence != null
                ? `${Math.round(Number(item.proposal.confidence) * 100)}%`
                : '-';
            return `<div class="annotation-image-item profile-list-item ${active}" data-index="${index}">
                <span class="image-number">${escapeHtml(statusLabel(item.review_status))}</span>
                <span class="image-name" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</span>
                <small>${escapeHtml(confidence)}</small>
            </div>`;
        }).join('');
        list.querySelectorAll('[data-index]').forEach(row => {
            row.addEventListener('click', () => selectProfile(Number(row.dataset.index)));
        });
        updateBackButton();
    }

    function statusLabel(status) {
        if (status === 'approved') return 'OK';
        if (status === 'edited') return 'EDIT';
        if (status === 'no_profile') return 'NONE';
        if (status === 'failed') return 'FAIL';
        if (status === 'pending') return 'WAIT';
        return 'NEW';
    }

    async function selectProfile(index, options = {}) {
        if (index < 0 || index >= state.filtered.length) return;
        const remember = options.remember !== false;
        if (remember && state.currentIndex >= 0 && state.currentIndex !== index) {
            state.previousIndex = state.currentIndex;
        }
        state.currentIndex = index;
        state.current = state.filtered[index];
        renderList();
        updateBackButton();

        const empty = document.getElementById('profile-empty-msg');
        const editor = document.getElementById('profile-editor');
        if (empty) empty.style.display = 'none';
        if (editor) editor.style.display = 'block';
        document.getElementById('profile-current-name').textContent = state.current.filename;
        document.getElementById('profile-current-status').textContent = state.current.review_status;
        document.getElementById('profile-confidence').textContent =
            state.current.proposal?.confidence != null
                ? `${Math.round(Number(state.current.proposal.confidence) * 100)}%`
                : '-';
        document.getElementById('profile-reasons').textContent =
            (state.current.proposal?.reasons || []).join(', ') || '-';
        document.getElementById('profile-review-note').value = state.current.review_note || '';

        state.originalImage = await loadImage(state.current.card_url);
        await loadMask(state.current.accepted_mask_url || state.current.auto_mask_url);
        state.undoStack = [];
        updateUndoButton();
        state.zoom = 1;
        resizeEditorCanvas();
        redraw();
        centerOnMask();
    }

    function updateBackButton() {
        const button = document.getElementById('profile-back-btn');
        if (!button) return;
        button.disabled = state.previousIndex == null || !state.filtered[state.previousIndex];
    }

    function updateUndoButton() {
        const button = document.getElementById('profile-undo-btn');
        if (!button) return;
        button.disabled = state.undoStack.length === 0;
    }

    function pushUndoState() {
        if (!state.maskCtx || !state.maskCanvas.width || !state.maskCanvas.height) return;
        state.undoStack.push(state.maskCtx.getImageData(
            0, 0, state.maskCanvas.width, state.maskCanvas.height));
        if (state.undoStack.length > state.undoLimit) state.undoStack.shift();
        updateUndoButton();
    }

    function undoProfileEdit() {
        if (!state.undoStack.length || !state.maskCtx) return;
        const previous = state.undoStack.pop();
        state.maskCtx.putImageData(previous, 0, 0);
        updateUndoButton();
        redraw();
        centerOnMask();
    }

    async function loadMask(url) {
        state.maskCanvas.width = state.originalImage.width;
        state.maskCanvas.height = state.originalImage.height;
        state.maskCtx = state.maskCanvas.getContext('2d');
        state.maskCtx.fillStyle = 'black';
        state.maskCtx.fillRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
        if (!url) return;
        try {
            const img = await loadImage(`${url}?v=${Date.now()}`);
            state.maskCtx.drawImage(img, 0, 0, state.maskCanvas.width, state.maskCanvas.height);
        } catch (error) {
            console.warn('[Profiles] could not load mask:', error);
        }
    }

    function resizeEditorCanvas() {
        const canvas = state.canvas;
        if (!canvas || !state.originalImage) return;
        const maxWidth = 980;
        const maxHeight = 720;
        state.baseScale = Math.min(1, maxWidth / state.originalImage.width, maxHeight / state.originalImage.height);
        const scale = state.baseScale * state.zoom;
        canvas.width = Math.max(1, Math.round(state.originalImage.width * scale));
        canvas.height = Math.max(1, Math.round(state.originalImage.height * scale));
    }

    function currentMaskBbox() {
        if (!state.maskCtx || !state.maskCanvas.width || !state.maskCanvas.height) return null;
        const data = state.maskCtx.getImageData(0, 0, state.maskCanvas.width, state.maskCanvas.height).data;
        let minX = state.maskCanvas.width;
        let minY = state.maskCanvas.height;
        let maxX = -1;
        let maxY = -1;
        for (let y = 0; y < state.maskCanvas.height; y++) {
            for (let x = 0; x < state.maskCanvas.width; x++) {
                const i = (y * state.maskCanvas.width + x) * 4;
                if (data[i] > 32) {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }
        return maxX >= minX ? { x1: minX, y1: minY, x2: maxX + 1, y2: maxY + 1 } : null;
    }

    function centerCanvasOnOriginalPoint(x, y) {
        const wrap = document.getElementById('profile-canvas-wrap');
        if (!wrap || !state.originalImage) return;
        const scaleX = state.canvas.width / state.originalImage.width;
        const scaleY = state.canvas.height / state.originalImage.height;
        wrap.scrollLeft = Math.max(0, x * scaleX - wrap.clientWidth / 2);
        wrap.scrollTop = Math.max(0, y * scaleY - wrap.clientHeight / 2);
    }

    function centerOnMask() {
        const bbox = currentMaskBbox();
        if (!bbox) return;
        centerCanvasOnOriginalPoint((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2);
    }

    function setProfileZoom(nextZoom, centerToBlob = true) {
        if (!state.originalImage) return;
        const bbox = currentMaskBbox();
        const center = bbox
            ? { x: (bbox.x1 + bbox.x2) / 2, y: (bbox.y1 + bbox.y2) / 2 }
            : { x: state.originalImage.width / 2, y: state.originalImage.height / 2 };
        state.zoom = Math.max(state.zoomMin, Math.min(state.zoomMax, nextZoom));
        resizeEditorCanvas();
        redraw();
        if (centerToBlob) centerCanvasOnOriginalPoint(center.x, center.y);
    }

    function zoomTowardBlob() {
        setProfileZoom(state.zoom * 1.6, true);
    }

    function zoomOutFromBlob() {
        setProfileZoom(state.zoom / 1.6, true);
    }

    function fitProfileCanvas() {
        state.zoom = 1;
        resizeEditorCanvas();
        redraw();
        centerOnMask();
    }

    function eventPoint(event) {
        const rect = state.canvas.getBoundingClientRect();
        return {
            x: Math.round((event.clientX - rect.left) * state.originalImage.width / rect.width),
            y: Math.round((event.clientY - rect.top) * state.originalImage.height / rect.height),
        };
    }

    function redraw(previewRect = null) {
        if (!state.ctx || !state.originalImage) return;
        const canvas = state.canvas;
        state.ctx.clearRect(0, 0, canvas.width, canvas.height);
        state.ctx.drawImage(state.originalImage, 0, 0, canvas.width, canvas.height);

        const overlay = document.createElement('canvas');
        overlay.width = state.maskCanvas.width;
        overlay.height = state.maskCanvas.height;
        const overlayCtx = overlay.getContext('2d');
        const maskData = state.maskCtx.getImageData(0, 0, overlay.width, overlay.height);
        const overlayData = overlayCtx.createImageData(overlay.width, overlay.height);
        for (let i = 0; i < maskData.data.length; i += 4) {
            if (maskData.data[i] > 32) {
                overlayData.data[i] = 239;
                overlayData.data[i + 1] = 68;
                overlayData.data[i + 2] = 68;
                overlayData.data[i + 3] = 120;
            }
        }
        overlayCtx.putImageData(overlayData, 0, 0);
        state.ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);

        if (previewRect) {
            const sx = canvas.width / state.originalImage.width;
            const sy = canvas.height / state.originalImage.height;
            state.ctx.save();
            state.ctx.strokeStyle = '#2563eb';
            state.ctx.lineWidth = 2;
            state.ctx.setLineDash([6, 4]);
            state.ctx.strokeRect(
                previewRect.x1 * sx, previewRect.y1 * sy,
                (previewRect.x2 - previewRect.x1) * sx,
                (previewRect.y2 - previewRect.y1) * sy);
            state.ctx.restore();
        }
        drawToolCursor();

        const preview = document.getElementById('profile-mask-preview');
        if (preview) preview.src = state.maskCanvas.toDataURL('image/png');
    }

    function drawToolCursor() {
        if (!state.cursorPoint || !state.originalImage || state.tool === 'rect') return;
        const canvas = state.canvas;
        const sx = canvas.width / state.originalImage.width;
        const sy = canvas.height / state.originalImage.height;
        const x = state.cursorPoint.x * sx;
        const y = state.cursorPoint.y * sy;
        const radius = state.tool === 'branch'
            ? 8
            : Math.max(3, (state.brushSize / 2) * ((sx + sy) / 2));
        state.ctx.save();
        state.ctx.lineWidth = 2;
        state.ctx.strokeStyle = state.tool === 'eraser' || state.tool === 'branch' ? '#111827' : '#ffffff';
        state.ctx.beginPath();
        state.ctx.arc(x, y, radius, 0, Math.PI * 2);
        state.ctx.stroke();
        state.ctx.lineWidth = 1;
        state.ctx.strokeStyle = state.tool === 'eraser' || state.tool === 'branch' ? '#ffffff' : '#111827';
        state.ctx.beginPath();
        state.ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
        state.ctx.stroke();
        if (state.tool === 'branch') {
            state.ctx.beginPath();
            state.ctx.moveTo(x - radius - 3, y);
            state.ctx.lineTo(x + radius + 3, y);
            state.ctx.moveTo(x, y - radius - 3);
            state.ctx.lineTo(x, y + radius + 3);
            state.ctx.stroke();
        }
        state.ctx.restore();
    }

    function maskBinary() {
        const width = state.maskCanvas.width;
        const height = state.maskCanvas.height;
        const data = state.maskCtx.getImageData(0, 0, width, height).data;
        const mask = new Uint8Array(width * height);
        for (let i = 0; i < mask.length; i++) mask[i] = data[i * 4] > 32 ? 1 : 0;
        return { mask, width, height };
    }

    function distanceFromBackground(mask, width, height) {
        const total = width * height;
        const dist = new Int32Array(total);
        const queue = new Int32Array(total);
        let head = 0, tail = 0;
        for (let i = 0; i < total; i++) {
            if (!mask[i]) {
                dist[i] = 0;
                queue[tail++] = i;
            } else {
                dist[i] = -1;
            }
        }
        while (head < tail) {
            const index = queue[head++];
            const x = index % width;
            const y = Math.floor(index / width);
            const nextDistance = dist[index] + 1;
            const neighbors = [];
            if (x > 0) neighbors.push(index - 1);
            if (x < width - 1) neighbors.push(index + 1);
            if (y > 0) neighbors.push(index - width);
            if (y < height - 1) neighbors.push(index + width);
            for (const next of neighbors) {
                if (dist[next] !== -1) continue;
                dist[next] = nextDistance;
                queue[tail++] = next;
            }
        }
        return dist;
    }

    function removeBranchAt(point) {
        if (!state.maskCtx || !state.originalImage) return false;
        const { mask, width, height } = maskBinary();
        let sx = Math.max(0, Math.min(width - 1, Math.round(point.x)));
        let sy = Math.max(0, Math.min(height - 1, Math.round(point.y)));
        let seed = sy * width + sx;
        if (!mask[seed]) {
            let best = -1;
            let bestDistance = Infinity;
            const radius = 8;
            for (let y = Math.max(0, sy - radius); y <= Math.min(height - 1, sy + radius); y++) {
                for (let x = Math.max(0, sx - radius); x <= Math.min(width - 1, sx + radius); x++) {
                    const index = y * width + x;
                    if (!mask[index]) continue;
                    const distance = Math.hypot(x - sx, y - sy);
                    if (distance < bestDistance) {
                        best = index;
                        bestDistance = distance;
                    }
                }
            }
            if (best < 0) return false;
            seed = best;
        }

        const thickness = distanceFromBackground(mask, width, height);
        const supported = new Uint8Array(width * height);
        const supportCutoff = 3;
        for (let i = 0; i < mask.length; i++) supported[i] = thickness[i] >= supportCutoff ? 1 : 0;

        const supportDistance = new Int32Array(width * height);
        supportDistance.fill(-1);
        const supportQueue = new Int32Array(width * height);
        let supportHead = 0, supportTail = 0;
        for (let i = 0; i < supported.length; i++) {
            if (!supported[i]) continue;
            supportDistance[i] = 0;
            supportQueue[supportTail++] = i;
        }
        while (supportHead < supportTail) {
            const index = supportQueue[supportHead++];
            const x = index % width;
            const y = Math.floor(index / width);
            const nextDistance = supportDistance[index] + 1;
            const neighbors = [];
            if (x > 0) neighbors.push(index - 1);
            if (x < width - 1) neighbors.push(index + 1);
            if (y > 0) neighbors.push(index - width);
            if (y < height - 1) neighbors.push(index + width);
            for (const next of neighbors) {
                if (!mask[next] || supportDistance[next] !== -1) continue;
                supportDistance[next] = nextDistance;
                supportQueue[supportTail++] = next;
            }
        }

        const branch = new Uint8Array(width * height);
        const branchQueue = new Int32Array(width * height);
        let head = 0, tail = 0;
        const allow = index => mask[index] && thickness[index] <= 2 && supportDistance[index] > 1;
        if (!allow(seed)) {
            let best = -1;
            let bestDistance = Infinity;
            const sx0 = seed % width;
            const sy0 = Math.floor(seed / width);
            for (let y = Math.max(0, sy0 - 12); y <= Math.min(height - 1, sy0 + 12); y++) {
                for (let x = Math.max(0, sx0 - 12); x <= Math.min(width - 1, sx0 + 12); x++) {
                    const index = y * width + x;
                    if (!allow(index)) continue;
                    const distance = Math.hypot(x - sx0, y - sy0);
                    if (distance < bestDistance) {
                        best = index;
                        bestDistance = distance;
                    }
                }
            }
            if (best < 0) return false;
            seed = best;
        }
        branch[seed] = 1;
        branchQueue[tail++] = seed;
        while (head < tail) {
            const index = branchQueue[head++];
            const x = index % width;
            const y = Math.floor(index / width);
            const neighbors = [];
            if (x > 0) neighbors.push(index - 1);
            if (x < width - 1) neighbors.push(index + 1);
            if (y > 0) neighbors.push(index - width);
            if (y < height - 1) neighbors.push(index + width);
            for (const next of neighbors) {
                if (branch[next] || !allow(next)) continue;
                branch[next] = 1;
                branchQueue[tail++] = next;
            }
        }
        if (tail <= 0) return false;

        const image = state.maskCtx.getImageData(0, 0, width, height);
        for (let i = 0; i < branch.length; i++) {
            if (!branch[i]) continue;
            const offset = i * 4;
            image.data[offset] = 0;
            image.data[offset + 1] = 0;
            image.data[offset + 2] = 0;
            image.data[offset + 3] = 255;
        }
        state.maskCtx.putImageData(image, 0, 0);
        return true;
    }

    function drawBrush(point, erase) {
        const ctx = state.maskCtx;
        ctx.save();
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = erase ? 'black' : 'white';
        ctx.fillStyle = erase ? 'black' : 'white';
        ctx.lineWidth = state.brushSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        const last = state.lastPoint || point;
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(point.x, point.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(point.x, point.y, state.brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        state.lastPoint = point;
        redraw();
    }

    async function rerunProfileInRect(rect) {
        const x1 = Math.max(0, Math.min(rect.x1, rect.x2));
        const y1 = Math.max(0, Math.min(rect.y1, rect.y2));
        const x2 = Math.min(state.originalImage.width, Math.max(rect.x1, rect.x2));
        const y2 = Math.min(state.originalImage.height, Math.max(rect.y1, rect.y2));
        if (x2 - x1 < 4 || y2 - y1 < 4) return;

        const pid = projectId();
        const response = await fetch(
            `/api/projects/${pid}/profiles/${encodeURIComponent(state.current.filename)}/rerun-rect`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bbox: [x1, y1, x2, y2] }),
            });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'Profile rectangle rerun failed');

        const img = await loadImage(data.mask_data);
        state.maskCtx.fillStyle = 'black';
        state.maskCtx.fillRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
        state.maskCtx.drawImage(img, 0, 0, state.maskCanvas.width, state.maskCanvas.height);
        if (data.proposal) {
            document.getElementById('profile-confidence').textContent =
                data.proposal.confidence != null
                    ? `${Math.round(Number(data.proposal.confidence) * 100)}%`
                    : '-';
            document.getElementById('profile-reasons').textContent =
                [...(data.proposal.reasons || []), 'manual_rectangle_rerun'].join(', ');
        }
        redraw();
        centerOnMask();
    }

    async function saveCurrent(status) {
        if (!state.current) return;
        const pid = projectId();
        const note = document.getElementById('profile-review-note')?.value || '';
        const body = { review_status: status, review_note: note };
        if (status !== 'approved' || state.current.accepted_mask_url || state.current.auto_mask_url) {
            body.mask_data = state.maskCanvas.toDataURL('image/png');
        }
        const response = await fetch(`/api/projects/${pid}/profiles/${encodeURIComponent(state.current.filename)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'Profile save failed');
        if (window.PyPotteryUtils) window.PyPotteryUtils.showToast('Profile saved', 'success');
        await loadProfileReview();
    }

    async function generateProfiles() {
        const pid = projectId();
        if (!pid) return;
        const btn = document.getElementById('profile-generate-btn');
        const summary = document.getElementById('profile-review-summary');
        try {
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'Generating...';
            }
            const response = await fetch(`/api/projects/${pid}/profiles/propose`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ force: true }),
            });
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Profile proposal failed');
            if (summary) {
                const s = data.summary || {};
                summary.textContent = `${s.generated || 0} regenerated and returned to review, ${s.failed || 0} failed.`;
                summary.className = 'status-message success';
            }
            await loadProfileReview();
        } catch (error) {
            if (summary) {
                summary.textContent = error.message;
                summary.className = 'status-message error';
            }
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = 'Generate profile proposals';
            }
        }
    }

    function setTool(tool) {
        state.tool = tool;
        document.querySelectorAll('[data-profile-tool]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.profileTool === tool);
        });
    }

    function initializeProfileReview() {
        state.canvas = document.getElementById('profile-canvas');
        if (!state.canvas) return;
        state.ctx = state.canvas.getContext('2d');
        state.maskCtx = state.maskCanvas.getContext('2d');

        document.getElementById('profile-generate-btn')?.addEventListener('click', generateProfiles);
        document.getElementById('profile-back-btn')?.addEventListener('click', () => {
            if (state.previousIndex == null || !state.filtered[state.previousIndex]) return;
            const target = state.previousIndex;
            state.previousIndex = state.currentIndex >= 0 ? state.currentIndex : null;
            selectProfile(target, { remember: false });
        });
        document.getElementById('profile-undo-btn')?.addEventListener('click', undoProfileEdit);
        document.getElementById('profile-zoom-in-btn')?.addEventListener('click', zoomTowardBlob);
        document.getElementById('profile-zoom-out-btn')?.addEventListener('click', zoomOutFromBlob);
        document.getElementById('profile-zoom-fit-btn')?.addEventListener('click', fitProfileCanvas);
        document.getElementById('profile-unresolved-only')?.addEventListener('change', () => {
            applyFilter();
            state.currentIndex = -1;
            renderList();
            if (state.filtered.length) selectProfile(0);
            else displayMessage('No unresolved profiles', 'Turn off the filter to see approved profile masks.');
        });
        document.getElementById('profile-approve-btn')?.addEventListener('click', () => saveCurrent('approved').catch(alert));
        document.getElementById('profile-save-edit-btn')?.addEventListener('click', () => saveCurrent('edited').catch(alert));
        document.getElementById('profile-no-profile-btn')?.addEventListener('click', () => saveCurrent('no_profile').catch(alert));
        document.getElementById('profile-reset-btn')?.addEventListener('click', async () => {
            if (!state.current?.auto_mask_url) return;
            pushUndoState();
            await loadMask(state.current.auto_mask_url);
            redraw();
            centerOnMask();
        });
        document.querySelectorAll('[data-profile-tool]').forEach(btn => {
            btn.addEventListener('click', () => setTool(btn.dataset.profileTool));
        });
        document.getElementById('profile-brush-size')?.addEventListener('input', event => {
            state.brushSize = Number(event.target.value || 22);
        });

        state.canvas.addEventListener('mousedown', event => {
            if (!state.originalImage) return;
            const point = eventPoint(event);
            state.cursorPoint = point;
            state.isDrawing = true;
            state.lastPoint = point;
            if (state.tool === 'rect') {
                state.rectStart = point;
            } else if (state.tool === 'branch') {
                pushUndoState();
                if (!removeBranchAt(point)) {
                    undoProfileEdit();
                    if (window.PyPotteryUtils) window.PyPotteryUtils.showToast('No removable thin branch found near that click.', 'warning');
                } else {
                    redraw();
                }
                state.isDrawing = false;
                state.lastPoint = null;
            } else {
                pushUndoState();
                drawBrush(point, state.tool === 'eraser');
            }
        });
        state.canvas.addEventListener('mousemove', event => {
            if (!state.originalImage) return;
            state.cursorPoint = eventPoint(event);
            if (!state.isDrawing) {
                redraw();
                return;
            }
            const point = state.cursorPoint;
            if (state.tool === 'rect' && state.rectStart) {
                redraw({ x1: state.rectStart.x, y1: state.rectStart.y, x2: point.x, y2: point.y });
            } else {
                drawBrush(point, state.tool === 'eraser');
            }
        });
        state.canvas.addEventListener('mouseup', async event => {
            if (!state.isDrawing) return;
            const point = eventPoint(event);
            if (state.tool === 'rect' && state.rectStart) {
                try {
                    pushUndoState();
                    await rerunProfileInRect({
                        x1: state.rectStart.x, y1: state.rectStart.y,
                        x2: point.x, y2: point.y,
                    });
                } catch (error) {
                    undoProfileEdit();
                    if (window.PyPotteryUtils) window.PyPotteryUtils.showToast(error.message, 'error');
                    else alert(error.message);
                }
            }
            state.isDrawing = false;
            state.rectStart = null;
            state.lastPoint = null;
        });
        state.canvas.addEventListener('mouseleave', () => {
            state.isDrawing = false;
            state.rectStart = null;
            state.lastPoint = null;
            state.cursorPoint = null;
            redraw();
        });
        state.canvas.addEventListener('mouseenter', event => {
            if (!state.originalImage) return;
            state.cursorPoint = eventPoint(event);
            redraw();
        });
        document.addEventListener('keydown', event => {
            if (!document.getElementById('profiles-tab')?.classList.contains('active')) return;
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
                event.preventDefault();
                undoProfileEdit();
            }
        });
    }

    document.addEventListener('DOMContentLoaded', initializeProfileReview);
    window.loadProfileReview = loadProfileReview;
})();
