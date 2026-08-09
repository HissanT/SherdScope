(function () {
    const state = {
        entries: [],
        filtered: [],
        currentIndex: -1,
        current: null,
        image: null,
        maskCanvas: document.createElement('canvas'),
        maskCtx: null,
        canvas: null,
        ctx: null,
        tool: 'brush',
        brushSize: 20,
        drawing: false,
        lastPoint: null,
        panStart: null,
        zoom: 1,
        baseScale: 1,
        undo: [],
        dirty: false,
        cursorPoint: null,
        maskVariant: 'original',
    };

    const $ = id => document.getElementById(id);

    function setStatus(message, kind = '') {
        const element = $('status');
        element.textContent = message || '';
        element.className = `status ${kind}`;
    }

    function loadImage(url) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error(`Could not load ${url}`));
            const inlineImage = url.startsWith('data:') || url.startsWith('blob:');
            image.src = inlineImage ? url : `${url}?v=${Date.now()}`;
        });
    }

    async function refreshEntries(keepCurrent = true) {
        const response = await fetch('/api/entries');
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'Could not load candidates');
        const currentId = keepCurrent ? state.current?.training_id : null;
        state.entries = data.entries || [];
        $('counts').textContent =
            `${data.counts.pending || 0} pending · ${data.counts.approved || 0} approved · ${data.counts.rejected || 0} rejected`;
        applyFilter();
        renderQueue();
        if (!state.filtered.length) {
            state.current = null;
            $('candidate-name').textContent = 'No candidates in this filter';
            return;
        }
        const index = Math.max(0, state.filtered.findIndex(item => item.training_id === currentId));
        await select(index);
    }

    function applyFilter() {
        const query = $('search').value.trim().toLowerCase();
        const decision = $('decision-filter').value;
        state.filtered = state.entries.filter(item =>
            (decision === 'all' || item.decision === decision) &&
            (!query || item.training_id.toLowerCase().includes(query)));
    }

    function renderQueue() {
        const list = $('queue-list');
        list.innerHTML = state.filtered.map((item, index) => `
            <button class="queue-item ${index === state.currentIndex ? 'active' : ''}"
                    type="button" data-index="${index}" title="${item.training_id}">
                <span class="queue-state">${item.decision}</span>
                <span class="queue-name">${item.training_id}</span>
            </button>`).join('');
        list.querySelectorAll('[data-index]').forEach(button => {
            button.addEventListener('click', () => select(Number(button.dataset.index)));
        });
        list.querySelector('.active')?.scrollIntoView({ block: 'nearest' });
    }

    function editorSize() {
        const wrap = $('canvas-scroll');
        const maxWidth = Math.max(320, wrap.clientWidth - 2);
        const maxHeight = Math.max(420, wrap.clientHeight - 2);
        state.baseScale = Math.min(
            1,
            maxWidth / state.image.width,
            maxHeight / state.image.height,
        );
        const scale = state.baseScale * state.zoom;
        state.canvas.width = Math.max(1, Math.round(state.image.width * scale));
        state.canvas.height = Math.max(1, Math.round(state.image.height * scale));
        $('zoom-level').textContent = `${Math.round(state.zoom * 100)}%`;
    }

    function redraw() {
        if (!state.image || !state.maskCtx) return;
        const canvas = state.canvas;
        state.ctx.clearRect(0, 0, canvas.width, canvas.height);
        state.ctx.imageSmoothingEnabled = true;
        state.ctx.imageSmoothingQuality = 'high';
        state.ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);

        const overlay = document.createElement('canvas');
        overlay.width = state.maskCanvas.width;
        overlay.height = state.maskCanvas.height;
        const context = overlay.getContext('2d');
        const source = state.maskCtx.getImageData(0, 0, overlay.width, overlay.height);
        const output = context.createImageData(overlay.width, overlay.height);
        for (let index = 0; index < source.data.length; index += 4) {
            if (source.data[index] > 32) {
                output.data[index] = 37;
                output.data[index + 1] = 99;
                output.data[index + 2] = 235;
                output.data[index + 3] = 72;
            }
        }
        context.putImageData(output, 0, 0);
        state.ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
        drawCursorRing();
        $('mask-preview').src = state.maskCanvas.toDataURL('image/png');
    }

    function drawCursorRing() {
        if (!state.cursorPoint || !['brush', 'eraser'].includes(state.tool)) return;
        const scaleX = state.canvas.width / state.maskCanvas.width;
        const scaleY = state.canvas.height / state.maskCanvas.height;
        const x = state.cursorPoint.x * scaleX;
        const y = state.cursorPoint.y * scaleY;
        const radius = Math.max(1.5, state.brushSize * (scaleX + scaleY) / 4);
        state.ctx.save();
        state.ctx.beginPath();
        state.ctx.arc(x, y, radius, 0, Math.PI * 2);
        state.ctx.strokeStyle = 'rgba(255, 255, 255, .98)';
        state.ctx.lineWidth = 2;
        state.ctx.stroke();
        state.ctx.beginPath();
        state.ctx.arc(x, y, radius, 0, Math.PI * 2);
        state.ctx.strokeStyle = 'rgba(12, 20, 34, .98)';
        state.ctx.lineWidth = .8;
        state.ctx.stroke();
        state.ctx.restore();
    }

    async function select(index) {
        if (index < 0 || index >= state.filtered.length) return;
        if (state.dirty && !window.confirm('Discard unsaved edits to this mask?')) return;
        state.currentIndex = index;
        state.current = state.filtered[index];
        renderQueue();
        setStatus('Loading candidate…');
        const [image, mask] = await Promise.all([
            loadImage(state.current.image_url),
            loadImage(state.current.mask_url),
        ]);
        state.image = image;
        state.maskCanvas.width = image.width;
        state.maskCanvas.height = image.height;
        state.maskCtx = state.maskCanvas.getContext('2d', { willReadFrequently: true });
        state.maskCtx.fillStyle = 'black';
        state.maskCtx.fillRect(0, 0, image.width, image.height);
        state.maskCtx.imageSmoothingEnabled = false;
        state.maskCtx.drawImage(mask, 0, 0, image.width, image.height);
        state.zoom = 1;
        state.undo = [];
        state.dirty = false;
        state.cursorPoint = null;
        state.maskVariant = 'original';
        $('use-original').disabled = true;
        $('use-unet').disabled = !state.current.prediction_url;
        $('review-note').value = state.current.review_note || '';
        $('candidate-name').textContent = state.current.training_id;
        $('candidate-meta').textContent =
            `${state.current.image_width}×${state.current.image_height} · PDF page ${Number(state.current.pdf_page_index) + 1} · ${state.current.source_render_dpi}→${state.current.training_render_dpi} DPI · ${state.current.decision}`;
        editorSize();
        redraw();
        setStatus('');
        $('previous').disabled = index === 0;
        $('next').disabled = index === state.filtered.length - 1;
    }

    function sourcePoint(event) {
        const rect = state.canvas.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(state.maskCanvas.width - 1,
                (event.clientX - rect.left) * state.maskCanvas.width / rect.width)),
            y: Math.max(0, Math.min(state.maskCanvas.height - 1,
                (event.clientY - rect.top) * state.maskCanvas.height / rect.height)),
        };
    }

    function pushUndo() {
        state.undo.push(state.maskCtx.getImageData(
            0, 0, state.maskCanvas.width, state.maskCanvas.height));
        if (state.undo.length > 20) state.undo.shift();
    }

    function undo() {
        const prior = state.undo.pop();
        if (!prior) return;
        state.maskCtx.putImageData(prior, 0, 0);
        state.dirty = true;
        redraw();
    }

    function draw(point) {
        const erase = state.tool === 'eraser';
        const context = state.maskCtx;
        context.save();
        context.strokeStyle = erase ? 'black' : 'white';
        context.fillStyle = erase ? 'black' : 'white';
        context.lineWidth = state.brushSize;
        context.lineCap = 'round';
        context.lineJoin = 'round';
        const prior = state.lastPoint || point;
        context.beginPath();
        context.moveTo(prior.x, prior.y);
        context.lineTo(point.x, point.y);
        context.stroke();
        context.beginPath();
        context.arc(point.x, point.y, state.brushSize / 2, 0, Math.PI * 2);
        context.fill();
        context.restore();
        state.lastPoint = point;
        state.dirty = true;
        redraw();
    }

    function setTool(tool) {
        state.tool = tool;
        document.querySelectorAll('[data-tool]').forEach(button =>
            button.classList.toggle('active', button.dataset.tool === tool));
        state.canvas.style.cursor = tool === 'pan' ? 'grab' : 'none';
        redraw();
    }

    function setZoom(value) {
        if (!state.image) return;
        state.zoom = Math.max(.5, Math.min(8, value));
        editorSize();
        redraw();
    }

    function payload() {
        return {
            mask_data: state.maskCanvas.toDataURL('image/png'),
            review_note: $('review-note').value || '',
        };
    }

    async function save(action) {
        if (!state.current) return;
        setStatus(`${action === 'approve' ? 'Approving' : action === 'reject' ? 'Rejecting' : 'Saving draft'}…`);
        const options = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(action === 'reject'
                ? { review_note: $('review-note').value || '' }
                : payload()),
        };
        const response = await fetch(`/api/entries/${encodeURIComponent(state.current.training_id)}/${action}`, options);
        const data = await response.json();
        if (!data.success) throw new Error(data.error || `${action} failed`);
        state.dirty = false;
        const removed = Number(data.entry?.approval_cleanup?.removed_pixels || 0);
        const cleanup = action === 'approve' && removed
            ? ` ${removed} isolated speck pixel${removed === 1 ? '' : 's'} removed.`
            : '';
        setStatus(action === 'draft' ? 'Draft saved.' : `Candidate ${action}d.${cleanup}`, 'success');
        if (action !== 'draft') {
            await refreshEntries(false);
        }
    }

    async function recoverProfile() {
        if (!state.current) return;
        const button = $('recover');
        button.disabled = true;
        setStatus('Checking the high-resolution crop for omitted profile ink...');
        try {
            const response = await fetch(
                `/api/entries/${encodeURIComponent(state.current.training_id)}/recover`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload()),
                });
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Recovery failed');
            const recovered = await loadImage(data.mask_data);
            pushUndo();
            state.maskCtx.fillStyle = 'black';
            state.maskCtx.fillRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
            state.maskCtx.imageSmoothingEnabled = false;
            state.maskCtx.drawImage(
                recovered, 0, 0, state.maskCanvas.width, state.maskCanvas.height);
            state.dirty = true;
            redraw();
            setStatus(
                data.added_pixels
                    ? `Recovered ${data.added_pixels} profile pixels. Inspect them, then approve or Undo.`
                    : 'The high-resolution proposal found no missing profile pixels.',
                'success');
        } finally {
            button.disabled = false;
        }
    }

    async function useMaskVariant(variant) {
        if (!state.current) return;
        const url = variant === 'unet'
            ? state.current.prediction_url
            : state.current.mask_url;
        if (!url) throw new Error('No U-Net prediction exists for this candidate');
        if (state.dirty && !window.confirm('Replace the edits currently on screen?')) return;
        const replacement = await loadImage(url);
        pushUndo();
        state.maskCtx.fillStyle = 'black';
        state.maskCtx.fillRect(0, 0, state.maskCanvas.width, state.maskCanvas.height);
        state.maskCtx.imageSmoothingEnabled = false;
        state.maskCtx.drawImage(
            replacement, 0, 0, state.maskCanvas.width, state.maskCanvas.height);
        state.maskVariant = variant;
        state.dirty = variant === 'unet';
        $('use-original').disabled = variant === 'original';
        $('use-unet').disabled = variant === 'unet' || !state.current.prediction_url;
        redraw();
        setStatus(
            variant === 'unet'
                ? 'Showing the optional U-Net mask. Inspect it before approval.'
                : 'Restored the original SherdScope automatic mask.',
            'success');
    }

    function typingTarget(event) {
        const target = event.target;
        return target instanceof HTMLElement && (
            target.isContentEditable ||
            ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
    }

    function initialize() {
        state.canvas = $('editor-canvas');
        state.ctx = state.canvas.getContext('2d');
        $('search').addEventListener('input', () => {
            applyFilter(); state.currentIndex = -1; renderQueue();
        });
        $('decision-filter').addEventListener('change', () => {
            applyFilter(); state.currentIndex = -1; renderQueue();
            if (state.filtered.length) select(0);
        });
        document.querySelectorAll('[data-tool]').forEach(button =>
            button.addEventListener('click', () => setTool(button.dataset.tool)));
        $('brush-size').addEventListener('input', event => {
            state.brushSize = Number(event.target.value || 20);
            redraw();
        });
        $('undo').addEventListener('click', undo);
        $('use-original').addEventListener('click', () =>
            useMaskVariant('original').catch(error => setStatus(error.message, 'error')));
        $('use-unet').addEventListener('click', () =>
            useMaskVariant('unet').catch(error => setStatus(error.message, 'error')));
        $('recover').addEventListener('click', () =>
            recoverProfile().catch(error => setStatus(error.message, 'error')));
        $('zoom-in').addEventListener('click', () => setZoom(state.zoom * 1.25));
        $('zoom-out').addEventListener('click', () => setZoom(state.zoom / 1.25));
        $('fit').addEventListener('click', () => setZoom(1));
        $('previous').addEventListener('click', () => select(state.currentIndex - 1));
        $('next').addEventListener('click', () => select(state.currentIndex + 1));
        $('save-draft').addEventListener('click', () => save('draft').catch(error => setStatus(error.message, 'error')));
        $('approve').addEventListener('click', () => save('approve').catch(error => setStatus(error.message, 'error')));
        $('reject').addEventListener('click', () => save('reject').catch(error => setStatus(error.message, 'error')));

        state.canvas.addEventListener('pointerdown', event => {
            if (!state.current) return;
            if (state.tool === 'pan') {
                state.panStart = {
                    x: event.clientX,
                    y: event.clientY,
                    left: $('canvas-scroll').scrollLeft,
                    top: $('canvas-scroll').scrollTop,
                };
                state.canvas.setPointerCapture(event.pointerId);
                return;
            }
            pushUndo();
            state.drawing = true;
            state.lastPoint = sourcePoint(event);
            state.cursorPoint = state.lastPoint;
            draw(state.lastPoint);
            state.canvas.setPointerCapture(event.pointerId);
        });
        state.canvas.addEventListener('pointermove', event => {
            state.cursorPoint = sourcePoint(event);
            if (state.panStart) {
                $('canvas-scroll').scrollLeft =
                    state.panStart.left - (event.clientX - state.panStart.x);
                $('canvas-scroll').scrollTop =
                    state.panStart.top - (event.clientY - state.panStart.y);
            } else if (state.drawing) {
                draw(state.cursorPoint);
            } else {
                redraw();
            }
        });
        state.canvas.addEventListener('pointerenter', event => {
            state.cursorPoint = sourcePoint(event);
            redraw();
        });
        state.canvas.addEventListener('pointerleave', () => {
            if (state.drawing) return;
            state.cursorPoint = null;
            redraw();
        });
        const finish = () => {
            state.drawing = false;
            state.lastPoint = null;
            state.panStart = null;
        };
        state.canvas.addEventListener('pointerup', finish);
        state.canvas.addEventListener('pointercancel', finish);
        state.canvas.addEventListener('wheel', event => {
            if (!event.ctrlKey) return;
            event.preventDefault();
            setZoom(state.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15));
        }, { passive: false });

        document.addEventListener('keydown', event => {
            if (event.repeat || typingTarget(event)) return;
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
                event.preventDefault(); undo(); return;
            }
            if (event.ctrlKey || event.metaKey || event.altKey) return;
            const key = event.key.toLowerCase();
            const actions = {
                a: () => save('approve'),
                r: () => save('reject'),
                s: () => save('draft'),
                b: () => setTool('brush'),
                e: () => setTool('eraser'),
                g: () => recoverProfile(),
                o: () => useMaskVariant('original'),
                u: () => useMaskVariant('unet'),
                p: () => setTool('pan'),
                f: () => setZoom(1),
                '+': () => setZoom(state.zoom * 1.25),
                '=': () => setZoom(state.zoom * 1.25),
                '-': () => setZoom(state.zoom / 1.25),
                arrowleft: () => select(state.currentIndex - 1),
                arrowright: () => select(state.currentIndex + 1),
            };
            if (!actions[key]) return;
            event.preventDefault();
            Promise.resolve(actions[key]()).catch(error => setStatus(error.message, 'error'));
        });
        window.addEventListener('resize', () => {
            if (!state.image) return;
            editorSize(); redraw();
        });
        refreshEntries(false).catch(error => setStatus(error.message, 'error'));
    }

    document.addEventListener('DOMContentLoaded', initialize);
})();
