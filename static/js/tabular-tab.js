// Tabular Tab JavaScript - Project-aware version
// Uses window.PyPotteryUtils.* functions directly

// State
let tabularState = {
    currentProject: null,
    cards: [],
    currentIndex: 0,
    totalCards: 0,
    tableData: [],
    columns: [],
    currentImageName: null,  // Track current image for saving
    imageList: [],  // List of all images with reviewed status
    isReviewed: false,  // Current image reviewed status
    fullImageUrl: null,  // Full resolution image URL for zoom
    showBoxes: true,  // Toggle bounding-box overlay on the page image
    metadataLinkState: null,
    metadataLinkPoll: null,
    metadataSaveTimers: new Map(),
    metadataDirty: new Set(),
    metadataUndo: new Map(),
    metadataSnapshots: new Map()
};

document.addEventListener('DOMContentLoaded', () => {
    setupTabularListeners();
    loadCurrentProject();
    
    // Listen for project changes
    window.addEventListener('projectChanged', (e) => {
        const project = e.detail && e.detail.project ? e.detail.project : null;
        tabularState.currentProject = project;
        tabularState.metadataDirty.clear();
        tabularState.metadataUndo.clear();
        tabularState.metadataSnapshots.clear();
        tabularState.metadataSaveTimers.forEach(timer => clearTimeout(timer));
        tabularState.metadataSaveTimers.clear();
        loadProjectCards();
    });
});

function loadCurrentProject() {
    if (window.projectManager && window.projectManager.getCurrentProject) {
        tabularState.currentProject = window.projectManager.getCurrentProject();
    } else {
        const pid = localStorage.getItem('currentProjectId');
        const pname = localStorage.getItem('currentProjectName');
        if (pid) {
            tabularState.currentProject = { project_id: pid, project_name: pname || 'Unnamed' };
        }
    }
    
    if (tabularState.currentProject) {
        loadProjectCards();
    }
}

function setupTabularListeners() {
    // Navigation
    document.getElementById('tabular-prev')?.addEventListener('click', () => navigateTabular(-1));
    document.getElementById('tabular-next')?.addEventListener('click', () => navigateTabular(1));
    document.getElementById('tabular-goto-btn')?.addEventListener('click', handleTabularGoto);

    // Add column
    document.getElementById('add-column-btn')?.addEventListener('click', handleAddColumn);

    // Clear current page's table (keep ID/index)
    document.getElementById('clear-table-btn')?.addEventListener('click', handleClearTable);

    // Toggle bounding-box overlay
    document.getElementById('toggle-boxes')?.addEventListener('change', (e) => {
        tabularState.showBoxes = e.target.checked;
        redrawTabularCanvas();
    });

    // AI bibliographic extraction
    document.getElementById('ai-bibliographic-btn')?.addEventListener('click', handleAiBibliographic);
    document.getElementById('ai-bibliographic-batch-btn')?.addEventListener('click', handleAiBibliographicBatch);
    document.getElementById('metadata-link-run-btn')?.addEventListener('click', runMetadataLinking);
    document.getElementById('metadata-link-refresh-btn')?.addEventListener('click', loadMetadataLinkState);

    // AI backend toggle panel
    document.getElementById('ai-backend-toggle-btn')?.addEventListener('click', () => {
        const panel = document.getElementById('ai-backend-panel');
        if (panel) panel.classList.toggle('show');
    });

    // Show/hide OpenRouter config based on radio selection
    document.querySelectorAll('input[name="ai-backend-choice"]').forEach(radio => {
        radio.addEventListener('change', () => {
            const isOpenRouter = document.getElementById('ai-backend-openrouter')?.checked;
            const configEl = document.getElementById('ai-openrouter-config');
            if (configEl) {
                if (isOpenRouter) configEl.classList.add('show');
                else configEl.classList.remove('show');
            }
            localStorage.setItem('pypottery_ai_backend', isOpenRouter ? 'openrouter' : 'local');
        });
    });

    // Restore saved backend choice from localStorage
    const _savedBackend = localStorage.getItem('pypottery_ai_backend');
    if (_savedBackend === 'openrouter') {
        const radioEl = document.getElementById('ai-backend-openrouter');
        if (radioEl) {
            radioEl.checked = true;
            const configEl = document.getElementById('ai-openrouter-config');
            if (configEl) configEl.classList.add('show');
        }
    }

    // Persist OpenRouter API key and model in sessionStorage (not localStorage for security)
    const _orKey = document.getElementById('ai-openrouter-apikey');
    const _orModel = document.getElementById('ai-openrouter-model');
    if (_orKey) {
        const _savedKey = sessionStorage.getItem('pypottery_or_apikey');
        if (_savedKey) _orKey.value = _savedKey;
        _orKey.addEventListener('input', () => sessionStorage.setItem('pypottery_or_apikey', _orKey.value));
    }
    if (_orModel) {
        const _savedModel = localStorage.getItem('pypottery_or_model');
        if (_savedModel) _orModel.value = _savedModel;
        _orModel.addEventListener('input', () => localStorage.setItem('pypottery_or_model', _orModel.value));
    }

    // Prompt customisation panel toggle
    document.getElementById('ai-prompt-toggle-btn')?.addEventListener('click', () => {
        const panel = document.getElementById('ai-prompt-panel');
        if (panel) panel.classList.toggle('show');
    });
    document.getElementById('ai-prompt-reset-btn')?.addEventListener('click', () => {
        const ta = document.getElementById('ai-prompt-suffix');
        if (ta) ta.value = '';
        localStorage.removeItem('pypottery_ai_prompt');
        _showPromptSaveIndicator('Reset');
    });

    // Load saved prompt from localStorage and auto-save on change
    const _promptTa = document.getElementById('ai-prompt-suffix');
    if (_promptTa) {
        const _saved = localStorage.getItem('pypottery_ai_prompt');
        if (_saved) _promptTa.value = _saved;

        let _promptSaveTimer = null;
        _promptTa.addEventListener('input', () => {
            clearTimeout(_promptSaveTimer);
            _promptSaveTimer = setTimeout(() => {
                localStorage.setItem('pypottery_ai_prompt', _promptTa.value);
                _showPromptSaveIndicator('Saved');
            }, 600);
        });
    }

    // Mark as reviewed
    document.getElementById('tabular-mark-reviewed-btn')?.addEventListener('click', markAsReviewed);
    
    // Setup magnifying glass zoom on hover
    setupMagnifyingGlass();
}

async function loadProjectCards() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        showEmptyState('No project selected', 'Select a project from the Project Manager tab');
        return;
    }
    
    try {
        window.PyPotteryUtils.showLoading('Loading project cards...');
        
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/cards`
        );
        
        window.PyPotteryUtils.hideLoading();
        
        if (response.success) {
            tabularState.cards = response.cards || [];
            tabularState.totalCards = response.total || 0;
            
            if (tabularState.totalCards === 0) {
                showEmptyState('No cards found', 'Extract cards from masks in the Annotation tab first');
                return;
            }
            
            // Load first card data
            await loadTabularData(0);
            await loadMetadataLinkState();
        } else {
            showEmptyState('Error loading cards', response.error);
        }
        
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error loading project cards:', error);
        showEmptyState('Error', error.message);
    }
}

function showEmptyState(title, message) {
    const canvas = document.getElementById('tabular-canvas');
    const tableContainer = document.getElementById('tabular-table-container');
    
    if (canvas) {
        const ctx = canvas.getContext('2d');
        canvas.width = 400;
        canvas.height = 300;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#64748b';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(title, canvas.width / 2, canvas.height / 2 - 20);
        ctx.fillText(message, canvas.width / 2, canvas.height / 2 + 20);
    }
    
    if (tableContainer) {
        tableContainer.innerHTML = `
            <div style="padding: 2rem; text-align: center; color: #64748b;">
                <h3>${title}</h3>
                <p>${message}</p>
            </div>
        `;
    }
}

async function loadTabularData(imgNum) {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        return;
    }

    try {
        window.PyPotteryUtils.showLoading('Loading card data...');
        
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/tabular/load`,
            {
                method: 'POST',
                body: JSON.stringify({
                    img_num: imgNum
                })
            }
        );

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            tabularState.currentIndex = response.current;
            tabularState.totalCards = response.total;
            tabularState.tableData = response.table;
            tabularState.columns = response.columns;

            displayTabularData(response);
        } else {
            window.PyPotteryUtils.showToast('Failed to load data', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error loading tabular data:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

function displayTabularData(data) {
    // Close any open bbox editor when switching images
    closeBboxEditor();

    // Store image_name and other metadata
    tabularState.currentImageName = data.image_name;
    tabularState.imageList = data.image_list || [];
    tabularState.isReviewed = data.is_reviewed || false;
    tabularState.fullImageUrl = data.full_image_url;
    tabularState.currentIndex = data.current;
    tabularState.totalCards = data.total;

    // Update reviewed status button
    updateReviewedButton();
    
    // Display image list sidebar
    displayImageList();

    // Display image with annotations
    if (data.image) {
        displayAnnotatedImage(data.image, data.annotations);
    }

    // Display table
    if (data.table && data.columns) {
        displayTable(data.table, data.columns);
    }
}

function _drawAnnotations(ctx, img, annotations, hoveredKey) {
    ctx.drawImage(img, 0, 0);
    // Boxes can be hidden to inspect the clean drawing
    if (!tabularState.showBoxes) return;
    if (!annotations || annotations.length === 0) return;

    const fontSize = Math.max(24, Math.min(42, Math.round(img.width / 34)));
    ctx.font = `bold ${fontSize}px Arial`;

    annotations.forEach(annot => {
        const [x1, y1, x2, y2] = annot.bbox;
        const label = annot.label;
        const hovered = (String(annot.row_key || '') === String(hoveredKey || ''));
        const color = hovered ? '#f97316' : '#2563eb';   // orange when hovered, blue otherwise

        // Semi-transparent fill on hover
        if (hovered) {
            ctx.fillStyle = 'rgba(249, 115, 22, 0.18)';
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        }

        // Box stroke (thicker on hover)
        ctx.strokeStyle = color;
        ctx.lineWidth = hovered ? 5 : 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label tag
        const labelText = `No. ${label}`;
        const textWidth = ctx.measureText(labelText).width;
        const textHeight = Math.ceil(fontSize * 1.25);
        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - textHeight - 6), textWidth + 16, textHeight + 6);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x1 + 8, Math.max(fontSize, y1 - 10));
    });
}

// Redraw the current page image + boxes using stored references
function redrawTabularCanvas() {
    const canvas = document.getElementById('tabular-canvas');
    if (!canvas || !tabularState._bboxImg) return;
    const ctx = canvas.getContext('2d');
    _drawAnnotations(ctx, tabularState._bboxImg, tabularState.annotations, null);
}

function displayAnnotatedImage(imageData, annotations) {
    const canvas = document.getElementById('tabular-canvas');
    if (!canvas) {
        console.error('[Tabular] Canvas not found!');
        return;
    }

    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;

        // Store image reference so hover redraws can use it
        tabularState._bboxImg = img;
        tabularState.annotations = annotations || [];

        _drawAnnotations(ctx, img, tabularState.annotations, null);
        _setupBboxInteraction(canvas, tabularState.annotations);
    };

    img.onerror = (error) => {
        console.error('[Tabular] Image load error:', error);
    };

    img.src = imageData;
}

function _setupBboxInteraction(canvas, annotations) {
    // Remove previous handlers to avoid accumulation
    if (tabularState._bboxClickHandler) {
        canvas.removeEventListener('click', tabularState._bboxClickHandler, true);
    }
    if (tabularState._bboxMoveHandler) {
        canvas.removeEventListener('mousemove', tabularState._bboxMoveHandler);
    }

    if (!annotations || annotations.length === 0) return;

    // Use capture=true so our handler fires before the zoom handler;
    // if a bbox is hit we stop propagation to prevent zoom.
    tabularState._bboxClickHandler = (e) => {
        const hit = _hitTestBbox(e, canvas, annotations);
        if (!hit) { closeBboxEditor(); return; }
        e.stopImmediatePropagation();
        const rowIndex = tabularState.tableData.findIndex(r =>
            String(r.mask_file || '').replace(/\.png$/i, '') === String(hit.row_key || ''));
        if (rowIndex === -1) return;
        highlightTableRow(String(hit.row_key));
        showBboxEditor(rowIndex, hit.label, e.clientX, e.clientY);
    };
    canvas.addEventListener('click', tabularState._bboxClickHandler, true);

    // Mousemove: change cursor + redraw with hover highlight
    let _lastHovered = null;
    const ctx = canvas.getContext('2d');
    tabularState._bboxMoveHandler = (e) => {
        const hit = _hitTestBbox(e, canvas, annotations);
        const hLabel = hit ? hit.row_key : null;
        canvas.style.cursor = hit ? 'pointer' : 'zoom-in';
        if (hLabel !== _lastHovered) {
            _lastHovered = hLabel;
            if (tabularState._bboxImg) {
                _drawAnnotations(ctx, tabularState._bboxImg, annotations, hLabel);
            }
        }
    };
    canvas.addEventListener('mousemove', tabularState._bboxMoveHandler);
}

function _hitTestBbox(e, canvas, annotations) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / canvas.clientWidth;
    const scaleY = canvas.height / canvas.clientHeight;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;
    // Among all boxes containing the click, pick the SMALLEST one so that a
    // nested piece (whose box sits inside a larger one) is still selectable.
    let best = null;
    let bestArea = Infinity;
    for (const annot of annotations) {
        const [x1, y1, x2, y2] = annot.bbox;
        if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) {
            const area = Math.abs((x2 - x1) * (y2 - y1));
            if (area < bestArea) { bestArea = area; best = annot; }
        }
    }
    return best;
}

function showBboxEditor(rowIndex, label, clientX, clientY) {
    closeBboxEditor();

    const row = tabularState.tableData[rowIndex];
    if (!row) return;

    const editableCols = tabularState.columns;

    const el = document.createElement('div');
    el.id = 'bbox-editor';
    el.className = 'bbox-editor';

    // Header
    const title = document.createElement('div');
    title.className = 'bbox-editor-title';
    title.innerHTML = `<span>Vessel No. ${label}</span>`;
    const closeBtn = document.createElement('button');
    closeBtn.className = 'bbox-editor-close';
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', closeBboxEditor);
    title.appendChild(closeBtn);
    el.appendChild(title);

    // Fields
    const fieldsDiv = document.createElement('div');
    fieldsDiv.className = 'bbox-editor-fields';
    editableCols.forEach(col => {
        const fieldDiv = document.createElement('div');
        fieldDiv.className = 'bbox-editor-field';
        const lbl = document.createElement('label');
        lbl.textContent = col;
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.value = row[col] || '';
        inp.dataset.col = col;
        inp.dataset.row = rowIndex;
        // Live update state on change
        inp.addEventListener('change', (e) => {
            if (tabularState.tableData[rowIndex]) {
                tabularState.tableData[rowIndex][col] = e.target.value;
                // Sync the main table input if visible
                const tableInput = document.querySelector(
                    `#table-body input[data-row="${rowIndex}"][data-col="${col}"]`
                );
                if (tableInput) tableInput.value = e.target.value;
            }
        });
        fieldDiv.appendChild(lbl);
        fieldDiv.appendChild(inp);
        fieldsDiv.appendChild(fieldDiv);
    });
    el.appendChild(fieldsDiv);

    // Save button
    const saveBtn = document.createElement('button');
    saveBtn.className = 'bbox-editor-save';
    saveBtn.textContent = '✓ Save';
    saveBtn.addEventListener('click', async () => {
        await saveTabularData();
        closeBboxEditor();
    });
    el.appendChild(saveBtn);

    // Position (viewport-relative, clamped to stay visible)
    document.body.appendChild(el);
    const pw = el.offsetWidth, ph = el.offsetHeight;
    let left = clientX + 12, top = clientY - 20;
    if (left + pw > window.innerWidth - 8) left = clientX - pw - 12;
    if (top + ph > window.innerHeight - 8) top = window.innerHeight - ph - 8;
    if (top < 8) top = 8;
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;

    // Close on outside click
    setTimeout(() => {
        document.addEventListener('click', _outsideEditorClick, true);
    }, 0);
}

function _outsideEditorClick(e) {
    const el = document.getElementById('bbox-editor');
    if (el && !el.contains(e.target)) {
        closeBboxEditor();
    }
}

function closeBboxEditor() {
    document.removeEventListener('click', _outsideEditorClick, true);
    const el = document.getElementById('bbox-editor');
    if (el) el.remove();
    // Clear table row highlight
    document.querySelectorAll('.data-table tr.bbox-highlighted').forEach(tr => {
        tr.classList.remove('bbox-highlighted');
    });
}

function highlightTableRow(rowId) {
    // Remove previous highlight
    document.querySelectorAll('.data-table tr.bbox-highlighted').forEach(tr => {
        tr.classList.remove('bbox-highlighted');
    });
    const tr = document.querySelector(`#table-body tr[data-row-id="${rowId}"]`);
    if (tr) {
        tr.classList.add('bbox-highlighted');
        // Scroll WITHIN the table container only — never move the whole page
        const container = document.getElementById('tabular-table-container');
        if (container) {
            const cRect = container.getBoundingClientRect();
            const rRect = tr.getBoundingClientRect();
            if (rRect.top < cRect.top) {
                container.scrollTop -= (cRect.top - rRect.top) + 8;
            } else if (rRect.bottom > cRect.bottom) {
                container.scrollTop += (rRect.bottom - cRect.bottom) + 8;
            }
        }
    }
}

function displayTable(data, columns) {
    const headerEl = document.getElementById('table-header');
    const bodyEl = document.getElementById('table-body');

    if (!headerEl || !bodyEl) return;

    // Clear existing content
    headerEl.innerHTML = '';
    bodyEl.innerHTML = '';

    // Create header
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    headerEl.appendChild(headerRow);

    // Create body
    data.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        tr.dataset.rowId = String(row.mask_file || rowIndex).replace(/\.png$/i, '');
        columns.forEach(col => {
            const td = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'text';
            input.value = row[col] || '';
            input.dataset.row = rowIndex;
            input.dataset.col = col;
            input.addEventListener('change', handleCellChange);
            td.appendChild(input);
            tr.appendChild(td);
        });
        bodyEl.appendChild(tr);
    });
}

async function handleCellChange(e) {
    const rowIndex = parseInt(e.target.dataset.row);
    const column = e.target.dataset.col;
    const value = e.target.value;

    // Update local state
    if (tabularState.tableData[rowIndex]) {
        tabularState.tableData[rowIndex][column] = value;
    }

    // Auto-save
    await saveTabularData();
}

async function saveTabularData() {
    if (!tabularState.currentProject || !tabularState.tableData.length) return;

    try {
        await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/tabular/save`,
            {
                method: 'POST',
                body: JSON.stringify({
                    table: tabularState.tableData,
                    image_name: tabularState.currentImageName  // Include current image name
                })
            }
        );
        console.log('Table auto-saved');
    } catch (error) {
        console.error('Error saving table:', error);
    }
}

async function handleClearTable() {
    if (!tabularState.tableData || tabularState.tableData.length === 0) {
        window.PyPotteryUtils.showToast('Nothing to clear', 'warning');
        return;
    }
    if (!confirm('Clear all values on this page? The vessel number is kept. This cannot be undone.')) {
        return;
    }
    // Empty every column except the ID/index
    tabularState.tableData = tabularState.tableData.map(row => {
        const cleared = {};
        for (const col of Object.keys(row)) {
            cleared[col] = (col === 'No.' || col === 'mask_file') ? row[col] : '';
        }
        return cleared;
    });
    displayTable(tabularState.tableData, tabularState.columns);
    await saveTabularData();
    window.PyPotteryUtils.showToast('Table cleared', 'success');
}

async function handleAddColumn() {
    const input = document.getElementById('new-column-name');
    const columnName = input.value.trim();

    if (!columnName) {
        window.PyPotteryUtils.showToast('Please enter a column name', 'warning');
        return;
    }

    try {
        const response = await window.PyPotteryUtils.apiRequest('/api/tabular/add-column', {
            method: 'POST',
            body: JSON.stringify({
                column_name: columnName,
                table: tabularState.tableData
            })
        });

        if (response.success) {
            tabularState.tableData = response.table;
            tabularState.columns = response.columns;
            displayTable(tabularState.tableData, tabularState.columns);
            input.value = '';
            window.PyPotteryUtils.showToast('Column added successfully', 'success');
            await saveTabularData();
        } else {
            window.PyPotteryUtils.showToast('Failed to add column', 'error');
        }
    } catch (error) {
        console.error('Error adding column:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

function navigateTabular(direction) {
    const newIndex = tabularState.currentIndex + direction;
    if (newIndex >= 0 && newIndex < tabularState.totalCards) {
        loadTabularData(newIndex);
    }
}

function handleTabularGoto() {
    const input = document.getElementById('tabular-goto');
    if (!input) return;
    
    const index = parseInt(input.value);

    if (!isNaN(index) && index >= 0 && index < tabularState.totalCards) {
        loadTabularData(index);
        input.value = '';
    } else {
        window.PyPotteryUtils.showToast('Invalid card number', 'warning');
    }
}

function exportToCSV() {
    if (!tabularState.tableData || tabularState.tableData.length === 0) {
        window.PyPotteryUtils.showToast('No data to export', 'warning');
        return;
    }
    
    try {
        // Convert table data to CSV
        const columns = tabularState.columns;
        const rows = tabularState.tableData;
        
        // Create CSV header
        let csv = columns.join(',') + '\n';
        
        // Add rows
        rows.forEach(row => {
            const values = columns.map(col => {
                const value = row[col] || '';
                // Escape quotes and wrap in quotes if contains comma
                return value.includes(',') ? `"${value.replace(/"/g, '""')}"` : value;
            });
            csv += values.join(',') + '\n';
        });
        
        // Create download link
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        const projectName = tabularState.currentProject ? tabularState.currentProject.project_name : 'project';
        link.setAttribute('href', url);
        link.setAttribute('download', `${projectName}_tabular_data.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        window.PyPotteryUtils.showToast('CSV exported successfully', 'success');
        
    } catch (error) {
        console.error('Error exporting CSV:', error);
        window.PyPotteryUtils.showToast('Failed to export CSV', 'error');
    }
}

function displayImageList() {
    const listContainer = document.getElementById('tabular-image-list');
    if (!listContainer) return;
    
    listContainer.innerHTML = '';
    
    if (!tabularState.imageList || tabularState.imageList.length === 0) {
        listContainer.innerHTML = '<div style="padding: 1rem; color: #64748b;">No images</div>';
        return;
    }
    
    tabularState.imageList.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'tabular-image-item';
        if (index === tabularState.currentIndex) {
            div.classList.add('active');
        }
        if (item.reviewed) {
            div.classList.add('reviewed');
        }
        
        div.innerHTML = `
            <span class="image-name">${item.image_name}</span>
            <span class="status-icon">${item.reviewed ? '✅' : '⚪'}</span>
        `;
        
        div.addEventListener('click', () => {
            loadTabularData(index);
        });
        
        listContainer.appendChild(div);
    });
}

function updateReviewedButton() {
    const btn = document.getElementById('tabular-mark-reviewed-btn');
    if (!btn) return;
    
    if (tabularState.isReviewed) {
        btn.textContent = '✅ Reviewed';
        btn.disabled = true;
        btn.style.opacity = '0.6';
    } else {
        btn.textContent = '👁️ Mark as Reviewed';
        btn.disabled = false;
        btn.style.opacity = '1';
    }
}

const HESBAN_LINK_FIELDS = [
    ['table_no', 'No.'], ['table_type', 'Type'], ['table_square', 'Sq'],
    ['table_locus', 'Loc'], ['table_pail', 'Pail'], ['table_registration', 'Reg'],
    ['fabric_exterior', 'Exterior'], ['fabric_core', 'Core'], ['fabric_interior', 'Interior'],
    ['nonplastics_type', 'Typ'], ['nonplastics_size', 'Siz'],
    ['nonplastics_shape', 'Shap'], ['nonplastics_density', 'Den'],
    ['voids_type_size', 'Ty/Sz'], ['voids_density', 'Den'], ['manufacture', 'Man'],
    ['surface_exterior', 'Ext'], ['surface_exterior_color', 'Color'],
    ['surface_interior', 'Int'], ['surface_interior_color', 'Color'],
    ['decor', 'Decor'], ['fire', 'Fire']
];
const HESBAN_LINK_COLUMNS = HESBAN_LINK_FIELDS.map(field => field[0]);

function hesbanGroupedHeaders() {
    return `<tr class="metadata-link-group-header">
        <th rowspan="2" class="sticky-col sticky-actions">Actions</th>
        <th rowspan="2" class="sticky-col sticky-no">No.</th>
        <th rowspan="2" class="sticky-col sticky-type">Type</th>
        <th rowspan="2">Sq</th><th rowspan="2">Loc</th><th rowspan="2">Pail</th><th rowspan="2">Reg</th>
        <th colspan="3">Fabric Color</th><th colspan="4">Non-Plastics</th>
        <th colspan="2">Voids</th><th rowspan="2">Man</th>
        <th colspan="4">Surface Treatment</th><th rowspan="2">Decor</th><th rowspan="2">Fire</th>
    </tr><tr class="metadata-link-sub-header">
        <th>Exterior</th><th>Core</th><th>Interior</th>
        <th>Typ</th><th>Siz</th><th>Shap</th><th>Den</th>
        <th>Ty/Sz</th><th>Den</th>
        <th>Ext</th><th>Color</th><th>Int</th><th>Color</th>
    </tr>`;
}

function linkEscape(value) {
    const div = document.createElement('div');
    div.textContent = value == null ? '' : String(value);
    // innerHTML escapes markup delimiters but not quotes, while this helper is
    // also used inside value/title/data attributes populated from AI output.
    return div.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

async function loadMetadataLinkState() {
    if (!tabularState.currentProject?.project_id) return;
    try {
        const data = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata-link/state`);
        if (!data.success) return;
        tabularState.metadataLinkState = data.state;
        // The tabular page is commonly already open when the background OCR
        // finishes. Update its box labels immediately; otherwise every box
        // remains the initial ``?`` until the user navigates away and back.
        const stagedNumbers = new Map((data.state?.figures || []).flatMap(figure =>
            (figure.drawings || []).map(drawing => [
                String(drawing.mask_file || '').replace(/\.png$/i, ''),
                String(drawing.vessel_number || '').trim()
            ])));
        let labelsChanged = false;
        (tabularState.annotations || []).forEach(annotation => {
            const number = stagedNumbers.get(String(annotation.row_key || ''));
            if (number && annotation.label !== number) {
                annotation.label = number;
                labelsChanged = true;
            }
        });
        if (labelsChanged) redrawTabularCanvas();
        tabularState.metadataLinkOcrAvailable = data.ocr_available;
        const sourceSelect = document.getElementById('metadata-link-source');
        if (sourceSelect) {
            const previous = sourceSelect.value;
            sourceSelect.innerHTML = data.sources.map(source =>
                `<option value="${linkEscape(source)}">${linkEscape(source)}</option>`).join('');
            if (data.sources.includes(previous)) sourceSelect.value = previous;
            sourceSelect.hidden = data.sources.length <= 1;
        }
        renderMetadataLinkState(data.state, data.active);
        clearTimeout(tabularState.metadataLinkPoll);
        if (data.active || data.state.status === 'running') {
            tabularState.metadataLinkPoll = setTimeout(loadMetadataLinkState, 1500);
        }
    } catch (error) {
        const summary = document.getElementById('metadata-link-summary');
        if (summary) summary.textContent = `Linkage state unavailable: ${error.message}`;
    }
}

async function runMetadataLinking() {
    if (!tabularState.currentProject?.project_id) return;
    const button = document.getElementById('metadata-link-run-btn');
    const source = document.getElementById('metadata-link-source')?.value || '';
    const backend = document.getElementById('metadata-link-backend')?.value || 'ocr';
    try {
        if (button) button.disabled = true;
        const backendParams = {backend};
        if (backend === 'openrouter') {
            backendParams.openrouter_api_key = document.getElementById('ai-openrouter-apikey')?.value.trim() || '';
            backendParams.openrouter_model = document.getElementById('ai-openrouter-model')?.value.trim() || 'google/gemini-2.5-flash';
        }
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata-link/run`, {
                method: 'POST',
                body: JSON.stringify({...backendParams, source_pdf: source || null})
            });
        if (!response.success) throw new Error(response.error || 'Could not start linking');
        window.PyPotteryUtils.showToast('Figure-table linking started', 'success');
        await loadMetadataLinkState();
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
    } finally {
        if (button) button.disabled = false;
    }
}

function renderMetadataLinkState(state, active) {
    const summary = document.getElementById('metadata-link-summary');
    const progress = document.getElementById('metadata-link-progress');
    const progressBar = document.getElementById('metadata-link-progress-bar');
    const figuresContainer = document.getElementById('metadata-link-figures');
    const figures = state?.figures || [];
    const matches = figures.flatMap(figure => figure.matches || []);
    const ready = matches.filter(match => match.status === 'ready').length;
    const unresolved = matches.length - ready;
    const approved = figures.filter(figure => figure.review_status === 'approved').length;
    if (summary) {
        const baseSummary = state?.status === 'error'
            ? `Error: ${state.error || state.progress?.message || 'Unknown error'}`
            : `${figures.length} figures | ${ready} ready matches | ${unresolved} unresolved | ${approved} approved`;
        summary.textContent = baseSummary + (tabularState.metadataLinkOcrAvailable === false
            ? ' | Local OCR is not installed yet (install requirements-ocr.txt).'
            : '');
    }
    const current = Number(state?.progress?.current || 0);
    const total = Number(state?.progress?.total || 0);
    if (progress && progressBar) {
        progress.hidden = !(active || state?.status === 'running');
        progressBar.style.width = `${total ? Math.round(current / total * 100) : 3}%`;
        progress.title = state?.progress?.message || '';
    }
    if (!figuresContainer) return;
    // Progress polling runs every 1.5 seconds while OCR is active. Preserve
    // the reviewer's UI state before replacing the updated figure markup;
    // otherwise every poll closes the figure they are currently inspecting.
    const renderProject = String(tabularState.currentProject?.project_id || '');
    const hadPreviousRender = figuresContainer.dataset.renderProject === renderProject;
    const editingCurrentFigure = hadPreviousRender && (
        tabularState.metadataDirty.size > 0 || figuresContainer.contains(document.activeElement));
    if (editingCurrentFigure) return;
    const openFigures = new Set(
        [...figuresContainer.querySelectorAll('.metadata-link-figure[open]')]
            .map(element => element.dataset.linkFigure));
    const tableScroll = new Map(
        [...figuresContainer.querySelectorAll('.metadata-link-figure')].map(element => {
            const wrap = element.querySelector('.metadata-link-table-wrap');
            return [element.dataset.linkFigure, {
                left: wrap?.scrollLeft || 0,
                top: wrap?.scrollTop || 0
            }];
        }));
    figuresContainer.innerHTML = figures.map(renderMetadataFigure).join('');
    figures.forEach(figure => tabularState.metadataSnapshots.set(
        figure.figure_id, structuredClone(figure)));
    figuresContainer.dataset.renderProject = renderProject;
    if (hadPreviousRender) {
        figuresContainer.querySelectorAll('.metadata-link-figure').forEach(element => {
            element.open = openFigures.has(element.dataset.linkFigure);
            const wrap = element.querySelector('.metadata-link-table-wrap');
            const saved = tableScroll.get(element.dataset.linkFigure);
            if (wrap && saved) {
                wrap.scrollLeft = saved.left;
                wrap.scrollTop = saved.top;
            }
        });
    }
    figuresContainer.querySelectorAll('[data-link-save]').forEach(button =>
        button.addEventListener('click', () => saveMetadataFigure(button.dataset.linkSave)
            .catch(error => window.PyPotteryUtils.showToast(error.message, 'error'))));
    figuresContainer.querySelectorAll('[data-link-approve]').forEach(button =>
        button.addEventListener('click', () => approveMetadataFigure(button.dataset.linkApprove)));
    figuresContainer.querySelectorAll('[data-link-reject]').forEach(button =>
        button.addEventListener('click', () => rejectMetadataFigure(button.dataset.linkReject)));
    figuresContainer.querySelectorAll('[data-link-add-row]').forEach(button =>
        button.addEventListener('click', () => addMetadataTableRow(button.dataset.linkAddRow)));
    figuresContainer.querySelectorAll('[data-link-delete-row]').forEach(button =>
        button.addEventListener('click', () => deleteMetadataTableRow(
            button.dataset.linkDeleteRow, Number(button.dataset.rowIndex))));
    figuresContainer.querySelectorAll('[data-link-duplicate-row]').forEach(button =>
        button.addEventListener('click', () => duplicateMetadataTableRow(
            button.dataset.linkDuplicateRow, Number(button.dataset.rowIndex))));
    figuresContainer.querySelectorAll('[data-link-sort-rows]').forEach(button =>
        button.addEventListener('click', () => sortMetadataTableRows(button.dataset.linkSortRows)));
    figuresContainer.querySelectorAll('[data-link-undo-row]').forEach(button =>
        button.addEventListener('click', () => undoMetadataRowDelete(button.dataset.linkUndoRow)));
    figuresContainer.querySelectorAll('[data-link-restore]').forEach(button =>
        button.addEventListener('click', () => restoreMetadataFigure(button.dataset.linkRestore)));
    figuresContainer.querySelectorAll('[data-warning-toggle]').forEach(button =>
        button.addEventListener('click', () => toggleMetadataWarning(
            button.dataset.figureId, button.dataset.warningToggle)));
    figuresContainer.querySelectorAll('[data-warning-focus]').forEach(button =>
        button.addEventListener('click', () => focusMetadataWarning(
            button.dataset.figureId, button.dataset.warningFocus)));
    figuresContainer.querySelectorAll('[data-link-rerun]').forEach(button =>
        button.addEventListener('click', () => rerunMetadataFigure(button.dataset.linkRerun)));
    figuresContainer.querySelectorAll('.metadata-link-table textarea').forEach(textarea => {
        const resize = () => {
            textarea.style.height = 'auto';
            textarea.style.height = `${Math.max(48, textarea.scrollHeight)}px`;
        };
        textarea.addEventListener('input', resize);
        resize();
    });
    figuresContainer.querySelectorAll(
        '[data-link-figure-id], [data-link-caption], [data-link-table-pages], ' +
        '[data-link-drawing-number], [data-link-column], [data-warning-reason], [data-warning-note]'
    ).forEach(input => {
        input.addEventListener('input', () => {
            input.classList.add('metadata-link-edited');
            scheduleMetadataAutosave(input.closest('[data-link-figure]').dataset.linkFigure);
        });
        input.addEventListener('change', () =>
            scheduleMetadataAutosave(input.closest('[data-link-figure]').dataset.linkFigure));
    });
    figuresContainer.querySelectorAll('[data-link-drawing-number]').forEach(input => {
        input.addEventListener('focus', () => selectMetadataDrawing(
            input.closest('[data-link-figure]').dataset.linkFigure, input));
        input.addEventListener('click', () => selectMetadataDrawing(
            input.closest('[data-link-figure]').dataset.linkFigure, input));
    });
    figuresContainer.querySelectorAll('[data-link-figure]').forEach(element =>
        validateMetadataFigureDom(element.dataset.linkFigure));
}

function renderMetadataFigure(figure) {
    const projectId = tabularState.currentProject.project_id;
    const processing = figure.processing_status === 'processing';
    const status = figure.review_status === 'approved' ? 'approved'
        : processing ? 'processing' : figure.status;
    const disabled = processing ? 'disabled' : '';
    const evidencePages = [
        ...(figure.drawing_pages || []).map(page => ({...page, kind: 'drawing'})),
        ...(figure.table_pages || []).map(page => ({...page, kind: 'table'}))
    ];
    const pageHtml = evidencePages.map(page => {
        const evidenceUrl = `/api/projects/${encodeURIComponent(projectId)}/metadata-link/evidence/${encodeURIComponent(page.image_name)}` +
            `?figure=${encodeURIComponent(figure.figure_id)}&kind=${page.kind}&overlay=1&v=${encodeURIComponent(tabularState.metadataLinkState?.updated_at || '')}`;
        return `<a href="${evidenceUrl}" target="_blank">
            <img src="${evidenceUrl}" data-evidence-kind="${page.kind}" alt="${linkEscape(page.image_name)}"
                 title="${linkEscape(page.kind)}: ${linkEscape(page.image_name)}">
        </a>`;
    }).join('');
    const drawings = (figure.drawings || []).map(drawing => `
        <label class="metadata-link-drawing" data-mask-file="${linkEscape(drawing.mask_file)}">
            <span title="${linkEscape(drawing.mask_file)}">Printed No.</span>
            <input class="form-control" data-link-drawing-number data-mask-file="${linkEscape(drawing.mask_file)}"
                   value="${linkEscape(drawing.vessel_number || '')}" aria-label="Printed vessel number" ${disabled}>
        </label>`).join('');
    const rows = (figure.table_rows || []).map((row, rowIndex) => `
        <tr data-link-row="${rowIndex}">
            <td class="sticky-col sticky-actions metadata-link-row-actions">
                <button type="button" aria-label="Duplicate row ${rowIndex + 1}" title="Duplicate row"
                        data-link-duplicate-row="${linkEscape(figure.figure_id)}" data-row-index="${rowIndex}" ${disabled}>⧉</button>
                <button type="button" aria-label="Delete row ${rowIndex + 1}" title="Delete row"
                        data-link-delete-row="${linkEscape(figure.figure_id)}" data-row-index="${rowIndex}" ${disabled}>🗑</button>
            </td>${HESBAN_LINK_COLUMNS.map((column, columnIndex) =>
            `<td class="${columnIndex === 0 ? 'sticky-col sticky-no' : columnIndex === 1 ? 'sticky-col sticky-type' : ''}"><textarea data-link-column="${column}" ${disabled}>${linkEscape(row[column] || '')}</textarea></td>`).join('')}</tr>`).join('');
    const overrides = figure.warning_overrides || {};
    const warnings = (figure.warnings || []).map(warning => {
        const active = !!warning.overridden;
        const reasonOptions = warning.code === 'missing_table_end'
            ? [['visually_confirmed_complete', 'Visually confirmed table ending']]
            : warning.code === 'missing_required_value'
                ? [['publication_field_blank', 'Publication field is genuinely blank']]
                : [['column_alignment_verified', 'Column alignment visually verified']];
        const override = overrides[warning.id] || {};
        const controls = warning.overrideable ? `<div class="metadata-link-warning-review">
            <select data-warning-reason ${disabled}>${reasonOptions.map(([value, label]) =>
                `<option value="${value}" ${override.reason === value ? 'selected' : ''}>${label}</option>`).join('')}</select>
            <input data-warning-note placeholder="Optional reviewer note" value="${linkEscape(override.note || '')}" ${disabled}>
            <button type="button" data-warning-toggle="${warning.id}" data-figure-id="${linkEscape(figure.figure_id)}" ${disabled}>
                ${active ? 'Remove override' : 'Mark reviewed and ignore'}
            </button></div>` : '';
        const focus = warning.row || warning.mask_file ? `<button type="button" class="metadata-warning-focus"
            data-warning-focus="${warning.id}" data-figure-id="${linkEscape(figure.figure_id)}">Go to problem</button>` : '';
        return `<article class="metadata-link-warning-card ${warning.blocking ? 'blocking' : 'resolved'}"
                         data-warning-id="${warning.id}" data-warning-code="${warning.code}"
                         data-warning-row="${linkEscape(warning.row || '')}"
                         data-warning-mask="${linkEscape(warning.mask_file || '')}"
                         data-override-active="${active ? '1' : '0'}">
            <div><strong>${active ? 'Reviewed' : warning.blocking ? 'Needs attention' : 'Information'}</strong>
            <span>${linkEscape(warning.message)}</span>${focus}</div>${controls}</article>`;
    }).join('');
    const tablePageNames = (figure.table_pages || []).map(page => page.image_name).join(', ');
    const open = figure.status === 'needs_review' ? 'open' : '';
    const blockers = (figure.warnings || []).filter(warning => warning.blocking).length;
    const unmatched = (figure.matches || []).filter(match => match.status !== 'ready').length;
    return `<details class="metadata-link-figure" data-link-figure="${linkEscape(figure.figure_id)}"
                    data-reviewer-revision="${Number(figure.reviewer_revision || 0)}" ${open}>
        <summary><strong>Figure ${linkEscape(figure.figure_id)}</strong>
            <span class="metadata-link-badge ${linkEscape(status)}">${linkEscape(status)}</span>
            <span>${(figure.drawings || []).length} drawings / ${(figure.table_rows || []).length} rows</span>
        </summary>
        <div class="metadata-link-figure-body">
            <div class="metadata-link-save-strip">
                <strong>Review workspace</strong>
                <span data-link-save-status aria-live="polite">${processing ? 'OCR processing…' : 'Saved'}</span>
            </div>
            <label>Figure ID <input class="form-control" data-link-figure-id value="${linkEscape(figure.figure_id || '')}" ${disabled}></label>
            <label>Caption <input class="form-control" data-link-caption value="${linkEscape(figure.figure_caption || '')}" ${disabled}></label>
            <label>Table page image names (comma-separated)
                <input class="form-control" data-link-table-pages value="${linkEscape(tablePageNames)}" ${disabled}>
            </label>
            <div class="metadata-link-evidence">
                <div><h4>Source evidence</h4><div class="metadata-link-pages">${pageHtml}</div></div>
                <div class="metadata-link-number-workspace"><h4>Drawing numbers</h4>
                    <p>Click a number to highlight its linked table row and drawing box.</p>
                    <div class="metadata-link-drawings">${drawings}</div></div>
            </div>
            <div class="metadata-link-table-section"><div class="metadata-link-table-toolbar">
                <h4>Extracted table</h4>
                <button type="button" data-link-add-row="${linkEscape(figure.figure_id)}" ${disabled}>Add row</button>
                <button type="button" data-link-sort-rows="${linkEscape(figure.figure_id)}" ${disabled}>Sort by No.</button>
                <button type="button" data-link-undo-row="${linkEscape(figure.figure_id)}" ${disabled}>Undo delete</button>
                <button type="button" data-link-restore="${linkEscape(figure.figure_id)}" ${disabled}>Restore last saved</button>
            </div><div class="metadata-link-table-wrap">
                <table class="metadata-link-table"><thead>${hesbanGroupedHeaders()}</thead><tbody>${rows}</tbody></table>
            </div></div>
            ${warnings ? `<section class="metadata-link-warnings"><h4>Review warnings</h4>${warnings}</section>` : ''}
            <div class="metadata-link-readiness ${blockers || unmatched ? 'blocked' : 'ready'}">
                <strong>CSV readiness</strong>
                <span>${processing ? 'This figure is still processing.' : blockers
                    ? `${blockers} blocking warning(s) remain.` : unmatched
                        ? `${unmatched} drawing/table match(es) remain unresolved.`
                        : 'Unique matches are ready for approval.'}</span>
            </div>
            <div class="metadata-link-review-actions">
                <button class="btn btn-secondary" data-link-save="${linkEscape(figure.figure_id)}" ${disabled}>Save now</button>
                <button class="btn btn-secondary" data-link-rerun="${linkEscape(figure.figure_id)}"
                        ${processing || tabularState.metadataLinkState?.status === 'running' ? 'disabled' : ''}>Rerun OCR for this figure</button>
                <button class="btn btn-success" data-link-approve="${linkEscape(figure.figure_id)}"
                        ${figure.status !== 'ready' || tabularState.metadataLinkState?.status === 'running' ? 'disabled' : ''}>Approve and apply to CSV</button>
                <button class="btn btn-danger" data-link-reject="${linkEscape(figure.figure_id)}" ${disabled}>Reject</button>
            </div>
        </div></details>`;
}

function addMetadataTableRow(figureId) {
    const figureEl = [...document.querySelectorAll('[data-link-figure]')]
        .find(element => element.dataset.linkFigure === figureId);
    const body = figureEl?.querySelector('.metadata-link-table tbody');
    if (!body) return;
    const row = document.createElement('tr');
    row.innerHTML = metadataDynamicRowCells(figureId, body.children.length, {});
    body.appendChild(row);
    wireMetadataDynamicRow(row, figureId);
    scheduleMetadataAutosave(figureId);
    row.querySelector('[data-link-column="table_no"]')?.focus();
}

function metadataDynamicRowCells(figureId, rowIndex, values) {
    return `<td class="sticky-col sticky-actions metadata-link-row-actions">
        <button type="button" title="Duplicate row" aria-label="Duplicate row ${rowIndex + 1}"
                data-link-duplicate-row="${linkEscape(figureId)}" data-row-index="${rowIndex}">⧉</button>
        <button type="button" title="Delete row" aria-label="Delete row ${rowIndex + 1}"
                data-link-delete-row="${linkEscape(figureId)}" data-row-index="${rowIndex}">🗑</button>
    </td>` + HESBAN_LINK_COLUMNS.map((column, columnIndex) =>
        `<td class="${columnIndex === 0 ? 'sticky-col sticky-no' : columnIndex === 1 ? 'sticky-col sticky-type' : ''}"><textarea data-link-column="${column}">${linkEscape(values[column] || '')}</textarea></td>`).join('');
}

function wireMetadataDynamicRow(row, figureId) {
    row.querySelector('[data-link-delete-row]')?.addEventListener('click', event =>
        deleteMetadataTableRow(figureId, Number(event.currentTarget.dataset.rowIndex)));
    row.querySelector('[data-link-duplicate-row]')?.addEventListener('click', event =>
        duplicateMetadataTableRow(figureId, Number(event.currentTarget.dataset.rowIndex)));
    row.querySelectorAll('textarea').forEach(textarea => {
        autoGrowMetadataTextarea(textarea);
        textarea.addEventListener('input', () => {
            autoGrowMetadataTextarea(textarea);
            textarea.classList.add('metadata-link-edited');
            scheduleMetadataAutosave(figureId);
        });
    });
}

function autoGrowMetadataTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.max(44, textarea.scrollHeight)}px`;
}

function reindexMetadataRows(figureId) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    [...(figureEl?.querySelectorAll('.metadata-link-table tbody tr') || [])].forEach((row, index) => {
        row.dataset.linkRow = index;
        row.querySelectorAll('[data-row-index]').forEach(button => button.dataset.rowIndex = index);
    });
    validateMetadataFigureDom(figureId);
}

function readMetadataRow(row) {
    const result = {};
    row.querySelectorAll('[data-link-column]').forEach(input => result[input.dataset.linkColumn] = input.value);
    return result;
}

function deleteMetadataTableRow(figureId, rowIndex) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    const rows = [...(figureEl?.querySelectorAll('.metadata-link-table tbody tr') || [])];
    const row = rows[rowIndex];
    if (!row) return;
    const stack = tabularState.metadataUndo.get(figureId) || [];
    stack.push({index: rowIndex, row: readMetadataRow(row)});
    tabularState.metadataUndo.set(figureId, stack);
    row.remove();
    reindexMetadataRows(figureId);
    scheduleMetadataAutosave(figureId);
}

function undoMetadataRowDelete(figureId) {
    const stack = tabularState.metadataUndo.get(figureId) || [];
    const deleted = stack.pop();
    if (!deleted) return;
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    const body = figureEl?.querySelector('.metadata-link-table tbody');
    if (!body) return;
    const row = document.createElement('tr');
    row.innerHTML = metadataDynamicRowCells(figureId, deleted.index, deleted.row);
    body.insertBefore(row, body.children[deleted.index] || null);
    wireMetadataDynamicRow(row, figureId);
    reindexMetadataRows(figureId);
    scheduleMetadataAutosave(figureId);
}

function duplicateMetadataTableRow(figureId, rowIndex) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    const body = figureEl?.querySelector('.metadata-link-table tbody');
    const source = body?.children[rowIndex];
    if (!source) return;
    const row = document.createElement('tr');
    row.innerHTML = metadataDynamicRowCells(figureId, rowIndex + 1, readMetadataRow(source));
    body.insertBefore(row, body.children[rowIndex + 1] || null);
    wireMetadataDynamicRow(row, figureId);
    reindexMetadataRows(figureId);
    scheduleMetadataAutosave(figureId);
}

function sortMetadataTableRows(figureId) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    const body = figureEl?.querySelector('.metadata-link-table tbody');
    if (!body) return;
    [...body.children].sort((a, b) =>
        (a.querySelector('[data-link-column="table_no"]')?.value || '').localeCompare(
            b.querySelector('[data-link-column="table_no"]')?.value || '', undefined,
            {numeric: true, sensitivity: 'base'})).forEach(row => body.appendChild(row));
    reindexMetadataRows(figureId);
    scheduleMetadataAutosave(figureId);
}

function restoreMetadataFigure(figureId) {
    const snapshot = tabularState.metadataSnapshots.get(figureId);
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    if (!snapshot || !figureEl) return;
    const body = figureEl.querySelector('.metadata-link-table tbody');
    body.innerHTML = (snapshot.table_rows || []).map((row, index) =>
        `<tr data-link-row="${index}">${metadataDynamicRowCells(figureId, index, row)}</tr>`).join('');
    body.querySelectorAll('tr').forEach(row => wireMetadataDynamicRow(row, figureId));
    const numbers = new Map((snapshot.drawings || []).map(drawing => [drawing.mask_file, drawing.vessel_number || '']));
    figureEl.querySelectorAll('[data-link-drawing-number]').forEach(input =>
        input.value = numbers.get(input.dataset.maskFile) || '');
    const figureIdInput = figureEl.querySelector('[data-link-figure-id]');
    const captionInput = figureEl.querySelector('[data-link-caption]');
    const pagesInput = figureEl.querySelector('[data-link-table-pages]');
    if (figureIdInput) figureIdInput.value = snapshot.figure_id || '';
    if (captionInput) captionInput.value = snapshot.figure_caption || '';
    if (pagesInput) pagesInput.value = (snapshot.table_pages || [])
        .map(page => page.image_name).filter(Boolean).join(', ');
    const overrides = snapshot.warning_overrides || {};
    figureEl.querySelectorAll('[data-warning-id]').forEach(card => {
        const override = overrides[card.dataset.warningId];
        card.dataset.overrideActive = override ? '1' : '0';
        card.classList.toggle('resolved', !!override);
        card.classList.toggle('blocking', !override);
        const reason = card.querySelector('[data-warning-reason]');
        const note = card.querySelector('[data-warning-note]');
        const toggle = card.querySelector('[data-warning-toggle]');
        if (reason) reason.value = override?.reason || '';
        if (note) note.value = override?.note || '';
        if (toggle) toggle.textContent = override
            ? 'Remove override' : 'Mark reviewed and ignore';
    });
    tabularState.metadataDirty.delete(figureId);
    setMetadataSaveStatus(figureId, 'Restored');
    validateMetadataFigureDom(figureId);
}

function collectMetadataFigureEdits(figureId) {
    const figureEl = [...document.querySelectorAll('[data-link-figure]')]
        .find(element => element.dataset.linkFigure === figureId);
    if (!figureEl) return null;
    const drawingNumbers = {};
    figureEl.querySelectorAll('[data-link-drawing-number]').forEach(input => {
        drawingNumbers[input.dataset.maskFile] = input.value;
    });
    const tableRows = [...figureEl.querySelectorAll('.metadata-link-table tbody tr')].map(row => {
        return readMetadataRow(row);
    });
    const existing = tabularState.metadataLinkState.figures.find(figure => figure.figure_id === figureId);
    const existingPages = new Map((existing?.table_pages || []).map(page => [page.image_name, page]));
    const pageNames = (figureEl.querySelector('[data-link-table-pages]')?.value || '')
        .split(',').map(value => value.trim()).filter(Boolean);
    const warningOverrides = {};
    figureEl.querySelectorAll('[data-warning-id][data-override-active="1"]').forEach(card => {
        warningOverrides[card.dataset.warningId] = {
            reason: card.querySelector('[data-warning-reason]')?.value || '',
            note: card.querySelector('[data-warning-note]')?.value || ''
        };
    });
    return {
        reviewer_revision: Number(figureEl.dataset.reviewerRevision || existing?.reviewer_revision || 0),
        figure_id: figureEl.querySelector('[data-link-figure-id]')?.value || figureId,
        figure_caption: figureEl.querySelector('[data-link-caption]')?.value || '',
        drawing_numbers: drawingNumbers,
        table_rows: tableRows,
        table_pages: pageNames.map(name => existingPages.get(name) || {image_name: name}),
        warning_overrides: warningOverrides
    };
}

function setMetadataSaveStatus(figureId, text, failed = false) {
    const element = document.querySelector(
        `[data-link-figure="${CSS.escape(figureId)}"] [data-link-save-status]`);
    if (element) {
        element.textContent = text;
        element.classList.toggle('failed', failed);
    }
}

function scheduleMetadataAutosave(figureId) {
    if (!figureId) return;
    tabularState.metadataDirty.add(figureId);
    setMetadataSaveStatus(figureId, 'Unsaved');
    validateMetadataFigureDom(figureId);
    clearTimeout(tabularState.metadataSaveTimers.get(figureId));
    tabularState.metadataSaveTimers.set(figureId, setTimeout(
        () => saveMetadataFigure(figureId, true).catch(() => {
            tabularState.metadataDirty.add(figureId);
            setMetadataSaveStatus(figureId, 'Save failed', true);
        }), 700));
}

async function saveMetadataFigure(figureId, quiet = false, retryConflict = true) {
    const edits = collectMetadataFigureEdits(figureId);
    if (!edits) return null;
    setMetadataSaveStatus(figureId, 'Saving…');
    const raw = await fetch(
        `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(figureId)}`, {
            method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(edits)
        });
    const response = await raw.json();
    if (raw.status === 409 && response.conflict && retryConflict) {
        const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
        if (figureEl) figureEl.dataset.reviewerRevision = response.reviewer_revision;
        return saveMetadataFigure(figureId, quiet, false);
    }
    if (!raw.ok || !response.success) {
        tabularState.metadataDirty.add(figureId);
        setMetadataSaveStatus(figureId, 'Save failed', true);
        if (!quiet) throw new Error(response.error || 'Could not save figure');
        return null;
    }
    const index = tabularState.metadataLinkState.figures.findIndex(figure => figure.figure_id === figureId);
    if (index >= 0) tabularState.metadataLinkState.figures[index] = response.figure;
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    if (figureEl) figureEl.dataset.reviewerRevision = response.reviewer_revision;
    tabularState.metadataSnapshots.set(figureId, structuredClone(response.figure));
    tabularState.metadataDirty.delete(figureId);
    setMetadataSaveStatus(figureId, 'Saved');
    updateMetadataReadinessFromFigure(figureId, response.figure);
    if (!quiet) window.PyPotteryUtils.showToast(`Figure ${figureId} saved`, 'success');
    validateMetadataFigureDom(figureId);
    return response.figure;
}

function updateMetadataReadinessFromFigure(figureId, figure) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    if (!figureEl) return;
    const blockers = (figure.warnings || []).filter(warning => warning.blocking).length;
    const unmatched = (figure.matches || []).filter(match => match.status !== 'ready').length;
    const readiness = figureEl.querySelector('.metadata-link-readiness');
    if (readiness) {
        readiness.classList.toggle('ready', !blockers && !unmatched);
        readiness.classList.toggle('blocked', !!blockers || !!unmatched);
        const text = readiness.querySelector('span');
        if (text) text.textContent = blockers ? `${blockers} blocking warning(s) remain.`
            : unmatched ? `${unmatched} drawing/table match(es) remain unresolved.`
                : 'Unique matches are ready for approval.';
    }
    const badge = figureEl.querySelector('.metadata-link-badge');
    if (badge) {
        badge.textContent = figure.status;
        badge.className = `metadata-link-badge ${figure.status}`;
    }
    const approve = figureEl.querySelector('[data-link-approve]');
    if (approve) approve.disabled = figure.status !== 'ready' ||
        tabularState.metadataLinkState?.status === 'running';
}

function toggleMetadataWarning(figureId, warningId) {
    const card = document.querySelector(
        `[data-link-figure="${CSS.escape(figureId)}"] [data-warning-id="${CSS.escape(warningId)}"]`);
    if (!card) return;
    const active = card.dataset.overrideActive === '1';
    card.dataset.overrideActive = active ? '0' : '1';
    card.classList.toggle('resolved', !active);
    card.classList.toggle('blocking', active);
    const button = card.querySelector('[data-warning-toggle]');
    if (button) button.textContent = active ? 'Mark reviewed and ignore' : 'Remove override';
    scheduleMetadataAutosave(figureId);
}

function focusMetadataWarning(figureId, warningId) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    const card = figureEl?.querySelector(`[data-warning-id="${CSS.escape(warningId)}"]`);
    if (!figureEl || !card) return;
    const mask = card.dataset.warningMask;
    const number = card.dataset.warningRow;
    let target = mask ? figureEl.querySelector(
        `[data-link-drawing-number][data-mask-file="${CSS.escape(mask)}"]`) : null;
    if (!target && number) {
        target = [...figureEl.querySelectorAll('[data-link-column="table_no"]')]
            .find(input => input.value.trim() === number);
    }
    if (target) {
        target.scrollIntoView({behavior: 'smooth', block: 'center', inline: 'center'});
        target.focus();
    }
}

function selectMetadataDrawing(figureId, input) {
    const figureEl = input.closest('[data-link-figure]');
    figureEl.querySelectorAll('.metadata-link-drawing').forEach(element =>
        element.classList.toggle('selected', element.contains(input)));
    const number = input.value.trim();
    figureEl.querySelectorAll('.metadata-link-table tbody tr').forEach(row =>
        row.classList.toggle('metadata-link-row-selected',
            number && row.querySelector('[data-link-column="table_no"]')?.value.trim() === number));
    figureEl.querySelectorAll('[data-evidence-kind="drawing"]').forEach(image => {
        const url = new URL(image.src, window.location.origin);
        url.searchParams.set('highlight', input.dataset.maskFile || '');
        image.src = url.toString();
    });
}

function validateMetadataFigureDom(figureId) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    if (!figureEl) return;
    const drawingInputs = [...figureEl.querySelectorAll('[data-link-drawing-number]')];
    const drawingCounts = new Map();
    drawingInputs.forEach(input => {
        const value = input.value.trim();
        if (value) drawingCounts.set(value, (drawingCounts.get(value) || 0) + 1);
    });
    drawingInputs.forEach(input => {
        const value = input.value.trim();
        const invalid = !/^[1-9]\d*[a-z]?$/i.test(value) || drawingCounts.get(value) > 1;
        input.classList.toggle('metadata-link-invalid', invalid);
        input.setAttribute('aria-invalid', invalid ? 'true' : 'false');
    });
    const rowInputs = [...figureEl.querySelectorAll('[data-link-column="table_no"]')];
    const rowCounts = new Map();
    rowInputs.forEach(input => {
        const value = input.value.trim();
        if (value) rowCounts.set(value, (rowCounts.get(value) || 0) + 1);
    });
    rowInputs.forEach(input => {
        const value = input.value.trim();
        const row = input.closest('tr');
        const invalid = !/^[1-9]\d*[a-z]?$/i.test(value) || rowCounts.get(value) > 1;
        const unmatched = value && !drawingCounts.has(value);
        row.classList.toggle('metadata-link-row-invalid', invalid);
        row.classList.toggle('metadata-link-row-unmatched', unmatched);
        input.setAttribute('aria-invalid', invalid ? 'true' : 'false');
    });
}

async function approveMetadataFigure(figureId) {
    try {
        const figure = await saveMetadataFigure(figureId, true);
        if (!figure || figure.status !== 'ready') throw new Error('Resolve the warnings before approval');
        const approvedFigureId = figure.figure_id || figureId;
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata-link/apply`, {
                method: 'POST', body: JSON.stringify({figure_ids: [approvedFigureId]})
            });
        if (!response.success) throw new Error(response.error || 'Could not apply figure');
        window.PyPotteryUtils.showToast(`Applied ${response.applied_rows} rows to CSV`, 'success');
        await loadMetadataLinkState();
        await loadTabularData(tabularState.currentIndex);
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function rerunMetadataFigure(figureId) {
    try {
        const saved = await saveMetadataFigure(figureId, true);
        const targetId = saved?.figure_id || figureId;
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(targetId)}/rerun`, {
                method: 'POST', body: JSON.stringify({backend: 'ocr'})
            });
        if (!response.success) throw new Error(response.error || 'Could not rerun OCR');
        window.PyPotteryUtils.showToast(`OCR rerun finished for figure ${targetId}`, 'success');
        await loadMetadataLinkState();
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function rejectMetadataFigure(figureId) {
    try {
        await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(figureId)}`, {
                method: 'PATCH', body: JSON.stringify({review_status: 'rejected'})
            });
        await loadMetadataLinkState();
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function markAsReviewed() {
    if (!tabularState.currentProject || !tabularState.currentImageName) return;
    
    try {
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/reviewed`,
            {
                method: 'POST',
                body: JSON.stringify({
                    image_name: tabularState.currentImageName
                })
            }
        );
        
        if (response.success) {
            tabularState.isReviewed = true;
            updateReviewedButton();
            
            // Update image list
            const item = tabularState.imageList.find(i => i.image_name === tabularState.currentImageName);
            if (item) {
                item.reviewed = true;
                displayImageList();
            }
            
            window.PyPotteryUtils.showToast('Marked as reviewed', 'success');
        }
    } catch (error) {
        console.error('Error marking as reviewed:', error);
        window.PyPotteryUtils.showToast('Failed to mark as reviewed', 'error');
    }
}

function openZoomModal() {
    if (!tabularState.fullImageUrl) {
        window.PyPotteryUtils.showToast('Full resolution image not available', 'warning');
        return;
    }
    
    // Create modal
    const modal = document.createElement('div');
    modal.id = 'zoom-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: zoom-out;
    `;
    
    const img = document.createElement('img');
    img.src = tabularState.fullImageUrl;
    img.style.cssText = `
        max-width: 95%;
        max-height: 95%;
        object-fit: contain;
    `;
    
    modal.appendChild(img);
    
    // Close on click
    modal.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    document.body.appendChild(modal);
}

function setupMagnifyingGlass() {
    const canvas = document.getElementById('tabular-canvas');
    const zoomHint = document.querySelector('.zoom-hint');
    if (!canvas) return;
    
    // Simple hover zoom: click to toggle between normal and zoomed view
    canvas.style.cursor = 'zoom-in';
    canvas.style.transition = 'transform 0.3s ease';
    canvas.style.transformOrigin = 'center center';
    
    let isZoomed = false;
    
    canvas.addEventListener('click', (e) => {
        if (isZoomed) {
            // Zoom out
            canvas.style.transform = 'scale(1)';
            canvas.style.cursor = 'zoom-in';
            canvas.style.position = 'relative';
            canvas.style.zIndex = '1';
            if (zoomHint) zoomHint.textContent = 'Click to zoom';
            isZoomed = false;
        } else {
            // Zoom in
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Calculate zoom origin as percentage
            const originX = (x / rect.width) * 100;
            const originY = (y / rect.height) * 100;
            
            canvas.style.transformOrigin = `${originX}% ${originY}%`;
            canvas.style.transform = 'scale(2)';
            canvas.style.cursor = 'zoom-out';
            canvas.style.position = 'relative';
            canvas.style.zIndex = '100';
            if (zoomHint) zoomHint.textContent = 'Click to zoom out';
            isZoomed = true;
        }
    });
    
    // Reset zoom when changing image
    const observer = new MutationObserver(() => {
        if (isZoomed) {
            canvas.style.transform = 'scale(1)';
            canvas.style.cursor = 'zoom-in';
            canvas.style.position = 'relative';
            canvas.style.zIndex = '1';
            if (zoomHint) zoomHint.textContent = 'Click to zoom';
            isZoomed = false;
        }
    });
    
    observer.observe(canvas, { attributes: true, attributeFilter: ['src'] });
}

// Export for use by main.js
window.refreshTabular = loadProjectCards;

/* =========================================================
 * GPU / download confirmation dialog
 * ========================================================= */

/** Return the user-defined prompt context, or empty string if not set. */
function getPromptSuffix() {
    const ta = document.getElementById('ai-prompt-suffix');
    return ta ? ta.value.trim() : '';
}

/** Return current AI backend params to include in every AI request body. */
function getAiBackendParams() {
    const numbers_from_crops = !!document.getElementById('numbers-from-crops')?.checked;
    const isOpenRouter = document.getElementById('ai-backend-openrouter')?.checked;
    if (!isOpenRouter) {
        return { ai_backend: 'local', numbers_from_crops };
    }
    return {
        ai_backend: 'openrouter',
        openrouter_api_key: document.getElementById('ai-openrouter-apikey')?.value.trim() || '',
        openrouter_model: document.getElementById('ai-openrouter-model')?.value.trim() || 'google/gemini-flash-1.5',
        numbers_from_crops,
    };
}

function showVisionUnsupportedDialog(modelName) {
    document.getElementById('ai-vision-unsupported-dialog')?.remove();
    const overlay = document.createElement('div');
    overlay.id = 'ai-vision-unsupported-dialog';
    overlay.style.cssText = `
        position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:20000;
        display:flex; align-items:center; justify-content:center;
    `;
    overlay.innerHTML = `
        <div style="background:#1e293b; color:#e2e8f0; border-radius:12px; padding:2rem;
                    max-width:460px; width:90%; box-shadow:0 20px 60px rgba(0,0,0,0.5);">
            <h3 style="margin:0 0 1rem; font-size:1.2rem; color:#f87171;">⚠️ Model does not support vision</h3>
            <p style="margin:0 0 0.75rem;">
                <code style="color:#f59e0b; background:#0f172a; padding:0.15rem 0.4rem; border-radius:4px;">${modelName}</code>
                does not support image input on OpenRouter.
            </p>
            <p style="margin:0 0 1.25rem; color:#94a3b8; font-size:0.85rem;">
                Please choose a vision-capable model. Browse available models at
                <a href="https://openrouter.ai/models" target="_blank" style="color:#6366f1;">openrouter.ai/models</a>
                and filter by image input support.
            </p>
            <div style="display:flex; justify-content:flex-end;">
                <button id="ai-vision-dialog-ok" class="btn btn-primary">OK, change model</button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    document.getElementById('ai-vision-dialog-ok').addEventListener('click', () => {
        overlay.remove();
        // Open the AI Backend panel so the user can change the model immediately
        const panel = document.getElementById('ai-backend-panel');
        if (panel) panel.style.display = 'block';
    });
}

/** Flash a brief "Saved" / "Reset" badge next to the prompt reset button. */
function _showPromptSaveIndicator(text) {
    const btn = document.getElementById('ai-prompt-reset-btn');
    if (!btn) return;
    let badge = document.getElementById('ai-prompt-save-badge');
    if (!badge) {
        badge = document.createElement('span');
        badge.id = 'ai-prompt-save-badge';
        badge.style.cssText = 'font-size:0.72rem;color:#22c55e;margin-left:0.5rem;opacity:1;transition:opacity 1s ease;';
        btn.parentNode.insertBefore(badge, btn.nextSibling);
    }
    badge.textContent = text === 'Reset' ? '✓ Reset' : '✓ Saved';
    badge.style.color = text === 'Reset' ? '#f59e0b' : '#22c55e';
    badge.style.opacity = '1';
    clearTimeout(badge._hideTimer);
    badge._hideTimer = setTimeout(() => { badge.style.opacity = '0'; }, 2000);
}

async function checkAiRequirements() {
    const res = await window.PyPotteryUtils.apiRequest('/api/check-ai-requirements');
    return res;
}

function showAiConfirmDialog(requirements, onConfirm) {
    // Remove any existing dialog
    document.getElementById('ai-requirements-dialog')?.remove();

    const { cuda_available, vram_gb, gpu_name, model_cached, meets_requirements } = requirements;

    const gpuLine = cuda_available
        ? `<p>GPU detected: <strong>${gpu_name}</strong> (${vram_gb.toFixed(1)} GB VRAM)</p>`
        : `<p style="color:#ef4444;">No CUDA GPU detected on this system.</p>`;

    const downloadNote = model_cached
        ? `<p style="color:#22c55e;">✅ Model already cached locally — no download needed.</p>`
        : `<p style="color:#f59e0b;">⚠️ The Gemma 4 E2B model (~10 GB) will be downloaded the first time. Make sure you have a stable internet connection and enough disk space.</p>`;

    const blocker = !meets_requirements
        ? `<p style="color:#ef4444; font-weight:600;">This feature requires a CUDA GPU with at least 6 GB of VRAM. Your system does not meet this requirement.</p>`
        : '';

    const overlay = document.createElement('div');
    overlay.id = 'ai-requirements-dialog';
    overlay.style.cssText = `
        position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:20000;
        display:flex; align-items:center; justify-content:center;
    `;
    overlay.innerHTML = `
        <div style="background:#1e293b; color:#e2e8f0; border-radius:12px; padding:2rem;
                    max-width:480px; width:90%; box-shadow:0 20px 60px rgba(0,0,0,0.5);">
            <h3 style="margin:0 0 1rem; font-size:1.2rem;">🤖 AI Bibliographic Extraction</h3>
            ${gpuLine}
            ${downloadNote}
            ${blocker}
            <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">
                The model uses the Gemma 4 E2B multimodal architecture from Google and runs
                entirely on your local machine — no data is sent to the cloud.
            </p>
            <div style="display:flex; justify-content:flex-end; gap:0.75rem; margin-top:1.5rem;">
                <button id="ai-dialog-cancel" class="btn btn-secondary">Cancel</button>
                <button id="ai-dialog-confirm" class="btn btn-primary"
                    ${meets_requirements ? '' : 'disabled'}>
                    ${model_cached ? 'Run Extraction' : 'Download & Run'}
                </button>
            </div>
            <div id="ai-download-progress-wrapper" style="display:none; margin-top:1rem;">
                <p id="ai-download-progress-label" style="font-size:0.85rem; color:#94a3b8; margin:0 0 0.4rem;"></p>
                <div style="background:#334155; border-radius:6px; overflow:hidden; height:12px;">
                    <div id="ai-download-progress-bar"
                         style="height:100%; background:#6366f1; transition:width 0.4s; width:0%"></div>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);

    document.getElementById('ai-dialog-cancel').addEventListener('click', () => overlay.remove());
    document.getElementById('ai-dialog-confirm').addEventListener('click', () => {
        document.getElementById('ai-dialog-confirm').disabled = true;
        document.getElementById('ai-dialog-cancel').disabled = true;
        if (!requirements.model_cached) {
            document.getElementById('ai-download-progress-wrapper').style.display = 'block';
        }
        onConfirm(overlay);
    });
}

function startProgressPolling(labelEl, barEl, stopSignal) {
    const interval = setInterval(async () => {
        if (stopSignal.stopped) { clearInterval(interval); return; }
        try {
            const prog = await window.PyPotteryUtils.apiRequest('/api/operation-progress');
            if (prog && prog.active) {
                labelEl.textContent = prog.message || '';
                barEl.style.width = (prog.percent || 0) + '%';
            }
        } catch (_) { /* ignore polling errors */ }
    }, 800);
    return interval;
}

function showBatchProgressOverlay() {
    document.getElementById('ai-batch-progress-overlay')?.remove();
    const overlay = document.createElement('div');
    overlay.id = 'ai-batch-progress-overlay';
    overlay.style.cssText = 'position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:20000; display:flex; align-items:center; justify-content:center;';
    overlay.innerHTML = `
        <div style="background:#1e293b; color:#e2e8f0; border-radius:12px; padding:2rem;
                    max-width:480px; width:90%; box-shadow:0 20px 60px rgba(0,0,0,0.5);">
            <h3 style="margin:0 0 1rem; font-size:1.2rem;">🤖 Batch AI Extraction</h3>
            <p id="ai-batch-progress-label" style="font-size:0.85rem; color:#94a3b8; margin:0 0 0.4rem;">Starting...</p>
            <div style="background:#334155; border-radius:6px; overflow:hidden; height:12px;">
                <div id="ai-batch-progress-bar"
                     style="height:100%; background:#6366f1; transition:width 0.4s; width:0%"></div>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    return overlay;
}

async function handleAiBibliographic() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        window.PyPotteryUtils.showToast('No project selected', 'warning');
        return;
    }

    const statusEl = document.getElementById('ai-bibliographic-status');
    const btn = document.getElementById('ai-bibliographic-btn');
    const backendParams = getAiBackendParams();

    // For OpenRouter, skip GPU check entirely and call directly
    if (backendParams.ai_backend === 'openrouter') {
        if (!backendParams.openrouter_api_key) {
            window.PyPotteryUtils.showToast('Please enter your OpenRouter API key in the AI Backend panel', 'warning');
            document.getElementById('ai-backend-panel').classList.add('show');
            return;
        }
        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Analysing via OpenRouter...';
        window.PyPotteryUtils.showLoading('Extracting references via OpenRouter...');
        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic`,
                { method: 'POST', body: JSON.stringify({ img_num: tabularState.currentIndex, prompt_suffix: getPromptSuffix(), ...backendParams }) }
            );
            window.PyPotteryUtils.hideLoading();
            if (response.success) {
                tabularState.tableData = response.table;
                tabularState.columns = response.columns;
                displayTable(response.table, response.columns);
                if (statusEl) statusEl.textContent = '✅ References extracted successfully';
                window.PyPotteryUtils.showToast('Bibliographic references extracted!', 'success');
            } else {
                if (statusEl) statusEl.textContent = '❌ Error: ' + (response.error || 'unknown');
                window.PyPotteryUtils.showToast(response.error || 'AI Error', 'error');
            }
        } catch (error) {
            window.PyPotteryUtils.hideLoading();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Bibliographic] Error:', error);
        } finally {
            btn.disabled = false;
        }
        return;
    }

    // Local backend: check GPU requirements first
    let requirements;
    try {
        requirements = await checkAiRequirements();
    } catch (e) {
        window.PyPotteryUtils.showToast('Could not check system requirements', 'error');
        return;
    }

    // If model is already cached, skip confirm dialog and run directly
    if (requirements.model_cached) {
        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Analysing with Gemma 4 AI...';
        window.PyPotteryUtils.showLoading('Extracting references with Gemma 4 AI...');
        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic`,
                { method: 'POST', body: JSON.stringify({ img_num: tabularState.currentIndex, prompt_suffix: getPromptSuffix(), ...backendParams }) }
            );
            window.PyPotteryUtils.hideLoading();
            if (response.success) {
                tabularState.tableData = response.table;
                tabularState.columns = response.columns;
                displayTable(response.table, response.columns);
                if (statusEl) statusEl.textContent = '✅ References extracted successfully';
                window.PyPotteryUtils.showToast('Bibliographic references extracted!', 'success');
            } else if (response.vision_unsupported) {
                if (statusEl) statusEl.textContent = '';
                showVisionUnsupportedDialog(backendParams.openrouter_model);
            } else {
                if (statusEl) statusEl.textContent = '❌ Error: ' + (response.error || 'unknown');
                window.PyPotteryUtils.showToast(response.error || 'AI Error', 'error');
            }
        } catch (error) {
            window.PyPotteryUtils.hideLoading();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Bibliographic] Error:', error);
        } finally {
            btn.disabled = false;
        }
        return;
    }

    // Model not yet cached: show confirm dialog with download progress bar
    showAiConfirmDialog(requirements, async (overlay) => {
        const labelEl = document.getElementById('ai-download-progress-label');
        const barEl = document.getElementById('ai-download-progress-bar');
        const stopSignal = { stopped: false };
        const pollInterval = startProgressPolling(labelEl, barEl, stopSignal);

        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Downloading model and analysing...';
        window.PyPotteryUtils.showLoading('Downloading Gemma 4 AI model (~10 GB)...');

        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic`,
                {
                    method: 'POST',
                    body: JSON.stringify({ img_num: tabularState.currentIndex, prompt_suffix: getPromptSuffix(), ...backendParams })
                }
            );

            stopSignal.stopped = true;
            clearInterval(pollInterval);
            window.PyPotteryUtils.hideLoading();
            overlay.remove();

            if (response.success) {
                tabularState.tableData = response.table;
                tabularState.columns = response.columns;
                displayTable(response.table, response.columns);
                if (statusEl) statusEl.textContent = '✅ References extracted successfully';
                window.PyPotteryUtils.showToast('Bibliographic references extracted!', 'success');
            } else if (response.vision_unsupported) {
                if (statusEl) statusEl.textContent = '';
                showVisionUnsupportedDialog(backendParams.openrouter_model);
            } else {
                if (statusEl) statusEl.textContent = '❌ Error: ' + (response.error || 'unknown');
                window.PyPotteryUtils.showToast(response.error || 'AI Error', 'error');
            }
        } catch (error) {
            stopSignal.stopped = true;
            clearInterval(pollInterval);
            window.PyPotteryUtils.hideLoading();
            overlay.remove();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Bibliographic] Error:', error);
        } finally {
            btn.disabled = false;
        }
    });
}

async function handleAiBibliographicBatch() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        window.PyPotteryUtils.showToast('No project selected', 'warning');
        return;
    }

    const statusEl = document.getElementById('ai-bibliographic-status');
    const btn = document.getElementById('ai-bibliographic-batch-btn');
    const backendParams = getAiBackendParams();

    // Helper: run the batch request with a given progress label/bar and overlay
    async function runBatch(overlay, labelEl, barEl) {
        const stopSignal = { stopped: false };
        const pollInterval = startProgressPolling(labelEl, barEl, stopSignal);
        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Running batch extraction...';
        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic-batch`,
                { method: 'POST', body: JSON.stringify({ prompt_suffix: getPromptSuffix(), ...backendParams }) }
            );
            stopSignal.stopped = true;
            clearInterval(pollInterval);
            overlay.remove();
            if (response.success) {
                const errMsg = response.errors && response.errors.length
                    ? ` (${response.errors.length} errors)` : '';
                if (statusEl) statusEl.textContent = `✅ Batch complete: ${response.processed} images${errMsg}`;
                window.PyPotteryUtils.showToast(`Batch extraction done: ${response.processed} images${errMsg}`, 'success');
                await loadTabularData(tabularState.currentIndex);
            } else if (response.vision_unsupported) {
                if (statusEl) statusEl.textContent = '';
                showVisionUnsupportedDialog(backendParams.openrouter_model);
            } else {
                if (statusEl) statusEl.textContent = '❌ Batch error: ' + (response.error || 'unknown');
                window.PyPotteryUtils.showToast(response.error || 'Batch AI Error', 'error');
            }
        } catch (error) {
            stopSignal.stopped = true;
            clearInterval(pollInterval);
            overlay.remove();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Batch] Error:', error);
        } finally {
            btn.disabled = false;
        }
    }

    // For OpenRouter, skip GPU check and run batch directly with progress overlay
    if (backendParams.ai_backend === 'openrouter') {
        if (!backendParams.openrouter_api_key) {
            window.PyPotteryUtils.showToast('Please enter your OpenRouter API key in the AI Backend panel', 'warning');
            document.getElementById('ai-backend-panel').classList.add('show');
            return;
        }
        const overlay = showBatchProgressOverlay();
        const labelEl = document.getElementById('ai-batch-progress-label');
        const barEl = document.getElementById('ai-batch-progress-bar');
        await runBatch(overlay, labelEl, barEl);
        return;
    }

    // Local backend: check GPU requirements first
    let requirements;
    try {
        requirements = await checkAiRequirements();
    } catch (e) {
        window.PyPotteryUtils.showToast('Could not check system requirements', 'error');
        return;
    }

    // If model is already cached, skip confirm dialog and show progress overlay directly
    if (requirements.model_cached) {
        const overlay = showBatchProgressOverlay();
        const labelEl = document.getElementById('ai-batch-progress-label');
        const barEl = document.getElementById('ai-batch-progress-bar');
        await runBatch(overlay, labelEl, barEl);
        return;
    }

    // Model not yet cached: show confirm dialog with download note
    showAiConfirmDialog(requirements, async (overlay) => {
        const labelEl = document.getElementById('ai-download-progress-label');
        const barEl = document.getElementById('ai-download-progress-bar');
        // Always show the progress bar inside the dialog for batch mode
        document.getElementById('ai-download-progress-wrapper').style.display = 'block';
        await runBatch(overlay, labelEl, barEl);
    });
}
