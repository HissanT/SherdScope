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
    metadataSaveQueues: new Map(),
    metadataDirty: new Set(),
    metadataEditVersions: new Map(),
    metadataUndo: new Map(),
    metadataSnapshots: new Map(),
    selectedFigureKey: null,
    metadataEvidenceState: new Map(),
    metadataRenderedFigureSignatures: new Map(),
    metadataProfile: null,
    metadataFigureDetails: new Map(),
    metadataBackendReload: null,
    metadataStaleFigureCount: 0
};

document.addEventListener('DOMContentLoaded', () => {
    setupTabularListeners();
    organizeAdvancedReviewTools();
    loadCurrentProject();
    
    // Listen for project changes
    window.addEventListener('projectChanged', (e) => {
        const project = e.detail && e.detail.project ? e.detail.project : null;
        tabularState.currentProject = project;
        tabularState.metadataDirty.clear();
        tabularState.metadataEditVersions.clear();
        tabularState.metadataUndo.clear();
        tabularState.metadataSnapshots.clear();
        tabularState.metadataEvidenceState.clear();
        tabularState.metadataRenderedFigureSignatures.clear();
        tabularState.metadataFigureDetails.clear();
        tabularState.metadataSaveTimers.forEach(timer => clearTimeout(timer));
        tabularState.metadataSaveTimers.clear();
        tabularState.metadataSaveQueues.clear();
        loadProjectCards();
    });
});

function organizeAdvancedReviewTools() {
    const tab = document.getElementById('tabular-tab');
    const tablePanel = tab?.querySelector('.tabular-table');
    const metadataPanel = document.getElementById('metadata-link-panel');
    if (!tab || !tablePanel || !metadataPanel || tab.querySelector('.metadata-advanced-tools')) return;
    const details = document.createElement('details');
    details.className = 'metadata-advanced-tools';
    details.innerHTML = '<summary>Advanced legacy tools</summary><div></div>';
    const body = details.querySelector('div');
    [tab.querySelector('.navigation-bar'), tab.querySelector('.tabular-sidebar'),
     tab.querySelector('.tabular-image'), ...tablePanel.querySelectorAll(':scope > .form-row, :scope > .collapsible-panel, :scope > #tabular-table-container')]
        .filter(Boolean).forEach(element => body.appendChild(element));
    metadataPanel.insertAdjacentElement('afterend', details);
    tab.querySelector('.tabular-layout')?.classList.add('metadata-primary-layout');
}

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

let HESBAN_LINK_FIELDS = [
    ['table_no', 'No.'], ['table_type', 'Type'], ['table_square', 'Sq/Area'],
    ['table_locus', 'Loc'], ['table_pail', 'Pail'], ['table_registration', 'Reg'],
    ['fabric_exterior', 'Exterior'], ['fabric_core', 'Core'], ['fabric_interior', 'Interior'],
    ['nonplastics_type', 'Typ'], ['nonplastics_size', 'Siz'],
    ['nonplastics_shape', 'Shap'], ['nonplastics_density', 'Den'],
    ['voids_type_size', 'Ty/Sz'], ['voids_density', 'Den'], ['manufacture', 'Man'],
    ['surface_exterior', 'Ext'], ['surface_exterior_color', 'Color'],
    ['surface_interior', 'Int'], ['surface_interior_color', 'Color'],
    ['decor', 'Decor'], ['fire', 'Fire']
];
let HESBAN_LINK_COLUMNS = HESBAN_LINK_FIELDS.map(field => field[0]);

function applyMetadataProfile(profile) {
    if (!profile?.columns?.length) return;
    tabularState.metadataProfile = profile;
    HESBAN_LINK_FIELDS = profile.columns.map(column => [column.key, column.ui_label]);
    HESBAN_LINK_COLUMNS = HESBAN_LINK_FIELDS.map(field => field[0]);
}

function hesbanGroupedHeaders() {
    const columns = tabularState.metadataProfile?.columns || HESBAN_LINK_FIELDS.map(
        ([key, label]) => ({key, ui_label: label, group: 'identity', header_tier: 'primary'}));
    const groupMap = new Map((tabularState.metadataProfile?.groups || []).map(group =>
        [group.key, group]));
    let top = '<th rowspan="2" class="sticky-col sticky-actions">Actions</th>';
    let bottom = '';
    for (let index = 0; index < columns.length;) {
        const column = columns[index];
        if (index === 2) top += '<th rowspan="2" class="metadata-diameter-header">Rim Diameter (cm)</th>';
        const group = groupMap.get(column.group);
        if (column.header_tier === 'secondary' && group?.header_label) {
            const members = [];
            while (index < columns.length && columns[index].group === column.group &&
                   columns[index].header_tier === 'secondary') members.push(columns[index++]);
            top += `<th colspan="${members.length}" data-column-group="${linkEscape(column.group)}">${linkEscape(group.header_label)}</th>`;
            bottom += members.map(member => `<th>${linkEscape(member.ui_label)}</th>`).join('');
            continue;
        }
        const sticky = index === 0 ? ' class="sticky-col sticky-no"' :
            index === 1 ? ' class="sticky-col sticky-type"' : '';
        top += `<th rowspan="2"${sticky} data-column-group="${linkEscape(column.group || '')}">${linkEscape(column.ui_label)}</th>`;
        index += 1;
    }
    return `<tr class="metadata-link-group-header">${top}</tr><tr class="metadata-link-sub-header">${bottom}</tr>`;
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
        applyMetadataProfile(data.profile);
        const summaries = data.state?.figures || [];
        if (!summaries.some(figure =>
            (figure.figure_key || figure.figure_id) === tabularState.selectedFigureKey)) {
            const attention = summaries.find(figure => figure.needs_column_review ||
                (figure.processing_status !== 'processing' && figure.processing_status !== 'queued' &&
                 figure.review_status !== 'approved' && figure.status !== 'ready'));
            const reviewable = summaries.find(figure =>
                !['processing', 'queued'].includes(figure.processing_status));
            const choice = attention || reviewable || summaries[0];
            tabularState.selectedFigureKey = choice ? (choice.figure_key || choice.figure_id) : null;
        }
        if (tabularState.selectedFigureKey) {
            try {
                const detail = await window.PyPotteryUtils.apiRequest(
                    `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(tabularState.selectedFigureKey)}`);
                if (detail.success) tabularState.metadataFigureDetails.set(
                    tabularState.selectedFigureKey, detail.figure);
            } catch (error) {
                console.warn('Could not load selected figure detail', error);
            }
        }
        const displayState = {...data.state, figures: summaries.map(summary => {
            const key = summary.figure_key || summary.figure_id;
            return tabularState.metadataFigureDetails.get(key) || summary;
        })};
        tabularState.metadataLinkState = displayState;
        // The tabular page is commonly already open when the background OCR
        // finishes. Update its box labels immediately; otherwise every box
        // remains the initial ``?`` until the user navigates away and back.
        const stagedNumbers = new Map((displayState.figures || []).flatMap(figure =>
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
        tabularState.metadataLinkOcrHealth = data.ocr_health || null;
        tabularState.metadataBackendReload = data.backend_reload || null;
        tabularState.metadataStaleFigureCount = Number(data.stale_figure_count || 0);
        const sourceSelect = document.getElementById('metadata-link-source');
        if (sourceSelect) {
            const previous = sourceSelect.value;
            sourceSelect.innerHTML = data.sources.map(source =>
                `<option value="${linkEscape(source)}">${linkEscape(source)}</option>`).join('');
            if (data.sources.includes(previous)) sourceSelect.value = previous;
            sourceSelect.hidden = data.sources.length <= 1;
        }
        renderMetadataLinkState(displayState, data.active);
        clearTimeout(tabularState.metadataLinkPoll);
        if (data.active || data.state.status === 'running' || data.jobs?.paused ||
                (data.jobs?.queued || []).length) {
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
    const backend = 'ocr';
    try {
        if (button) {
            button.disabled = true;
            button.textContent = 'Queuing…';
            button.setAttribute('aria-busy', 'true');
        }
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
        if (button) button.textContent = 'Queued';
        window.PyPotteryUtils.showToast('Figure-table linking queued', 'success');
        await loadMetadataLinkState();
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
        if (button) { button.disabled = false; button.textContent = 'Failed'; }
    } finally {
        button?.removeAttribute('aria-busy');
    }
}

async function runMetadataButtonAction(button, runningText, doneText, action) {
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = runningText;
    button.setAttribute('aria-busy', 'true');
    try {
        const result = await action();
        if (button.isConnected) button.textContent = doneText;
        return result;
    } catch (error) {
        if (button.isConnected) button.textContent = 'Failed';
        throw error;
    } finally {
        if (button.isConnected) {
            button.removeAttribute('aria-busy');
            if ([doneText, 'Failed'].includes(button.textContent)) {
                const finalText = button.textContent;
                setTimeout(() => {
                    if (button.isConnected && button.textContent === finalText) {
                        button.disabled = false;
                        button.textContent = originalText;
                    }
                }, 1400);
            } else {
                button.disabled = false;
                button.textContent = originalText;
            }
        }
    }
}

function renderMetadataLinkState(state, active) {
    const summary = document.getElementById('metadata-link-summary');
    const progress = document.getElementById('metadata-link-progress');
    const progressBar = document.getElementById('metadata-link-progress-bar');
    const figuresContainer = document.getElementById('metadata-link-figures');
    const jobsElement = document.getElementById('metadata-link-jobs');
    const figures = state?.figures || [];
    const ready = figures.reduce((total, figure) => total +
        (figure.matches ? figure.matches.filter(match => match.status === 'ready').length :
         Math.max(0, Number(figure.match_count || 0) - Number(figure.unresolved_count || 0))), 0);
    const unresolved = figures.reduce((total, figure) => total +
        (figure.matches ? figure.matches.filter(match => match.status !== 'ready').length :
         Number(figure.unresolved_count || 0)), 0);
    const approved = figures.filter(figure => figure.review_status === 'approved').length;
    if (jobsElement) {
        const restartRequired = tabularState.metadataBackendReload?.required === true;
        const activeJob = state?.jobs?.active;
        const pausedJob = state?.jobs?.paused;
        const queuedJobs = state?.jobs?.queued || [];
        const latestJob = state?.jobs?.recent?.[0];
        jobsElement.hidden = !restartRequired && !activeJob && !pausedJob && !queuedJobs.length && !latestJob;
        jobsElement.innerHTML = [
            restartRequired
                ? `<strong>Restart SherdScope</strong><span class="failed">${linkEscape(tabularState.metadataBackendReload.message)}</span>`
                : '',
            activeJob ? `<strong>${linkEscape(activeJob.progress?.message || activeJob.kind)}</strong><span>Running</span>` : '',
            pausedJob ? `<strong>${linkEscape(pausedJob.progress?.message || pausedJob.kind)}</strong><span>Paused; resumes automatically</span>` : '',
            ...queuedJobs.slice(0, 4).map(job => `<strong>${linkEscape(job.kind.replaceAll('_', ' '))}</strong><span>Queued</span>`),
            latestJob && !activeJob && !pausedJob && !queuedJobs.length
                ? `<strong>${linkEscape(latestJob.kind.replaceAll('_', ' '))}</strong><span class="${latestJob.status === 'failed' ? 'failed' : 'done'}">${latestJob.status === 'failed' ? `Failed: ${linkEscape(latestJob.error || 'Unknown error')}` : 'Done'}</span>` : ''
        ].filter(Boolean).join('');
    }
    const runButton = document.getElementById('metadata-link-run-btn');
    const bulkJob = [state?.jobs?.active, state?.jobs?.paused,
        ...(state?.jobs?.queued || [])].find(job => job?.kind === 'bulk_link');
    if (runButton) {
        const restartRequired = tabularState.metadataBackendReload?.required === true;
        runButton.disabled = !!bulkJob || restartRequired;
        runButton.setAttribute('aria-busy', bulkJob?.status === 'running' ? 'true' : 'false');
        runButton.textContent = restartRequired ? 'Restart SherdScope first' : bulkJob
            ? bulkJob.status === 'running' ? 'Reading and linking…' :
                bulkJob.status === 'paused' ? 'Paused for priority work' : 'Read and Link queued'
            : 'Read and Link Tables';
    }
    if (summary) {
        const activeJob = state?.jobs?.active;
        const pausedJob = state?.jobs?.paused;
        const queueCount = (state?.jobs?.queued || []).length;
        const jobText = activeJob
            ? ` | ${activeJob.progress?.message || activeJob.kind}`
            : pausedJob ? ' | Bulk work paused for priority work' : queueCount ? ` | ${queueCount} queued` : '';
        const baseSummary = state?.status === 'error'
            ? `Error: ${state.error || state.progress?.message || 'Unknown error'}`
            : `${figures.length} figures | ${ready} ready matches | ${unresolved} unresolved | ${approved} approved`;
        const restartText = tabularState.metadataBackendReload?.required
            ? ` | ${tabularState.metadataBackendReload.message}`
            : tabularState.metadataStaleFigureCount
                ? ` | ${tabularState.metadataStaleFigureCount} figure(s) were read by older OCR logic; run Read and Link Tables again.`
                : '';
        summary.textContent = baseSummary + jobText + restartText + (tabularState.metadataLinkOcrAvailable === false
            ? ` | ${tabularState.metadataLinkOcrHealth?.message || 'Local OCR could not start.'}`
            : '');
    }
    const currentProgress = state?.jobs?.active?.progress || state?.progress || {};
    const current = Number(currentProgress.current || 0);
    const total = Number(currentProgress.total || 0);
    if (progress && progressBar) {
        progress.hidden = !(active || state?.status === 'running');
        progressBar.style.width = `${total ? Math.round(current / total * 100) : 3}%`;
        progress.title = currentProgress.message || '';
    }
    if (!figuresContainer) return;
    const selectedExists = figures.some(figure =>
        (figure.figure_key || figure.figure_id) === tabularState.selectedFigureKey);
    if (!selectedExists) {
        const firstAttention = figures.find(figure =>
            !['processing', 'queued'].includes(figure.processing_status) &&
            figure.review_status !== 'approved' &&
            figure.status !== 'ready');
        const firstReviewable = figures.find(figure =>
            !['processing', 'queued'].includes(figure.processing_status));
        const choice = firstAttention || firstReviewable || figures[0];
        tabularState.selectedFigureKey = choice ? (choice.figure_key || choice.figure_id) : null;
    }
    const selectedFigure = figures.find(figure =>
        (figure.figure_key || figure.figure_id) === tabularState.selectedFigureKey) || figures[0];
    const selectedKey = selectedFigure
        ? String(selectedFigure.figure_key || selectedFigure.figure_id || '') : '';
    const selectedSignature = JSON.stringify(selectedFigure || null);
    // Progress polling runs every 1.5 seconds while OCR is active. Preserve
    // the selected figure as a real, mounted DOM node when its data has not
    // changed. Replacing an unchanged publication image and restoring its
    // viewport afterward still creates a visible jump on every poll.
    const renderProject = String(tabularState.currentProject?.project_id || '');
    const hadPreviousRender = figuresContainer.dataset.renderProject === renderProject;
    const editingCurrentFigure = hadPreviousRender && (
        tabularState.metadataDirty.size > 0 || figuresContainer.contains(document.activeElement));
    const forcedRender = figuresContainer.dataset.forceRender === '1';
    if (editingCurrentFigure && !forcedRender) return;
    const renderedKeys = [...figuresContainer.querySelectorAll('[data-select-figure]')]
        .map(button => button.dataset.selectFigure || '');
    const currentKeys = figures.map(figure => String(figure.figure_key || figure.figure_id || ''));
    const sameFigureList = renderedKeys.length === currentKeys.length &&
        renderedKeys.every((key, index) => key === currentKeys[index]);
    const sameSelectedFigure = hadPreviousRender && sameFigureList && !forcedRender &&
        figuresContainer.dataset.activeFigureKey === selectedKey &&
        tabularState.metadataRenderedFigureSignatures.get(selectedKey) === selectedSignature;
    if (sameSelectedFigure) {
        updateMetadataFigureSidebar(figuresContainer, figures);
        figures.forEach(figure => tabularState.metadataSnapshots.set(
            figure.figure_key || figure.figure_id, structuredClone(figure)));
        return;
    }
    const preserveViewport = hadPreviousRender && !forcedRender;
    const pageViewport = preserveViewport ? {left: window.scrollX, top: window.scrollY} : null;
    const figureListScrollTop = preserveViewport
        ? figuresContainer.querySelector('.metadata-figure-list')?.scrollTop || 0 : 0;
    if (preserveViewport) {
        figuresContainer.querySelectorAll('.metadata-link-figure').forEach(element => {
            const viewer = element.querySelector('.metadata-link-pages');
            if (!viewer) return;
            const key = element.dataset.linkFigure;
            const stored = tabularState.metadataEvidenceState.get(key) || {};
            stored.scrollLeft = viewer.scrollLeft;
            stored.scrollTop = viewer.scrollTop;
            tabularState.metadataEvidenceState.set(key, stored);
        });
    }
    delete figuresContainer.dataset.forceRender;
    const tableScroll = new Map(
        [...figuresContainer.querySelectorAll('.metadata-link-figure')].map(element => {
            const wrap = element.querySelector('.metadata-link-table-wrap');
            return [element.dataset.linkFigure, {
                left: wrap?.scrollLeft || 0,
                top: wrap?.scrollTop || 0
            }];
        }));
    const sidebar = figures.map(figure => {
        const key = figure.figure_key || figure.figure_id;
        const label = metadataFigureStatusLabel(figure);
        return `<button type="button" class="metadata-figure-list-item ${key === tabularState.selectedFigureKey ? 'active' : ''}"
            data-select-figure="${linkEscape(key)}"><strong>Figure ${linkEscape(figure.figure_id || '?')}</strong>
            <span class="${linkEscape(label.toLowerCase().replace(/\s+/g, '-'))}">${linkEscape(label)}</span></button>`;
    }).join('');
    figuresContainer.innerHTML = `<aside class="metadata-figure-list" aria-label="Figures">${sidebar || '<p>No figures found.</p>'}</aside>
        <div class="metadata-active-figure">${selectedFigure ? renderMetadataFigure(selectedFigure) : ''}</div>`;
    figures.forEach(figure => tabularState.metadataSnapshots.set(
        figure.figure_key || figure.figure_id, structuredClone(figure)));
    figuresContainer.dataset.renderProject = renderProject;
    figuresContainer.dataset.activeFigureKey = selectedKey;
    tabularState.metadataRenderedFigureSignatures.set(selectedKey, selectedSignature);
    figuresContainer.querySelectorAll('[data-select-figure]').forEach(button =>
        button.addEventListener('click', () => {
            tabularState.selectedFigureKey = button.dataset.selectFigure;
            figuresContainer.dataset.forceRender = '1';
            button.blur();
            loadMetadataLinkState();
        }));
    figuresContainer.querySelectorAll('[data-link-save]').forEach(button =>
        button.addEventListener('click', () => runMetadataButtonAction(
            button, 'Saving…', 'Done', () => saveMetadataFigure(button.dataset.linkSave))
            .catch(error => window.PyPotteryUtils.showToast(error.message, 'error'))));
    figuresContainer.querySelectorAll('[data-link-approve]').forEach(button =>
        button.addEventListener('click', () => runMetadataButtonAction(
            button, 'Saving…', 'Done', () => approveMetadataFigure(button.dataset.linkApprove))
            .catch(() => null)));
    figuresContainer.querySelectorAll('[data-link-reject]').forEach(button =>
        button.addEventListener('click', () => runMetadataButtonAction(
            button, 'Saving…', 'Done', () => rejectMetadataFigure(button.dataset.linkReject))
            .catch(() => null)));
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
    wireMetadataWarningActions(figuresContainer);
    wireMetadataCellDebug(figuresContainer);
    figuresContainer.querySelectorAll('[data-link-rerun]').forEach(button =>
        button.addEventListener('click', () => rerunMetadataFigure(button.dataset.linkRerun)));
    figuresContainer.querySelectorAll('[data-link-measure]').forEach(button =>
        button.addEventListener('click', () => redetectMetadataMeasurements(button.dataset.linkMeasure)));
    figuresContainer.querySelectorAll('[data-link-inspect-measurement]').forEach(button =>
        button.addEventListener('click', () => openMetadataMeasurementEditor(
            button.closest('[data-link-figure]').dataset.linkFigure,
            button.dataset.linkInspectMeasurement)));
    figuresContainer.querySelectorAll('[data-link-inspect-scale]').forEach(button =>
        button.addEventListener('click', () => openMetadataMeasurementEditor(
            button.closest('[data-link-figure]').dataset.linkFigure,
            null, button.dataset.linkInspectScale)));
    figuresContainer.querySelectorAll('.metadata-link-figure').forEach(setupMetadataEvidenceViewer);
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
        '[data-link-drawing-number], [data-link-column]'
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
        input.addEventListener('input', () => selectMetadataDrawing(
            input.closest('[data-link-figure]').dataset.linkFigure, input));
    });
    figuresContainer.querySelectorAll('[data-link-diameter]').forEach(input => {
        input.addEventListener('input', () => synchronizeDiameterInputs(input));
        input.addEventListener('change', () => synchronizeDiameterInputs(input));
    });
    figuresContainer.querySelectorAll('[data-link-verify-diameter]').forEach(button => {
        button.addEventListener('click', () => verifyMetadataDiameter(button));
    });
    figuresContainer.querySelectorAll('[data-link-figure]').forEach(setupMetadataTableNavigation);
    figuresContainer.querySelectorAll('[data-link-figure]').forEach(element =>
        validateMetadataFigureDom(element.dataset.linkFigure));
    // Textarea auto-sizing and image restoration both change layout after the
    // markup is replaced. Restore every viewport after two animation frames so
    // the 1.5-second OCR poll cannot pull the researcher back up the page or to
    // the top-left of the publication/table panels.
    if (preserveViewport) {
        requestAnimationFrame(() => requestAnimationFrame(() => {
            const list = figuresContainer.querySelector('.metadata-figure-list');
            if (list) list.scrollTop = figureListScrollTop;
            figuresContainer.querySelectorAll('.metadata-link-figure').forEach(element => {
                const wrap = element.querySelector('.metadata-link-table-wrap');
                const saved = tableScroll.get(element.dataset.linkFigure);
                if (wrap && saved) {
                    wrap.scrollLeft = saved.left;
                    wrap.scrollTop = saved.top;
                }
            });
            if (pageViewport) window.scrollTo(pageViewport.left, pageViewport.top);
        }));
    }
}

function metadataFigureStatusLabel(figure) {
    const needsColumns = figure.needs_column_review || (figure.warnings || []).some(warning =>
        ['column_headers_incomplete', 'manual_column_bounds_invalid'].includes(warning.code));
    return figure.review_status === 'approved' ? 'Approved'
        : needsColumns ? 'Adjust columns'
        : figure.processing_status === 'processing' ? 'Processing'
            : figure.processing_status === 'queued' ? 'Waiting'
                : figure.status === 'ready' ? 'Ready' : 'Needs attention';
}

function updateMetadataFigureSidebar(container, figures) {
    const figuresByKey = new Map(figures.map(figure => [
        String(figure.figure_key || figure.figure_id || ''), figure
    ]));
    container.querySelectorAll('[data-select-figure]').forEach(button => {
        const key = String(button.dataset.selectFigure || '');
        const figure = figuresByKey.get(key);
        if (!figure) return;
        const label = metadataFigureStatusLabel(figure);
        button.classList.toggle('active', key === String(tabularState.selectedFigureKey || ''));
        const title = button.querySelector('strong');
        if (title) title.textContent = `Figure ${figure.figure_id || '?'}`;
        const status = button.querySelector('span');
        if (status) {
            status.className = label.toLowerCase().replace(/\s+/g, '-');
            status.textContent = label;
        }
    });
}

function metadataWarningsMarkup(figure, disabled, reviewKey) {
    const overrides = figure.warning_overrides || {};
    return (figure.warnings || []).map(warning => {
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
            <button type="button" data-warning-toggle="${warning.id}" data-figure-id="${linkEscape(reviewKey)}" ${disabled}>
                ${active ? 'Remove override' : 'Mark reviewed and ignore'}
            </button></div>` : '';
        const focusLabel = warning.code === 'missing_drawing_number'
            ? 'Edit drawing number' : 'Go to row';
        const focus = warning.row || warning.mask_file ? `<button type="button" class="metadata-warning-focus"
            data-warning-focus="${warning.id}" data-figure-id="${linkEscape(reviewKey)}">${focusLabel}</button>` : '';
        const addMissing = warning.code === 'missing_table_row' ? `<button type="button"
            class="metadata-warning-focus" data-warning-add-row="${linkEscape(warning.row || '')}"
            data-figure-id="${linkEscape(reviewKey)}">Add missing row</button>` : '';
        return `<article class="metadata-link-warning-card ${warning.blocking ? 'blocking' : 'resolved'}"
                         data-warning-id="${warning.id}" data-warning-code="${warning.code}"
                         data-warning-row="${linkEscape(warning.row || '')}"
                         data-warning-mask="${linkEscape(warning.mask_file || '')}"
                         data-override-active="${active ? '1' : '0'}">
            <div><strong>${active ? 'Reviewed' : warning.blocking ? 'Needs attention' : 'Information'}</strong>
            <span>${linkEscape(warning.message)}</span>${focus}${addMissing}</div>${controls}</article>`;
    }).join('');
}

function wireMetadataWarningActions(scope) {
    scope.querySelectorAll('[data-warning-toggle]').forEach(button =>
        button.addEventListener('click', () => toggleMetadataWarning(
            button.dataset.figureId, button.dataset.warningToggle)));
    scope.querySelectorAll('[data-warning-focus]').forEach(button =>
        button.addEventListener('click', () => focusMetadataWarning(
            button.dataset.figureId, button.dataset.warningFocus)));
    scope.querySelectorAll('[data-warning-add-row]').forEach(button =>
        button.addEventListener('click', () => addMetadataTableRow(
            button.dataset.figureId, button.dataset.warningAddRow)));
    scope.querySelectorAll('[data-warning-reason], [data-warning-note]').forEach(input => {
        input.addEventListener('input', () => {
            input.classList.add('metadata-link-edited');
            scheduleMetadataAutosave(input.closest('[data-link-figure]').dataset.linkFigure);
        });
        input.addEventListener('change', () =>
            scheduleMetadataAutosave(input.closest('[data-link-figure]').dataset.linkFigure));
    });
}

function updateMetadataWarningsFromFigure(figureId, figure) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    const section = figureEl?.querySelector('[data-link-warnings]');
    if (!section) return;
    const disabled = figure.processing_status === 'processing' || figure.processing_status === 'queued'
        ? 'disabled' : '';
    const markup = metadataWarningsMarkup(figure, disabled, figureId);
    section.innerHTML = markup ? `<h4>Review warnings</h4>${markup}` : '';
    section.hidden = !markup;
    if (markup) wireMetadataWarningActions(section);
}

function metadataCellDebugMarkup(figure, projectId, reviewKey) {
    if (window.SHERDSCOPE_OCR_CELL_DEBUG !== true) return '';
    const cells = (figure.table_pages || []).flatMap(page =>
        ((page.boundary || {}).cell_diagnostics || []).map(cell => ({
            ...cell, image_name: page.image_name
        })));
    const diagnosticPages = (figure.table_pages || []).filter(page => page.diagnostics_ref);
    const statusPages = (figure.table_pages || []).filter(page =>
        (page.boundary || {}).diagnostic_status);
    const statusMarkup = statusPages.map(page => {
        const status = page.boundary.diagnostic_status || {};
        const missing = (status.missing_labels || []).length
            ? `<br><strong>Missing:</strong> ${linkEscape(status.missing_labels.join(', '))}` : '';
        return `<p><strong>${linkEscape(page.image_name || 'Table page')}:</strong>
            ${linkEscape(status.message || '')}${missing}<br>${linkEscape(status.action || '')}</p>`;
    }).join('');
    if (!cells.length && diagnosticPages.length) return `<details class="metadata-cell-debug">
        <summary>Development: inspect OCR cells</summary>
        <p>Detailed OCR evidence is stored separately so normal progress polling stays fast.</p>
        ${statusMarkup}
        ${diagnosticPages.map(page => `<button type="button" data-load-cell-diagnostics="${linkEscape(page.image_name)}"
            data-figure-id="${linkEscape(reviewKey)}">Load ${linkEscape(page.image_name)}</button>`).join('')}
    </details>`;
    if (!cells.length) return `<details class="metadata-cell-debug">
        <summary>Development: inspect every OCR cell</summary>
        ${statusMarkup}
        <p>No saved cell grid exists for this figure yet. Use More → Re-read this figure to generate it with the development extractor.</p>
    </details>`;
    const tokenText = tokens => (tokens || []).map(token =>
        `${token.text || '(blank)'} (${Math.round(Number(token.confidence || 0) * 100)}%)`
    ).join(', ');
    const buttons = cells.map((cell, index) => {
        const baseUrl = `/api/projects/${encodeURIComponent(projectId)}/metadata-link/evidence/` +
            `${encodeURIComponent(cell.image_name)}?figure=${encodeURIComponent(reviewKey)}` +
            `&kind=table&cell_row=${encodeURIComponent(cell.row || '')}` +
            `&cell_field=${encodeURIComponent(cell.field || '')}` +
            `&v=${encodeURIComponent(tabularState.metadataLinkState?.updated_at || '')}`;
        const focusedUrl = cell.focused_crop ? `${baseUrl}&cell_view=focused` : '';
        const focusedText = cell.focused_text ||
            ((cell.focused_tokens || []).length ? '(below acceptance threshold)' : '(blank)');
        return `<button type="button" class="metadata-cell-debug-button"
            data-cell-debug-open="${index}" data-cell-title="Row ${linkEscape(cell.row || '?')} · ${linkEscape(cell.field || '?')}"
            data-cell-raw-url="${linkEscape(baseUrl)}" data-cell-focused-url="${linkEscape(focusedUrl)}"
            data-cell-initial="${linkEscape(cell.initial_text || '(blank)')}"
            data-cell-focused="${linkEscape(focusedText)}"
            data-cell-accepted="${linkEscape(cell.accepted_text || '(blank)')}"
            data-cell-source="${linkEscape(cell.accepted_source || 'table_pass')}"
            data-cell-geometry="${cell.page_geometry_reliable === false ? 'Crossed a column boundary' : 'Inside this cell'}"
            data-cell-reason="${linkEscape(cell.decision_reason || '')}"
            data-cell-tokens="${linkEscape(tokenText(cell.initial_tokens) || 'No assigned page token')}"
            data-cell-focused-tokens="${linkEscape(tokenText(cell.focused_tokens) || 'No focused token')}">
            <strong>${linkEscape(cell.row || '?')}</strong><span>${linkEscape(cell.field || '?')}</span>
            <small>${linkEscape(cell.accepted_text || '(blank)')}</small></button>`;
    }).join('');
    const conflicts = (figure.table_pages || []).flatMap(page =>
        (page.boundary?.row_anchor_conflicts || []).map(conflict => ({
            ...conflict, image_name: page.image_name
        })));
    const conflictMarkup = conflicts.length ? `<p><strong>${conflicts.length} row-number conflict${conflicts.length === 1 ? '' : 's'} resolved:</strong> ${conflicts.map(conflict =>
        `${linkEscape((conflict.candidates || []).map(item => item.number).join(' / '))} → ${linkEscape(conflict.chosen || '?')}`).join('; ')}</p>` : '';
    return `<details class="metadata-cell-debug">
        <summary>Development: inspect every OCR cell (${cells.length})</summary>
        <p>Orange horizontal lines and the existing column lines show the exact calculated cell grid. Select a cell to compare the whole-table page pass with the separate cell OCR result used as the final value.</p>
        ${conflictMarkup}
        <div class="metadata-cell-debug-grid">${buttons}</div>
        <dialog class="metadata-cell-debug-dialog">
            <form method="dialog"><button aria-label="Close cell details">Close</button></form>
            <h4 data-cell-debug-title></h4>
            <div class="metadata-cell-debug-images">
                <figure><figcaption>Raw calculated cell</figcaption><img data-cell-debug-raw alt="Raw OCR cell crop"></figure>
                <figure data-cell-debug-focused-wrap><figcaption>Cell OCR input</figcaption><img data-cell-debug-focused-image alt="Prepared cell OCR crop"></figure>
            </div>
            <dl>
                <dt>Page-pass text</dt><dd data-cell-debug-initial></dd>
                <dt>Page-pass tokens</dt><dd data-cell-debug-tokens></dd>
                <dt>Cell-pass text</dt><dd data-cell-debug-focused-text></dd>
                <dt>Cell-pass tokens</dt><dd data-cell-debug-focused-tokens></dd>
                <dt>Accepted value</dt><dd data-cell-debug-accepted></dd>
                <dt>Page geometry</dt><dd data-cell-debug-geometry></dd>
                <dt>Decision</dt><dd data-cell-debug-reason></dd>
            </dl>
        </dialog>
    </details>`;
}

function wireMetadataCellDebug(scope) {
    scope.querySelectorAll('[data-load-cell-diagnostics]').forEach(button =>
        button.addEventListener('click', async () => {
            try {
                button.disabled = true; button.textContent = 'Loading…';
                const response = await window.PyPotteryUtils.apiRequest(
                    `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(button.dataset.figureId)}/diagnostics/${encodeURIComponent(button.dataset.loadCellDiagnostics)}`);
                if (!response.success) throw new Error(response.error || 'Could not load diagnostics');
                const figure = tabularState.metadataLinkState?.figures.find(item =>
                    (item.figure_key || item.figure_id) === button.dataset.figureId);
                const page = (figure?.table_pages || []).find(item =>
                    item.image_name === button.dataset.loadCellDiagnostics);
                if (page) {
                    page.boundary.cell_diagnostics = response.diagnostics?.cell_diagnostics || [];
                    page.boundary.diagnostic_status = response.diagnostics?.status || page.boundary.diagnostic_status;
                    page.boundary.row_anchor_conflicts = response.diagnostics?.row_anchor_conflicts || [];
                }
                document.getElementById('metadata-link-figures').dataset.forceRender = '1';
                renderMetadataLinkState(tabularState.metadataLinkState, true);
            } catch (error) {
                button.disabled = false; button.textContent = 'Retry diagnostics';
                window.PyPotteryUtils.showToast(error.message, 'error');
            }
        }));
    scope.querySelectorAll('[data-cell-debug-open]').forEach(button => {
        button.addEventListener('click', () => {
            const panel = button.closest('.metadata-cell-debug');
            const dialog = panel?.querySelector('.metadata-cell-debug-dialog');
            if (!dialog) return;
            dialog.querySelector('[data-cell-debug-title]').textContent = button.dataset.cellTitle || '';
            dialog.querySelector('[data-cell-debug-raw]').src = button.dataset.cellRawUrl || '';
            const focusedUrl = button.dataset.cellFocusedUrl || '';
            const focusedWrap = dialog.querySelector('[data-cell-debug-focused-wrap]');
            focusedWrap.hidden = !focusedUrl;
            dialog.querySelector('[data-cell-debug-focused-image]').src = focusedUrl;
            dialog.querySelector('[data-cell-debug-initial]').textContent = button.dataset.cellInitial || '';
            dialog.querySelector('[data-cell-debug-tokens]').textContent = button.dataset.cellTokens || '';
            dialog.querySelector('[data-cell-debug-focused-text]').textContent = button.dataset.cellFocused || '';
            dialog.querySelector('[data-cell-debug-focused-tokens]').textContent = button.dataset.cellFocusedTokens || '';
            dialog.querySelector('[data-cell-debug-accepted]').textContent =
                `${button.dataset.cellAccepted || ''} (${button.dataset.cellSource || ''})`;
            dialog.querySelector('[data-cell-debug-geometry]').textContent =
                button.dataset.cellGeometry || '';
            dialog.querySelector('[data-cell-debug-reason]').textContent =
                button.dataset.cellReason || '';
            dialog.showModal();
        });
    });
}

function metadataMeasurementValue(measurement) {
    return ['verified', 'verified_automatic', 'verified_manual'].includes(measurement?.status)
        ? measurement.verified_cm : measurement?.suggested_cm;
}

function metadataMeasurementLabel(status) {
    if (status === 'verified_automatic') return 'Automatic';
    if (status === 'verified_manual' || status === 'verified') return 'Manually corrected';
    return 'Needs attention';
}

function metadataMeasurementExplanation(measurement) {
    const reasons = {
        missing_scale_calibration: 'The page scale could not be measured reliably.',
        rim_span_not_found: 'The top rim line could not be found reliably.',
        centreline_not_found: 'The central vertical construction line could not be found.',
        diameter_estimators_minor_disagreement: 'The two diameter estimates differ by 5–15%. The centreline estimate remains accepted.',
        diameter_estimators_disagree: 'The full rim span and centreline-based estimate differ by more than 15%.',
        rim_endpoints_exceed_drawing_bbox: 'The measured rim extends more than 15% outside the drawing box.',
        invalid_drawing_bbox: 'The drawing crop is invalid or has changed.',
        image_not_found: 'The original publication page is unavailable.'
    };
    return reasons[measurement?.warning] || (measurement?.status === 'unresolved'
        ? 'The automatic measurement did not pass its geometry checks.'
        : 'The automatic scale and two diameter estimates passed their geometry checks.');
}

function renderMetadataFigure(figure) {
    const projectId = tabularState.currentProject.project_id;
    const reviewKey = figure.figure_key || figure.figure_id;
    const processing = figure.processing_status === 'processing';
    const waiting = figure.processing_status === 'queued';
    const unavailable = processing || waiting;
    const status = metadataFigureStatusLabel(figure);
    const disabled = unavailable ? 'disabled' : '';
    const jobs = tabularState.metadataLinkState?.jobs || {};
    const figureJobs = [jobs.active, jobs.paused, ...(jobs.queued || [])].filter(job => {
        const target = String(job?.payload?.figure_key || '');
        return target && [String(reviewKey), String(figure.figure_id || '')].includes(target);
    });
    const measurementJob = figureJobs.find(job => job.kind === 'measurement_reread');
    const rereadJob = figureJobs.find(job => ['figure_reread', 'boundary_reread'].includes(job.kind));
    const jobButtonLabel = (job, fallback) => job
        ? job.status === 'running' ? 'Running…' : job.status === 'paused' ? 'Paused' : 'Queued'
        : fallback;
    const evidencePages = [
        ...(figure.drawing_pages || []).map(page => ({...page, kind: 'drawing'})),
        ...(figure.table_pages || []).map(page => ({...page, kind: 'table'}))
    ];
    const pageHtml = evidencePages.map((page, pageIndex) => {
        const evidenceUrl = `/api/projects/${encodeURIComponent(projectId)}/metadata-link/evidence/${encodeURIComponent(page.image_name)}` +
            `?figure=${encodeURIComponent(reviewKey)}&kind=${page.kind}&overlay=1&measurement=${page.kind === 'drawing' ? '1' : '0'}&v=${encodeURIComponent(tabularState.metadataLinkState?.updated_at || '')}`;
        return `<a href="${evidenceUrl}" data-evidence-page="${pageIndex}" ${pageIndex ? 'hidden' : ''}>
            <img src="${evidenceUrl}" data-evidence-kind="${page.kind}" alt="${linkEscape(page.image_name)}"
                 title="${linkEscape(page.kind)}: ${linkEscape(page.image_name)}">
        </a>`;
    }).join('');
    const drawingByNumber = new Map((figure.drawings || []).map(drawing =>
        [String(drawing.vessel_number || '').trim().toLowerCase(), drawing]));
    const drawings = (figure.drawings || []).map(drawing => {
        const measurement = drawing.measurement || {};
        const diameter = metadataMeasurementValue(measurement);
        const measurementHelp = metadataMeasurementExplanation(measurement);
        return `<div class="metadata-link-drawing" data-mask-file="${linkEscape(drawing.mask_file)}">
        <label>
            <span title="${linkEscape(drawing.mask_file)}">Printed No.</span>
            <input class="form-control" data-link-drawing-number data-mask-file="${linkEscape(drawing.mask_file)}"
                   value="${linkEscape(drawing.vessel_number || '')}" aria-label="Printed vessel number" ${disabled}>
        </label><label><span>Rim diameter (cm)</span>
            <input class="form-control" type="number" min="0.1" step="0.1" data-link-diameter
                   data-mask-file="${linkEscape(drawing.mask_file)}" data-measurement-status="${linkEscape(measurement.status || 'unresolved')}"
                   data-measurement-exact="${diameter == null ? '' : linkEscape(diameter)}" data-measurement-edited="0"
                   value="${diameter == null ? '' : Number(diameter).toFixed(1)}" ${disabled}>
        </label><div class="metadata-diameter-actions">
            <span class="metadata-measure-status ${linkEscape(measurement.status || 'unresolved')}" title="${linkEscape(measurementHelp)}">${linkEscape(metadataMeasurementLabel(measurement.status))}</span>
            ${measurement.status === 'unresolved' ? `<small>${linkEscape(measurementHelp)}</small>` : ''}
            <button type="button" data-link-inspect-measurement="${linkEscape(drawing.mask_file)}" ${disabled}>Correct</button>
        </div></div>`;
    }).join('');
    const rows = (figure.table_rows || []).map((row, rowIndex) => {
        const drawing = drawingByNumber.get(String(row.table_no || '').trim().toLowerCase());
        const measurement = drawing?.measurement || {};
        const diameter = metadataMeasurementValue(measurement);
        const diameterCell = `<td class="metadata-diameter-cell"><input class="form-control" type="number" min="0.1" step="0.1"
            data-link-diameter data-mask-file="${linkEscape(drawing?.mask_file || '')}"
            data-measurement-status="${linkEscape(measurement.status || 'unresolved')}"
            data-measurement-exact="${diameter == null ? '' : linkEscape(diameter)}" data-measurement-edited="0"
            value="${diameter == null ? '' : Number(diameter).toFixed(1)}" ${!drawing || processing ? 'disabled' : ''}>
            <small>${drawing ? linkEscape(metadataMeasurementLabel(measurement.status)) : 'No matching drawing'}</small></td>`;
        const dataCells = HESBAN_LINK_COLUMNS.map((column, columnIndex) =>
            `<td class="${columnIndex === 0 ? 'sticky-col sticky-no' : columnIndex === 1 ? 'sticky-col sticky-type' : ''}"><textarea data-link-column="${column}" ${disabled}>${linkEscape(row[column] || '')}</textarea></td>`);
        dataCells.splice(2, 0, diameterCell);
        return `
        <tr data-link-row="${rowIndex}" ${metadataReviewRowAttribute(row, rowIndex)}>
            <td class="sticky-col sticky-actions metadata-link-row-actions">
                <button type="button" aria-label="Duplicate row ${rowIndex + 1}" title="Duplicate row"
                        data-link-duplicate-row="${linkEscape(reviewKey)}" data-row-index="${rowIndex}" ${disabled}>⧉</button>
                <button type="button" aria-label="Delete row ${rowIndex + 1}" title="Delete row"
                        data-link-delete-row="${linkEscape(reviewKey)}" data-row-index="${rowIndex}" ${disabled}>🗑</button>
            </td>${dataCells.join('')}</tr>`;
    }).join('');
    const cellDebugPanel = metadataCellDebugMarkup(figure, projectId, reviewKey);
    const warnings = metadataWarningsMarkup(figure, disabled, reviewKey);
    const tablePageNames = (figure.table_pages || []).map(page => page.image_name).join(', ');
    const open = figure.status === 'needs_review' ? 'open' : '';
    const blockers = (figure.warnings || []).filter(warning => warning.blocking).length;
    const unmatched = (figure.matches || []).filter(match => match.status !== 'ready').length;
    const calibrations = Object.entries(figure.scale_calibrations || {});
    const scaleSummary = calibrations.length ? calibrations.map(([imageName, calibration]) =>
        `<article class="metadata-scale-card"><strong>${linkEscape(imageName)}</strong>
         <span>${['suggested', 'verified', 'verified_automatic', 'verified_manual'].includes(calibration.status) && calibration.px_per_cm ? Number(calibration.px_per_cm).toFixed(2) + ' px/cm' : 'Scale unresolved'}</span>
         <span class="metadata-measure-status ${linkEscape(calibration.status || 'unresolved')}">${linkEscape(metadataMeasurementLabel(calibration.status))}</span>
         ${calibration.warning ? `<small>${linkEscape(calibration.warning)}</small>` : ''}
         <button type="button" data-link-inspect-scale="${linkEscape(calibration.evidence_image || imageName)}" ${disabled}>Inspect / correct scale</button></article>`).join('')
        : '<p>No scale has been detected yet.</p>';
    return `<details class="metadata-link-figure" data-link-figure="${linkEscape(reviewKey)}"
                    data-reviewer-revision="${Number(figure.reviewer_revision || 0)}" open>
        <summary><strong>Figure ${linkEscape(figure.figure_id)}</strong>
            <span class="metadata-link-badge ${linkEscape(status.toLowerCase().replace(/\s+/g, '-'))}">${linkEscape(status)}</span>
            <span>${(figure.drawings || []).length} drawings / ${(figure.table_rows || []).length} rows</span>
        </summary>
        <div class="metadata-link-figure-body">
            <div class="metadata-link-save-strip">
                <strong>Review workspace</strong>
                <span data-link-save-status aria-live="polite">${processing ? 'OCR processing…' : waiting ? 'Waiting for OCR…' : 'Saved'}</span>
            </div>
            <label>Figure ID <input class="form-control" data-link-figure-id value="${linkEscape(figure.figure_id || '')}" ${disabled}></label>
            <label>Caption <input class="form-control" data-link-caption value="${linkEscape(figure.figure_caption || '')}" ${disabled}></label>
            <label>Table page image names (comma-separated)
                <input class="form-control" data-link-table-pages value="${linkEscape(tablePageNames)}" ${disabled}>
            </label>
            <div class="metadata-link-evidence">
                <div><div class="metadata-evidence-toolbar"><h4>Publication page</h4>
                    <span><button type="button" data-evidence-prev aria-label="Previous page">Previous</button>
                    <button type="button" data-evidence-next aria-label="Next page">Next</button>
                    <output data-evidence-page-status aria-live="polite"></output>
                    <button type="button" data-evidence-zoom-out aria-label="Zoom out">Zoom out</button>
                    <button type="button" data-evidence-zoom-in aria-label="Zoom in">Zoom in</button>
                    <button type="button" data-evidence-columns>Adjust columns</button>
                    <button type="button" data-evidence-boxes>Hide boxes</button>
                    <button type="button" data-evidence-reset>Reset</button></span></div>
                    <div class="metadata-link-pages" data-evidence-index="0" data-evidence-zoom="1">${pageHtml}</div></div>
            </div>
            <section class="metadata-scale-workspace"><div class="metadata-scale-heading">
                <div><h4>Scale and rim diameters</h4><p>Valid automatic measurements are already accepted. Correct only the values that look wrong.</p></div>
                <button type="button" data-link-measure="${linkEscape(reviewKey)}"
                    ${unavailable || measurementJob ? 'disabled' : ''} aria-busy="${measurementJob?.status === 'running'}">${jobButtonLabel(measurementJob, 'Re-read measurements')}</button>
            </div><div class="metadata-scale-cards">${scaleSummary}</div></section>
            <div class="metadata-link-number-workspace"><h4>Drawing numbers</h4>
                <p>Correct a printed number or diameter only when it does not match the publication.</p>
                <div class="metadata-link-drawings">${drawings}</div>
            </div>
            <div class="metadata-link-table-section"><div class="metadata-link-table-toolbar">
                <h4>Extracted table</h4>
                <button type="button" data-link-add-row="${linkEscape(reviewKey)}" ${disabled}>Add row</button>
                <button type="button" data-link-undo-row="${linkEscape(reviewKey)}" ${disabled}>Undo delete</button>
                <button type="button" data-link-restore="${linkEscape(reviewKey)}" ${disabled}>Restore last saved</button>
                <button type="button" data-link-table-expand>Full-screen table</button>
                <nav class="metadata-column-jumps" aria-label="Jump to column group">
                    <button type="button" data-column-jump="identity">Identity</button><button type="button" data-column-jump="fabric">Fabric</button>
                    <button type="button" data-column-jump="nonplastics">Non-Plastics</button><button type="button" data-column-jump="voids">Voids</button>
                    <button type="button" data-column-jump="surface">Surface</button><button type="button" data-column-jump="finish">Finish</button>
                </nav>
            </div><div class="metadata-link-table-wrap" role="region" tabindex="0" aria-label="Editable publication table">
                <table class="metadata-link-table"><thead>${hesbanGroupedHeaders()}</thead><tbody>${rows}</tbody></table>
            </div></div>
            ${cellDebugPanel}
            <section class="metadata-link-warnings" data-link-warnings ${warnings ? '' : 'hidden'}>
                ${warnings ? `<h4>Review warnings</h4>${warnings}` : ''}
            </section>
            <div class="metadata-link-readiness ${blockers || unmatched ? 'blocked' : 'ready'}">
                <strong>CSV readiness</strong>
                <span>${processing ? 'This figure is still processing.' : waiting ? 'This figure is waiting for OCR.' : blockers
                    ? `${blockers} blocking warning(s) remain.` : unmatched
                        ? `${unmatched} drawing/table match(es) remain unresolved.`
                        : 'Unique matches are ready for approval.'}</span>
            </div>
            <div class="metadata-link-review-actions">
                <button class="btn btn-secondary" data-link-save="${linkEscape(reviewKey)}" ${disabled}>Save now</button>
                <button class="btn btn-success" data-link-approve="${linkEscape(reviewKey)}"
                        ${figure.status !== 'ready' || unavailable ? 'disabled' : ''}>Approve and apply to CSV</button>
                <details class="metadata-more-actions"><summary>More</summary><div>
                    <button class="btn btn-secondary" data-link-rerun="${linkEscape(reviewKey)}"
                        ${unavailable || rereadJob ? 'disabled' : ''} aria-busy="${rereadJob?.status === 'running'}">${jobButtonLabel(rereadJob, 'Re-read this figure')}</button>
                    <button class="btn btn-danger" data-link-reject="${linkEscape(reviewKey)}" ${disabled}>Reject figure</button>
                </div></details>
            </div>
        </div></details>`;
}

function setupMetadataEvidenceViewer(figureElement) {
    const viewer = figureElement.querySelector('.metadata-link-pages');
    if (!viewer) return;
    const pages = [...viewer.querySelectorAll('[data-evidence-page]')];
    const figureKey = figureElement.dataset.linkFigure;
    const stored = tabularState.metadataEvidenceState.get(figureKey) || {
        index: 0, zoom: 1, overlays: true, scrollLeft: 0, scrollTop: 0
    };
    const status = figureElement.querySelector('[data-evidence-page-status]');
    const overlayButton = figureElement.querySelector('[data-evidence-boxes]');
    const previousButton = figureElement.querySelector('[data-evidence-prev]');
    const nextButton = figureElement.querySelector('[data-evidence-next]');
    const columnsButton = figureElement.querySelector('[data-evidence-columns]');
    const applyOverlays = visible => {
        pages.forEach(page => {
            const image = page.querySelector('img');
            if (!image) return;
            const url = new URL(image.src, window.location.origin);
            url.searchParams.set('overlay', visible ? '1' : '0');
            image.src = url.pathname + url.search;
            page.href = image.src;
        });
        if (overlayButton) {
            overlayButton.dataset.hidden = visible ? '0' : '1';
            overlayButton.textContent = visible ? 'Hide boxes' : 'Show boxes';
        }
    };
    const showPage = delta => {
        let index = Number(viewer.dataset.evidenceIndex || 0) + delta;
        index = Math.max(0, Math.min(pages.length - 1, index));
        viewer.dataset.evidenceIndex = String(index);
        pages.forEach((page, pageIndex) => { page.hidden = pageIndex !== index; });
        stored.index = index;
        tabularState.metadataEvidenceState.set(figureKey, stored);
        if (status) status.textContent = pages.length ? `${index + 1} / ${pages.length}` : 'No pages';
        if (previousButton) previousButton.disabled = !pages.length || index === 0;
        if (nextButton) nextButton.disabled = !pages.length || index === pages.length - 1;
        if (columnsButton) {
            const image = pages[index]?.querySelector('img');
            columnsButton.hidden = image?.dataset.evidenceKind !== 'table';
            columnsButton.dataset.imageName = image?.alt || '';
        }
    };
    const setZoom = value => {
        const zoom = Math.max(.5, Math.min(2.5, value));
        viewer.dataset.evidenceZoom = String(zoom);
        viewer.style.setProperty('--evidence-zoom', zoom);
        stored.zoom = zoom;
        tabularState.metadataEvidenceState.set(figureKey, stored);
    };
    viewer.dataset.evidenceIndex = String(Math.max(0, Math.min(pages.length - 1, stored.index || 0)));
    setZoom(Number(stored.zoom || 1));
    showPage(0);
    applyOverlays(stored.overlays !== false);
    requestAnimationFrame(() => {
        viewer.scrollLeft = Number(stored.scrollLeft || 0);
        viewer.scrollTop = Number(stored.scrollTop || 0);
    });
    viewer.addEventListener('scroll', () => {
        stored.scrollLeft = viewer.scrollLeft;
        stored.scrollTop = viewer.scrollTop;
        tabularState.metadataEvidenceState.set(figureKey, stored);
    }, {passive: true});
    pages.forEach(page => page.addEventListener('click', event => {
        event.preventDefault();
        const image = page.querySelector('img');
        if (image && typeof showImageModal === 'function') showImageModal(image.src);
    }));
    previousButton?.addEventListener('click', () => showPage(-1));
    nextButton?.addEventListener('click', () => showPage(1));
    figureElement.querySelector('[data-evidence-zoom-out]')?.addEventListener('click', () =>
        setZoom(Number(viewer.dataset.evidenceZoom || 1) - .15));
    figureElement.querySelector('[data-evidence-zoom-in]')?.addEventListener('click', () =>
        setZoom(Number(viewer.dataset.evidenceZoom || 1) + .15));
    figureElement.querySelector('[data-evidence-reset]')?.addEventListener('click', () => setZoom(1));
    columnsButton?.addEventListener('click', () => {
        if (columnsButton.dataset.imageName) openMetadataColumnEditor(
            figureKey, columnsButton.dataset.imageName);
    });
    overlayButton?.addEventListener('click', event => {
        const currentlyVisible = event.currentTarget.dataset.hidden !== '1';
        stored.overlays = !currentlyVisible;
        tabularState.metadataEvidenceState.set(figureKey, stored);
        applyOverlays(stored.overlays);
    });
}

async function openMetadataColumnEditor(figureId, imageName) {
    const figure = tabularState.metadataLinkState?.figures.find(item =>
        (item.figure_key || item.figure_id) === figureId);
    const page = (figure?.table_pages || []).find(item => item.image_name === imageName);
    if (!figure || !page) return;
    const boundary = page.boundary || {};
    const imageSize = boundary.image_size || [1, 1];
    const imageWidth = Number(imageSize[0] || 1);
    const imageHeight = Number(imageSize[1] || 1);
    const table = boundary.table_bounds || [0, 0, imageWidth, imageHeight];
    let edges = page.manual_column_edges || boundary.normalized_column_edges;
    if (!Array.isArray(edges) || edges.length !== HESBAN_LINK_COLUMNS.length + 1) {
        const left = Number(table[0] || 0) / imageWidth;
        const right = Number(table[2] || imageWidth) / imageWidth;
        const suggestions = Array(HESBAN_LINK_COLUMNS.length + 1).fill(null);
        const anchors = Array.isArray(boundary.header_anchors) ? boundary.header_anchors : [];
        const heights = anchors.map(anchor => Number(anchor.bbox?.[3]) - Number(anchor.bbox?.[1]))
            .filter(height => Number.isFinite(height) && height > 0).sort((a, b) => a - b);
        const medianHeight = heights.length ? heights[Math.floor(heights.length / 2)] : 20;
        const lead = Math.max(3, Math.min(18, medianHeight * .2));
        anchors.forEach(anchor => {
            const index = HESBAN_LINK_COLUMNS.indexOf(anchor.column);
            const anchorLeft = Number(anchor.bbox?.[0]);
            if (index >= 0 && Number.isFinite(anchorLeft)) {
                suggestions[index] = Math.max(0, anchorLeft - lead) / imageWidth;
            }
        });
        suggestions[22] = right;
        if (suggestions.some(value => value != null)) {
            if (suggestions[0] == null) suggestions[0] = left;
            const known = suggestions.map((value, index) => value == null ? null : index)
                .filter(index => index != null);
            for (let part = 0; part < known.length - 1; part += 1) {
                const start = known[part]; const end = known[part + 1];
                for (let index = start + 1; index < end; index += 1) {
                    suggestions[index] = suggestions[start] +
                        (suggestions[end] - suggestions[start]) * (index - start) / (end - start);
                }
            }
            edges = suggestions;
        } else {
            edges = Array.from({length: HESBAN_LINK_COLUMNS.length + 1}, (_, index) =>
                left + (right - left) * index / HESBAN_LINK_COLUMNS.length);
        }
    }
    edges = edges.map(Number);
    const dialog = document.createElement('dialog');
    dialog.className = 'metadata-column-dialog';
    dialog.innerHTML = `<form method="dialog"><header><div><strong>Adjust table columns</strong>
        <span>${linkEscape(imageName)}</span></div><button aria-label="Cancel column changes">Cancel</button></header>
        <p>Drag a line, or select it and use the arrow keys. Lines cannot cross.</p>
        <div class="metadata-column-canvas-wrap"><canvas tabindex="0" aria-label="Draggable table column lines"></canvas></div>
        <footer><output data-column-output aria-live="polite"></output><div>
            <button type="button" data-column-reset>Reset to detected</button>
            <button type="button" data-column-save>Save and re-read</button>
        </div></footer></form>`;
    document.body.appendChild(dialog);
    const canvas = dialog.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const output = dialog.querySelector('[data-column-output]');
    const image = new Image();
    image.src = `/api/projects/${tabularState.currentProject.project_id}/metadata-link/evidence/${encodeURIComponent(imageName)}` +
        `?figure=${encodeURIComponent(figureId)}&kind=table&overlay=0&full=1&v=${Date.now()}`;
    let selected = 0;
    let dragging = false;
    const padding = Math.max(18, imageWidth * .01);
    const topPadding = Math.max(80, (Number(table[3]) - Number(table[1])) * .08);
    const view = [Math.max(0, Number(table[0]) - padding),
        Math.max(0, Number(table[1]) - topPadding),
        Math.min(imageWidth, Number(table[2]) + padding),
        Math.min(imageHeight, Number(table[3]) + padding)];
    const resize = () => {
        const width = Math.min(1500, Math.max(720, window.innerWidth * .88));
        const ratio = (view[3] - view[1]) / Math.max(1, view[2] - view[0]);
        canvas.width = Math.round(width);
        canvas.height = Math.round(Math.min(window.innerHeight * .7, width * ratio));
    };
    const imageX = normalized => normalized * imageWidth;
    const canvasX = normalized => (imageX(normalized) - view[0]) /
        Math.max(1, view[2] - view[0]) * canvas.width;
    const pointerImageX = event => {
        const rect = canvas.getBoundingClientRect();
        return view[0] + (event.clientX - rect.left) / rect.width * (view[2] - view[0]);
    };
    const redraw = () => {
        if (!image.complete) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, view[0], view[1], view[2] - view[0], view[3] - view[1],
            0, 0, canvas.width, canvas.height);
        edges.forEach((edge, index) => {
            const x = canvasX(edge);
            ctx.strokeStyle = index === selected ? '#f97316' : '#2563eb';
            ctx.lineWidth = index === selected ? 4 : 2;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
            if (index < HESBAN_LINK_FIELDS.length) {
                ctx.save(); ctx.translate(x + 4, 14); ctx.rotate(-Math.PI / 4);
                ctx.fillStyle = index === selected ? '#9a3412' : '#172033';
                ctx.font = 'bold 12px sans-serif';
                ctx.fillText(HESBAN_LINK_FIELDS[index][1], 0, 0); ctx.restore();
            }
        });
        output.textContent = `Selected line ${selected + 1} of ${edges.length}`;
    };
    const moveSelected = pageX => {
        const minimum = .0015;
        const normalized = pageX / imageWidth;
        const low = selected ? edges[selected - 1] + minimum : 0;
        const high = selected < edges.length - 1 ? edges[selected + 1] - minimum : 1;
        edges[selected] = Math.max(low, Math.min(high, normalized));
        redraw();
    };
    canvas.addEventListener('pointerdown', event => {
        const pageX = pointerImageX(event);
        selected = edges.map(edge => Math.abs(imageX(edge) - pageX))
            .reduce((best, distance, index, values) => distance < values[best] ? index : best, 0);
        dragging = true; canvas.setPointerCapture(event.pointerId); moveSelected(pageX);
    });
    canvas.addEventListener('pointermove', event => {
        if (dragging) moveSelected(pointerImageX(event));
    });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('pointercancel', () => { dragging = false; });
    canvas.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const amount = event.shiftKey ? 5 : 1;
        moveSelected(imageX(edges[selected]) + (event.key === 'ArrowLeft' ? -amount : amount));
    });
    image.addEventListener('load', () => { resize(); redraw(); });
    dialog.querySelector('[data-column-save]').addEventListener('click', async event => {
        const button = event.currentTarget;
        try {
            button.disabled = true; button.textContent = 'Saving…'; button.setAttribute('aria-busy', 'true');
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(figureId)}/pages/${encodeURIComponent(imageName)}/columns`, {
                    method: 'PUT', body: JSON.stringify({
                        reviewer_revision: Number(figure.reviewer_revision || 0),
                        normalized_column_edges: edges
                    })
                });
            if (!response.success) throw new Error(response.error || 'Could not save columns');
            dialog.close();
            window.PyPotteryUtils.showToast('Column reprocessing queued', 'success');
            await loadMetadataLinkState();
        } catch (error) {
            button.disabled = false; button.textContent = 'Save and re-read';
            button.removeAttribute('aria-busy');
            window.PyPotteryUtils.showToast(error.message, 'error');
        }
    });
    dialog.querySelector('[data-column-reset]').addEventListener('click', async event => {
        const button = event.currentTarget;
        try {
            button.disabled = true; button.textContent = 'Resetting…';
            button.setAttribute('aria-busy', 'true');
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(figureId)}/pages/${encodeURIComponent(imageName)}/columns`, {
                    method: 'DELETE', body: JSON.stringify({reviewer_revision: Number(figure.reviewer_revision || 0)})
                });
            if (!response.success) throw new Error(response.error || 'Could not reset columns');
            dialog.close(); await loadMetadataLinkState();
        } catch (error) {
            button.disabled = false; button.textContent = 'Reset to detected';
            button.removeAttribute('aria-busy');
            window.PyPotteryUtils.showToast(error.message, 'error');
        }
    });
    dialog.addEventListener('close', () => dialog.remove());
    dialog.showModal();
}

function addMetadataTableRow(figureId, tableNumber = '') {
    const figureEl = [...document.querySelectorAll('[data-link-figure]')]
        .find(element => element.dataset.linkFigure === figureId);
    const body = figureEl?.querySelector('.metadata-link-table tbody');
    if (!body) return;
    const row = document.createElement('tr');
    row.innerHTML = metadataDynamicRowCells(
        figureId, body.children.length, {table_no: tableNumber});
    body.appendChild(row);
    wireMetadataDynamicRow(row, figureId);
    scheduleMetadataAutosave(figureId);
    row.querySelector('[data-link-column="table_no"]')?.focus();
}

function metadataReviewRowKey(row, index = 0) {
    if (row?.review_override_id) return `manual:${row.review_override_id}`;
    const number = String(row?.normalized_table_no || row?.table_no || `row-${index}`).trim().toLowerCase();
    return `${row?.source_image || ''}|${number}`;
}

function metadataReviewRowAttribute(row, index = 0) {
    return `data-review-key="${linkEscape(metadataReviewRowKey(row, index))}"`;
}

function metadataDynamicRowCells(figureId, rowIndex, values) {
    return `<td class="sticky-col sticky-actions metadata-link-row-actions">
        <button type="button" title="Duplicate row" aria-label="Duplicate row ${rowIndex + 1}"
                data-link-duplicate-row="${linkEscape(figureId)}" data-row-index="${rowIndex}">⧉</button>
        <button type="button" title="Delete row" aria-label="Delete row ${rowIndex + 1}"
                data-link-delete-row="${linkEscape(figureId)}" data-row-index="${rowIndex}">🗑</button>
    </td>` + HESBAN_LINK_COLUMNS.map((column, columnIndex) => {
        const cell = `<td class="${columnIndex === 0 ? 'sticky-col sticky-no' : columnIndex === 1 ? 'sticky-col sticky-type' : ''}"><textarea data-link-column="${column}">${linkEscape(values[column] || '')}</textarea></td>`;
        return columnIndex === 1 ? cell + '<td class="metadata-diameter-cell"><input class="form-control" disabled placeholder="Match a drawing number"><small>no matched drawing</small></td>' : cell;
    }).join('');
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
    if (row.dataset.reviewKey) result._review_key = row.dataset.reviewKey;
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
    if (deleted.row._review_key) row.dataset.reviewKey = deleted.row._review_key;
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
    tabularState.metadataEditVersions.set(
        figureId, (tabularState.metadataEditVersions.get(figureId) || 0) + 1);
    const body = figureEl.querySelector('.metadata-link-table tbody');
    body.innerHTML = (snapshot.table_rows || []).map((row, index) =>
        `<tr data-link-row="${index}" ${metadataReviewRowAttribute(row, index)}>${metadataDynamicRowCells(figureId, index, row)}</tr>`).join('');
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
    tabularState.metadataUndo.delete(figureId);
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
    const existing = tabularState.metadataLinkState.figures.find(figure =>
        (figure.figure_key || figure.figure_id) === figureId);
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
    const measurements = {};
    const seenMeasurements = new Set();
    figureEl.querySelectorAll('[data-link-diameter]').forEach(input => {
        const maskFile = input.dataset.maskFile;
        if (!maskFile || seenMeasurements.has(maskFile)) return;
        seenMeasurements.add(maskFile);
        if (input.dataset.measurementDirty === '1') {
            const exact = input.dataset.measurementExact;
            measurements[maskFile] = {
                verified_cm: input.dataset.measurementEdited === '1' || exact === ''
                    ? input.value : exact
            };
        }
    });
    return {
        reviewer_revision: Number(figureEl.dataset.reviewerRevision || existing?.reviewer_revision || 0),
        figure_id: figureEl.querySelector('[data-link-figure-id]')?.value || figureId,
        figure_caption: figureEl.querySelector('[data-link-caption]')?.value || '',
        drawing_numbers: drawingNumbers,
        table_rows: tableRows,
        table_pages: pageNames.map(name => existingPages.get(name) || {image_name: name}),
        warning_overrides: warningOverrides,
        measurements
    };
}

function synchronizeDiameterInputs(source, userEdited = true) {
    const figureEl = source.closest('[data-link-figure]');
    const maskFile = source.dataset.maskFile;
    if (!figureEl || !maskFile) return;
    const edited = userEdited || source.dataset.measurementEdited === '1';
    figureEl.querySelectorAll('[data-link-diameter]').forEach(input => {
        if (input.dataset.maskFile !== maskFile) return;
        if (input !== source) input.value = source.value;
        input.dataset.measurementDirty = '1';
        input.dataset.measurementEdited = edited ? '1' : '0';
        input.dataset.measurementStatus = 'verified_manual';
    });
    figureEl.querySelectorAll(`[data-mask-file="${CSS.escape(maskFile)}"] .metadata-measure-status`)
        .forEach(status => { status.textContent = 'Manually corrected'; status.className = 'metadata-measure-status verified_manual'; });
    scheduleMetadataAutosave(figureEl.dataset.linkFigure);
}

function verifyMetadataDiameter(button) {
    const figureEl = button.closest('[data-link-figure]');
    const input = figureEl?.querySelector(`[data-link-diameter][data-mask-file="${CSS.escape(button.dataset.maskFile)}"]`);
    if (!input || !Number.isFinite(Number(input.value)) || Number(input.value) <= 0) {
        window.PyPotteryUtils.showToast('Enter a positive diameter before verifying it', 'warning');
        input?.focus();
        return;
    }
    synchronizeDiameterInputs(input, false);
    figureEl.querySelectorAll(`[data-mask-file="${CSS.escape(button.dataset.maskFile)}"] .metadata-measure-status`)
        .forEach(status => { status.textContent = 'Manually corrected'; status.className = 'metadata-measure-status verified_manual'; });
}

function setupMetadataTableNavigation(figureEl) {
    const section = figureEl.querySelector('.metadata-link-table-section');
    const wrap = section?.querySelector('.metadata-link-table-wrap');
    const table = wrap?.querySelector('.metadata-link-table');
    if (!section || !wrap || !table) return;
    section.querySelector('[data-link-table-expand]')?.addEventListener('click', event => {
        section.classList.toggle('metadata-table-fullscreen');
        event.currentTarget.textContent = section.classList.contains('metadata-table-fullscreen')
            ? 'Exit full screen' : 'Full-screen table';
    });
    section.querySelectorAll('[data-column-jump]').forEach(button => button.addEventListener('click', () => {
        const group = button.dataset.columnJump;
        const firstColumn = {
            fabric: 'fabric_exterior',
            nonplastics: 'nonplastics_type',
            voids: 'voids_type_size',
            surface: 'surface_exterior',
            finish: 'decor'
        }[group];
        const field = firstColumn
            ? table.querySelector(`[data-link-column="${CSS.escape(firstColumn)}"]`)
            : null;
        const cell = field?.closest('td');
        const target = group === 'identity' || !cell
            ? 0
            : Math.max(0, Math.round(cell.getBoundingClientRect().left -
                table.getBoundingClientRect().left - 8));
        wrap.scrollTo({left: target, behavior: 'auto'});
        section.querySelectorAll('[data-column-jump]').forEach(item =>
            item.classList.toggle('active', item === button));
    }));
}

async function redetectMetadataMeasurements(figureId) {
    const button = document.querySelector(`[data-link-measure="${CSS.escape(figureId)}"]`);
    try {
        if (button) { button.disabled = true; button.textContent = 'Saving…'; button.setAttribute('aria-busy', 'true'); }
        const figure = await saveMetadataFigure(figureId, true);
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(figureId)}/measure`, {
                method: 'POST', body: JSON.stringify({reviewer_revision: figure?.reviewer_revision || 0})
            });
        if (!response.success) throw new Error(response.error || 'Could not detect measurements');
        if (button) button.textContent = 'Queued';
        window.PyPotteryUtils.showToast('Measurement reread queued', 'success');
        await loadMetadataLinkState();
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
        if (button) { button.disabled = false; button.textContent = 'Re-read measurements'; }
    } finally {
        button?.removeAttribute('aria-busy');
    }
}

function openMetadataMeasurementEditor(figureId, maskFile = null, scalePage = null) {
    const figure = tabularState.metadataLinkState?.figures.find(item =>
        (item.figure_key || item.figure_id) === figureId);
    if (!figure) return;
    const drawing = maskFile ? (figure.drawings || []).find(item => item.mask_file === maskFile) : null;
    const imageName = scalePage || drawing?.image_name;
    if (!imageName) return;
    const calibrationEntry = Object.entries(figure.scale_calibrations || {}).find(
        ([name, calibration]) => name === imageName || calibration.evidence_image === imageName);
    const calibrationName = calibrationEntry?.[0] || imageName;
    const calibration = structuredClone(calibrationEntry?.[1] || {});
    const measurement = structuredClone(drawing?.measurement || {});
    const mode = drawing ? 'rim' : 'scale';
    const dialog = document.createElement('dialog');
    dialog.className = 'metadata-measure-dialog';
    dialog.innerHTML = `<form method="dialog"><header><div><strong>${mode === 'rim' ? 'Correct rim diameter' : 'Correct 10 cm scale'}</strong>
        <span>${linkEscape(imageName)}</span></div><button value="cancel" aria-label="Close">×</button></header>
        <p>Drag either endpoint. If no endpoints exist, click twice on the image.</p>
        <div class="metadata-measure-canvas-wrap"><canvas></canvas></div>
        <footer><output data-measure-output></output><div class="metadata-measure-zoom-controls">
            <button type="button" data-measure-zoom-out>Zoom out</button>
            <button type="button" data-measure-fit>Fit evidence</button>
            <button type="button" data-measure-zoom-in>Zoom in</button>
        </div><button type="button" data-measure-save>Save and verify</button></footer></form>`;
    document.body.appendChild(dialog);
    const canvas = dialog.querySelector('canvas');
    const context = canvas.getContext('2d');
    const image = new Image();
    const evidenceUrl = `/api/projects/${encodeURIComponent(tabularState.currentProject.project_id)}/metadata-link/evidence/${encodeURIComponent(imageName)}` +
        `?figure=${encodeURIComponent(figureId)}&kind=drawing&overlay=0&measurement=0&v=${Date.now()}`;
    let coordinateSize = (mode === 'rim' ? measurement.image_size : calibration.image_size) ||
        calibration.image_size || measurement.image_size || [1, 1];
    let points = structuredClone(mode === 'rim' ? measurement.rim_endpoints :
        (calibration.p1 && calibration.p2 ? [calibration.p1, calibration.p2] : []));
    let dragIndex = -1;
    let fitView = null;
    let view = null;
    const output = dialog.querySelector('[data-measure-output]');
    const validBounds = raw => {
        if (!Array.isArray(raw) || raw.length !== 4 || raw.some(value => !Number.isFinite(Number(value)))) return null;
        const values = raw.map(Number);
        return values[2] > values[0] && values[3] > values[1] ? values : null;
    };
    const clampView = raw => {
        const pageWidth = Number(coordinateSize[0]) || 1;
        const pageHeight = Number(coordinateSize[1]) || 1;
        const width = Math.min(pageWidth, Math.max(20, raw[2] - raw[0]));
        const height = Math.min(pageHeight, Math.max(20, raw[3] - raw[1]));
        const left = Math.max(0, Math.min(pageWidth - width, raw[0]));
        const top = Math.max(0, Math.min(pageHeight - height, raw[1]));
        return [left, top, left + width, top + height];
    };
    const evidenceView = () => {
        let bounds = validBounds(mode === 'scale' ? calibration.evidence_bounds :
            (measurement.crop || drawing?.bbox));
        if (!bounds && points.length === 2) {
            bounds = [Math.min(points[0][0], points[1][0]), Math.min(points[0][1], points[1][1]),
                      Math.max(points[0][0], points[1][0]), Math.max(points[0][1], points[1][1])];
        }
        if (!bounds) return [0, 0, coordinateSize[0], coordinateSize[1]];
        let [left, top, right, bottom] = bounds;
        if (mode === 'rim') {
            bottom = Math.min(bottom, top + Math.max(35, (bottom - top) * 0.38));
        }
        const width = Math.max(1, right - left);
        const height = Math.max(1, bottom - top);
        const padX = mode === 'scale' ? Math.max(30, width * 0.45) : Math.max(20, width * 0.08);
        const padY = mode === 'scale' ? Math.max(30, height * 5, width * 0.12) : Math.max(20, height * 0.20);
        return clampView([left - padX, top - padY, right + padX, bottom + padY]);
    };
    const resizeCanvas = () => {
        if (!view) return;
        const viewWidth = view[2] - view[0], viewHeight = view[3] - view[1];
        const maxWidth = Math.min(1400, Math.max(180, window.innerWidth - 100));
        const maxHeight = Math.max(240, window.innerHeight - 260);
        const ratio = Math.min(3, maxWidth / viewWidth, maxHeight / viewHeight);
        canvas.width = Math.max(1, Math.round(viewWidth * ratio));
        canvas.height = Math.max(1, Math.round(viewHeight * ratio));
    };
    const updateOutput = () => {
        if (points.length !== 2) { output.textContent = 'Choose two endpoints'; return; }
        const pixels = Math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]);
        if (mode === 'scale') output.textContent = `${pixels.toFixed(1)} px = 10 cm`;
        else {
            const ratio = Number(calibration.px_per_cm || measurement.scale_px_per_cm || 0);
            output.textContent = ratio > 0 ? `${(pixels / ratio).toFixed(1)} cm` : 'Verify the page scale first';
        }
    };
    const redraw = () => {
        if (!image.complete || !canvas.width || !view) return;
        context.clearRect(0, 0, canvas.width, canvas.height);
        const sourceScaleX = image.naturalWidth / coordinateSize[0];
        const sourceScaleY = image.naturalHeight / coordinateSize[1];
        context.drawImage(image,
            view[0] * sourceScaleX, view[1] * sourceScaleY,
            (view[2] - view[0]) * sourceScaleX, (view[3] - view[1]) * sourceScaleY,
            0, 0, canvas.width, canvas.height);
        if (points.length === 2) {
            const sx = canvas.width / (view[2] - view[0]);
            const sy = canvas.height / (view[3] - view[1]);
            const screenPoint = point => [(point[0] - view[0]) * sx, (point[1] - view[1]) * sy];
            const rendered = points.map(screenPoint);
            context.strokeStyle = mode === 'scale'
                ? 'rgba(22, 163, 74, .78)' : 'rgba(14, 165, 233, .78)';
            context.lineWidth = 1.5;
            context.beginPath(); context.moveTo(...rendered[0]);
            context.lineTo(...rendered[1]); context.stroke();
            rendered.forEach((point, index) => {
                context.beginPath();
                context.arc(point[0], point[1], dragIndex === index ? 6 : 4, 0, Math.PI * 2);
                context.stroke();
            });
        }
        updateOutput();
    };
    image.onload = () => {
        if (coordinateSize[0] <= 1 || coordinateSize[1] <= 1) {
            coordinateSize = [image.naturalWidth, image.naturalHeight];
        }
        fitView = evidenceView();
        view = [...fitView];
        resizeCanvas();
        redraw();
    };
    image.src = evidenceUrl;
    const eventPoint = event => {
        const rect = canvas.getBoundingClientRect();
        return [view[0] + (event.clientX - rect.left) * (view[2] - view[0]) / rect.width,
                view[1] + (event.clientY - rect.top) * (view[3] - view[1]) / rect.height];
    };
    canvas.addEventListener('pointerdown', event => {
        const point = eventPoint(event);
        if (points.length < 2) { points.push(point); dragIndex = points.length - 1; }
        else {
            const distances = points.map(item => Math.hypot(item[0] - point[0], item[1] - point[1]));
            dragIndex = distances[0] <= distances[1] ? 0 : 1;
            points[dragIndex] = point;
        }
        canvas.setPointerCapture(event.pointerId); redraw();
    });
    canvas.addEventListener('pointermove', event => {
        if (dragIndex < 0) return;
        points[dragIndex] = eventPoint(event); redraw();
    });
    canvas.addEventListener('pointerup', () => { dragIndex = -1; redraw(); });
    canvas.addEventListener('pointercancel', () => { dragIndex = -1; redraw(); });
    const zoom = factor => {
        if (!view) return;
        const centreX = (view[0] + view[2]) / 2;
        const centreY = (view[1] + view[3]) / 2;
        const halfWidth = (view[2] - view[0]) * factor / 2;
        const halfHeight = (view[3] - view[1]) * factor / 2;
        view = clampView([centreX - halfWidth, centreY - halfHeight,
                          centreX + halfWidth, centreY + halfHeight]);
        resizeCanvas(); redraw();
    };
    dialog.querySelector('[data-measure-zoom-in]').addEventListener('click', () => zoom(0.72));
    dialog.querySelector('[data-measure-zoom-out]').addEventListener('click', () => zoom(1.4));
    dialog.querySelector('[data-measure-fit]').addEventListener('click', () => {
        if (!fitView) return;
        view = [...fitView]; resizeCanvas(); redraw();
    });
    dialog.querySelector('[data-measure-save]').addEventListener('click', async () => {
        if (points.length !== 2) return;
        const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
        const body = {reviewer_revision: Number(figureEl?.dataset.reviewerRevision || figure.reviewer_revision || 0)};
        if (mode === 'scale') body.scale_calibrations = {[calibrationName]: {
            p1: points[0], p2: points[1], real_cm: 10,
            evidence_image: imageName !== calibrationName ? imageName : undefined
        }};
        else body.measurements = {[maskFile]: {rim_endpoints: points}};
        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(figureId)}`,
                {method: 'PATCH', body: JSON.stringify(body)});
            if (!response.success) throw new Error(response.error || 'Could not save measurement');
            dialog.close(); dialog.remove(); await loadMetadataLinkState();
        } catch (error) { window.PyPotteryUtils.showToast(error.message, 'error'); }
    });
    dialog.addEventListener('close', () => dialog.remove());
    dialog.showModal();
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
    tabularState.metadataEditVersions.set(
        figureId, (tabularState.metadataEditVersions.get(figureId) || 0) + 1);
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

function saveMetadataFigure(figureId, quiet = false, retryConflict = true) {
    const previous = tabularState.metadataSaveQueues.get(figureId) || Promise.resolve();
    const queued = previous.catch(() => null).then(() =>
        performMetadataFigureSave(figureId, quiet, retryConflict));
    tabularState.metadataSaveQueues.set(figureId, queued);
    return queued.finally(() => {
        if (tabularState.metadataSaveQueues.get(figureId) === queued) {
            tabularState.metadataSaveQueues.delete(figureId);
        }
    });
}

async function performMetadataFigureSave(figureId, quiet = false, retryConflict = true) {
    const edits = collectMetadataFigureEdits(figureId);
    if (!edits) return null;
    const submittedVersion = tabularState.metadataEditVersions.get(figureId) || 0;
    setMetadataSaveStatus(figureId, 'Saving…');
    const raw = await fetch(
        `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(figureId)}`, {
            method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(edits)
        });
    const response = await raw.json();
    if (raw.status === 409 && response.conflict && retryConflict) {
        const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
        if (figureEl) figureEl.dataset.reviewerRevision = response.reviewer_revision;
        if (!mergeMetadataConflict(figureId, response.figure)) {
            tabularState.metadataDirty.add(figureId);
            setMetadataSaveStatus(figureId, 'Save conflict — review newer changes', true);
            window.PyPotteryUtils.showToast(
                'This figure was also edited elsewhere. Your draft is still visible; review it before saving again.',
                'error');
            return null;
        }
        return performMetadataFigureSave(figureId, quiet, false);
    }
    if (!raw.ok || !response.success) {
        tabularState.metadataDirty.add(figureId);
        setMetadataSaveStatus(figureId, 'Save failed', true);
        if (!quiet) throw new Error(response.error || 'Could not save figure');
        return null;
    }
    const index = tabularState.metadataLinkState.figures.findIndex(figure =>
        (figure.figure_key || figure.figure_id) === figureId);
    if (index >= 0) tabularState.metadataLinkState.figures[index] = response.figure;
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    if (figureEl) figureEl.dataset.reviewerRevision = response.reviewer_revision;
    tabularState.metadataSnapshots.set(figureId, structuredClone(response.figure));
    if ((tabularState.metadataEditVersions.get(figureId) || 0) !== submittedVersion) {
        // Text changed while this request was in flight. Never label the older
        // response Saved or let polling redraw over the newer browser draft.
        tabularState.metadataDirty.add(figureId);
        setMetadataSaveStatus(figureId, 'Unsaved');
        return performMetadataFigureSave(figureId, quiet, true);
    }
    tabularState.metadataDirty.delete(figureId);
    const savedMeasurements = new Map((response.figure.drawings || [])
        .map(drawing => [drawing.mask_file, drawing.measurement || {}]));
    figureEl?.querySelectorAll('[data-link-diameter]').forEach(input => {
        const measurement = savedMeasurements.get(input.dataset.maskFile);
        if (!measurement) return;
        input.dataset.measurementDirty = '0';
        input.dataset.measurementStatus = measurement.status || 'unresolved';
        const value = metadataMeasurementValue(measurement);
        input.dataset.measurementExact = value == null ? '' : String(value);
        input.dataset.measurementEdited = '0';
    });
    setMetadataSaveStatus(figureId, 'Saved');
    updateMetadataWarningsFromFigure(figureId, response.figure);
    updateMetadataReadinessFromFigure(figureId, response.figure);
    refreshMetadataDrawingEvidence(figureId);
    if (!quiet) window.PyPotteryUtils.showToast(`Figure ${figureId} saved`, 'success');
    validateMetadataFigureDom(figureId);
    return response.figure;
}

function refreshMetadataDrawingEvidence(figureId) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    figureEl?.querySelectorAll('[data-evidence-kind="drawing"]').forEach(image => {
        const url = new URL(image.src, window.location.origin);
        url.searchParams.set('v', String(Date.now()));
        image.src = url.toString();
    });
}

function metadataRowsFromFigureElement(figureEl) {
    return [...figureEl.querySelectorAll('.metadata-link-table tbody tr')]
        .map(row => readMetadataRow(row));
}

function metadataComparableRows(rows) {
    return (rows || []).map(row => Object.fromEntries(HESBAN_LINK_COLUMNS
        .map(column => [column, String(row?.[column] || '')])));
}

function metadataPageNames(figure) {
    return (figure?.table_pages || []).map(page => page.image_name).filter(Boolean);
}

function sameMetadataValue(left, right) {
    return JSON.stringify(left ?? null) === JSON.stringify(right ?? null);
}

function mergeMetadataConflict(figureId, serverFigure) {
    const figureEl = document.querySelector(`[data-link-figure="${CSS.escape(figureId)}"]`);
    const baseline = tabularState.metadataSnapshots.get(figureId);
    if (!figureEl || !baseline || !serverFigure) return false;
    let overlappingChange = false;
    const mergeInput = (selector, baselineValue, serverValue) => {
        const input = figureEl.querySelector(selector);
        if (!input) return;
        if (input.value === String(baselineValue || '')) {
            input.value = String(serverValue || '');
        } else if (String(serverValue || '') !== String(baselineValue || '')) {
            overlappingChange = true;
        }
    };
    mergeInput('[data-link-figure-id]', baseline.figure_id, serverFigure.figure_id);
    mergeInput('[data-link-caption]', baseline.figure_caption, serverFigure.figure_caption);
    const pageInput = figureEl.querySelector('[data-link-table-pages]');
    const baselinePages = metadataPageNames(baseline);
    const serverPages = metadataPageNames(serverFigure);
    const localPages = (pageInput?.value || '').split(',').map(value => value.trim()).filter(Boolean);
    if (sameMetadataValue(localPages, baselinePages)) {
        if (pageInput) pageInput.value = serverPages.join(', ');
    } else if (!sameMetadataValue(serverPages, baselinePages)) {
        overlappingChange = true;
    }
    const baselineNumbers = new Map((baseline.drawings || [])
        .map(drawing => [drawing.mask_file, String(drawing.vessel_number || '')]));
    const serverNumbers = new Map((serverFigure.drawings || [])
        .map(drawing => [drawing.mask_file, String(drawing.vessel_number || '')]));
    figureEl.querySelectorAll('[data-link-drawing-number]').forEach(input => {
        const before = baselineNumbers.get(input.dataset.maskFile) || '';
        const server = serverNumbers.get(input.dataset.maskFile) || '';
        if (input.value === before) input.value = server;
        else if (server !== before) overlappingChange = true;
    });
    const baselineMeasurements = new Map((baseline.drawings || [])
        .map(drawing => [drawing.mask_file, drawing.measurement || {}]));
    const serverMeasurements = new Map((serverFigure.drawings || [])
        .map(drawing => [drawing.mask_file, drawing.measurement || {}]));
    figureEl.querySelectorAll('[data-link-diameter]').forEach(input => {
        const before = baselineMeasurements.get(input.dataset.maskFile) || {};
        const server = serverMeasurements.get(input.dataset.maskFile) || {};
        if (input.dataset.measurementDirty !== '1') {
            const value = metadataMeasurementValue(server);
            input.value = value == null ? '' : Number(value).toFixed(1);
            input.dataset.measurementStatus = server.status || 'unresolved';
            input.dataset.measurementExact = value == null ? '' : String(value);
            input.dataset.measurementEdited = '0';
        } else if (!sameMetadataValue(server, before)) {
            overlappingChange = true;
        }
    });
    const localRows = metadataComparableRows(metadataRowsFromFigureElement(figureEl));
    const baselineRows = metadataComparableRows(baseline.table_rows);
    const serverRows = metadataComparableRows(serverFigure.table_rows);
    if (sameMetadataValue(localRows, baselineRows)) {
        const body = figureEl.querySelector('.metadata-link-table tbody');
        body.innerHTML = (serverFigure.table_rows || []).map((row, index) =>
            `<tr data-link-row="${index}" ${metadataReviewRowAttribute(row, index)}>${metadataDynamicRowCells(figureId, index, row)}</tr>`).join('');
        body.querySelectorAll('tr').forEach(row => wireMetadataDynamicRow(row, figureId));
    } else if (!sameMetadataValue(serverRows, baselineRows)) {
        overlappingChange = true;
    }
    const simplifyOverrides = overrides => Object.fromEntries(Object.entries(overrides || {})
        .map(([id, value]) => [id, {reason: value.reason || '', note: value.note || ''}]));
    const localOverrides = {};
    figureEl.querySelectorAll('[data-warning-id][data-override-active="1"]').forEach(card => {
        localOverrides[card.dataset.warningId] = {
            reason: card.querySelector('[data-warning-reason]')?.value || '',
            note: card.querySelector('[data-warning-note]')?.value || ''
        };
    });
    const baselineOverrides = simplifyOverrides(baseline.warning_overrides);
    const serverOverrides = simplifyOverrides(serverFigure.warning_overrides);
    if (sameMetadataValue(localOverrides, baselineOverrides)) {
        figureEl.querySelectorAll('[data-warning-id]').forEach(card => {
            const override = serverOverrides[card.dataset.warningId];
            card.dataset.overrideActive = override ? '1' : '0';
            card.classList.toggle('resolved', !!override);
            card.classList.toggle('blocking', !override);
            const reason = card.querySelector('[data-warning-reason]');
            const note = card.querySelector('[data-warning-note]');
            const toggle = card.querySelector('[data-warning-toggle]');
            if (reason) reason.value = override?.reason || reason.options?.[0]?.value || '';
            if (note) note.value = override?.note || '';
            if (toggle) toggle.textContent = override
                ? 'Remove override' : 'Mark reviewed and ignore';
        });
    } else if (!sameMetadataValue(serverOverrides, baselineOverrides)) {
        overlappingChange = true;
    }
    if (overlappingChange) return false;
    const index = tabularState.metadataLinkState.figures.findIndex(figure =>
        (figure.figure_key || figure.figure_id) === figureId);
    if (index >= 0) tabularState.metadataLinkState.figures[index] = serverFigure;
    tabularState.metadataSnapshots.set(figureId, structuredClone(serverFigure));
    validateMetadataFigureDom(figureId);
    return true;
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
        figure.processing_status === 'processing';
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
    let hasDraftBlocker = false;
    drawingInputs.forEach(input => {
        const value = input.value.trim();
        const invalid = !/^[1-9]\d*[a-z]?$/i.test(value) || drawingCounts.get(value) > 1;
        hasDraftBlocker ||= invalid;
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
        hasDraftBlocker ||= invalid || !!unmatched;
        row.classList.toggle('metadata-link-row-invalid', invalid);
        row.classList.toggle('metadata-link-row-unmatched', unmatched);
        input.setAttribute('aria-invalid', invalid ? 'true' : 'false');
    });
    drawingInputs.forEach(input => {
        const value = input.value.trim();
        if (value && !rowCounts.has(value)) hasDraftBlocker = true;
    });
    if (hasDraftBlocker) {
        const approve = figureEl.querySelector('[data-link-approve]');
        if (approve) approve.disabled = true;
        const readiness = figureEl.querySelector('.metadata-link-readiness');
        readiness?.classList.remove('ready');
        readiness?.classList.add('blocked');
        const text = readiness?.querySelector('span');
        if (text) text.textContent = 'The current draft has missing, duplicate, or unmatched numbers.';
    }
}

async function approveMetadataFigure(figureId) {
    try {
        const figure = await saveMetadataFigure(figureId, true);
        if (!figure || figure.status !== 'ready') throw new Error('Resolve the warnings before approval');
        const approvedFigureId = figure.figure_id || figureId;
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata-link/apply`, {
                method: 'POST', body: JSON.stringify({
                    figure_ids: [approvedFigureId], replace_imported: true
                })
            });
        if (!response.success) throw new Error(response.error || 'Could not apply figure');
        window.PyPotteryUtils.showToast(`Applied ${response.applied_rows} rows to CSV`, 'success');
        await loadMetadataLinkState();
        await loadTabularData(tabularState.currentIndex);
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
        throw error;
    }
}

async function rerunMetadataFigure(figureId) {
    const button = document.querySelector(`[data-link-rerun="${CSS.escape(figureId)}"]`);
    try {
        if (button) { button.disabled = true; button.textContent = 'Saving…'; button.setAttribute('aria-busy', 'true'); }
        const saved = await saveMetadataFigure(figureId, true);
        const targetId = saved?.figure_id || figureId;
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata-link/figures/${encodeURIComponent(targetId)}/rerun`, {
                method: 'POST', body: JSON.stringify({backend: 'ocr'})
            });
        if (!response.success) throw new Error(response.error || 'Could not rerun OCR');
        if (button) button.textContent = 'Queued';
        window.PyPotteryUtils.showToast(`OCR reread queued for figure ${targetId}`, 'success');
        await loadMetadataLinkState();
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
        if (button) { button.disabled = false; button.textContent = 'Re-read this figure'; }
    } finally {
        button?.removeAttribute('aria-busy');
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
        throw error;
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
