// Clean researcher-facing export workflow.
(function () {
    const state = {
        masks: [], rows: [], columns: [], unresolved: [], saveTimer: null,
        savePromise: null, settingsDirty: false
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

    function acronym() {
        return document.getElementById('research-export-acronym')?.value.trim() || '';
    }

    function validAcronym() {
        return /^[A-Za-z0-9_]+$/.test(acronym());
    }

    function renderSummary(summary = {}) {
        const container = document.getElementById('research-export-summary');
        if (!container) return;
        const cards = [
            ['Approved figures', summary.approved_figures || 0],
            ['Approved vessels', summary.approved_vessels || 0],
            ['Unresolved figures', summary.unresolved_figures || 0],
            ['Included masks', summary.included_masks || 0],
            ['Excluded masks', summary.excluded_masks || 0],
        ];
        container.innerHTML = cards.map(([label, value]) =>
            `<div><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join('');
    }

    function renderMasks() {
        const container = document.getElementById('research-export-masks');
        if (!container) return;
        const query = (document.getElementById('research-export-search')?.value || '').trim().toLowerCase();
        const visible = state.masks.filter(mask =>
            `${mask.figure} ${mask.vessel_number} ${mask.vessel_type}`.toLowerCase().includes(query));
        if (!visible.length) {
            container.innerHTML = '<p class="research-export-empty">No approved vessel masks match this search.</p>';
            return;
        }
        container.innerHTML = visible.map(mask => `
            <label class="research-export-mask ${mask.included ? '' : 'excluded'}">
                <input type="checkbox" data-export-mask="${escapeHtml(mask.mask_key)}" ${mask.included ? 'checked' : ''}>
                <img src="${escapeHtml(mask.thumbnail_url)}" loading="lazy" decoding="async"
                     alt="Vessel ${escapeHtml(mask.vessel_number)}">
                <span><strong>Figure ${escapeHtml(mask.figure)}, No. ${escapeHtml(mask.vessel_number)}</strong>
                    <small>${escapeHtml(mask.vessel_type || 'Type not recorded')}</small>
                    <small>${mask.included ? 'Included in export' : 'Excluded from export'}</small></span>
            </label>`).join('');
        container.querySelectorAll('[data-export-mask]').forEach(input => {
            input.addEventListener('change', () => {
                const mask = state.masks.find(item => item.mask_key === input.dataset.exportMask);
                if (mask) mask.included = input.checked;
                renderMasks();
                scheduleSave();
            });
        });
    }

    function renderUnresolved() {
        const container = document.getElementById('research-export-unresolved');
        if (!container) return;
        if (!state.unresolved.length) {
            container.innerHTML = '<p class="export-ready-message">All discovered figures are approved for export.</p>';
            return;
        }
        container.innerHTML = `<details><summary>${state.unresolved.length} unresolved figure(s) are withheld from export</summary>
            <ul>${state.unresolved.map(item => `<li><strong>Figure ${escapeHtml(item.figure || '?')}</strong> — ${escapeHtml(item.status)}
                ${item.warnings?.length ? `<small>${escapeHtml(item.warnings.join(' '))}</small>` : ''}</li>`).join('')}</ul></details>`;
    }

    function renderTable() {
        const head = document.getElementById('research-export-table-head');
        const body = document.getElementById('research-export-table-body');
        if (!head || !body) return;
        head.innerHTML = `<tr>${state.columns.map(column => `<th title="${escapeHtml(column)}">${escapeHtml(column)}</th>`).join('')}</tr>`;
        body.innerHTML = state.rows.length ? state.rows.map(row =>
            `<tr>${state.columns.map(column => `<td>${escapeHtml(row[column] || '')}</td>`).join('')}</tr>`).join('') :
            `<tr><td colspan="${Math.max(1, state.columns.length)}">No approved, included rows are ready to export.</td></tr>`;
    }

    async function loadResearchExport() {
        const pid = projectId();
        if (!pid) return;
        const status = document.getElementById('research-export-status');
        try {
            if (status) status.textContent = 'Loading export preview…';
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${encodeURIComponent(pid)}/export/preview?acronym=${encodeURIComponent(acronym() || 'DATA')}`);
            state.masks = response.masks || [];
            state.rows = response.rows || [];
            state.columns = response.columns || [];
            state.unresolved = response.unresolved || [];
            renderSummary(response.summary);
            renderMasks();
            renderUnresolved();
            renderTable();
            if (status) textStatus(status, '', '');
        } catch (error) {
            if (status) textStatus(status, error.message, 'error');
        }
    }

    function textStatus(element, message, kind) {
        element.textContent = message;
        element.className = `status-message${kind ? ` ${kind}` : ''}`;
    }

    function showExportProgress(message, percent = null, kind = 'info') {
        const status = document.getElementById('research-export-status');
        if (!status) return;
        status.replaceChildren();
        status.className = `status-message ${kind} research-export-progress-status`;
        const label = document.createElement('strong');
        label.textContent = message;
        const track = document.createElement('div');
        track.className = `research-export-progress-track${percent == null ? ' indeterminate' : ''}`;
        track.setAttribute('role', 'progressbar');
        track.setAttribute('aria-valuemin', '0');
        track.setAttribute('aria-valuemax', '100');
        const fill = document.createElement('span');
        fill.className = 'research-export-progress-fill';
        if (percent == null) {
            track.setAttribute('aria-valuetext', message);
        } else {
            const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
            track.setAttribute('aria-valuenow', String(safePercent));
            fill.style.setProperty('--export-progress', `${safePercent}%`);
        }
        track.appendChild(fill);
        status.append(label, track);
    }

    function transferPreparedZip(url, expectedSize) {
        return new Promise((resolve, reject) => {
            const request = new XMLHttpRequest();
            request.open('GET', url, true);
            request.responseType = 'blob';
            request.onprogress = event => {
                const total = event.lengthComputable && event.total > 0
                    ? event.total : expectedSize;
                const percent = total > 0 ? event.loaded * 100 / total : null;
                showExportProgress(
                    percent == null
                        ? `Downloading ZIP: ${(event.loaded / 1048576).toFixed(1)} MB`
                        : `Downloading ZIP: ${Math.min(100, Math.round(percent))}%`,
                    percent);
            };
            request.onload = () => {
                if (request.status < 200 || request.status >= 300) {
                    reject(new Error(`The prepared ZIP request failed (${request.status}).`));
                    return;
                }
                resolve(request.response);
            };
            request.onerror = () => reject(
                new Error('Chrome could not transfer the prepared ZIP from SherdScope.'));
            request.onabort = () => reject(new Error('The ZIP transfer was cancelled.'));
            request.send();
        });
    }

    function scheduleSave() {
        clearTimeout(state.saveTimer);
        state.settingsDirty = true;
        const label = document.getElementById('research-export-save-status');
        if (label) label.textContent = 'Unsaved';
        state.saveTimer = setTimeout(() => {
            state.saveTimer = null;
            startSettingsSave();
        }, 350);
    }

    function startSettingsSave() {
        if (!state.savePromise) {
            state.savePromise = saveSettings().finally(() => { state.savePromise = null; });
        }
        return state.savePromise;
    }

    async function saveSettings() {
        const pid = projectId();
        if (!pid) return;
        const label = document.getElementById('research-export-save-status');
        if (label) label.textContent = 'Saving…';
        try {
            await window.PyPotteryUtils.apiRequest(`/api/projects/${encodeURIComponent(pid)}/export/settings`, {
                method: 'PATCH',
                body: JSON.stringify({
                    excluded_masks: state.masks.filter(mask => !mask.included).map(mask => mask.mask_key),
                    known_masks: state.masks.map(mask => mask.mask_key)
                })
            });
            state.settingsDirty = false;
            if (label) label.textContent = 'Saved';
            await loadResearchExport();
            return true;
        } catch (error) {
            if (label) label.textContent = 'Save failed';
            window.PyPotteryUtils.showToast(error.message, 'error');
            return false;
        }
    }

    async function download(kind) {
        const pid = projectId();
        if (!pid || !validAcronym()) {
            window.PyPotteryUtils.showToast('Enter an acronym using only letters, numbers, and underscores.', 'warning');
            return;
        }
        const button = document.getElementById(kind === 'csv' ? 'research-export-csv' : 'research-export-dataset');
        const originalButtonText = button?.textContent;
        try {
            if (button) {
                button.disabled = true;
                button.setAttribute('aria-busy', 'true');
            }
            // A fast click immediately after changing a checkbox must export
            // the new selection, not the last server-saved selection.
            if (state.saveTimer) {
                clearTimeout(state.saveTimer);
                state.saveTimer = null;
            }
            const saved = state.settingsDirty ? await startSettingsSave()
                : state.savePromise ? await state.savePromise : true;
            if (!saved) throw new Error('Export choices could not be saved. Please try again.');
            if (kind === 'dataset') {
                if (button) button.textContent = 'Preparing ZIP...';
                showExportProgress('Preparing the dataset ZIP...');
                const preparedResponse = await fetch(
                    `/api/projects/${encodeURIComponent(pid)}/export/dataset/prepare`, {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({acronym: acronym()})
                    });
                const prepared = await preparedResponse.json().catch(() => ({}));
                if (!preparedResponse.ok || !prepared.success || !prepared.download_url) {
                    throw new Error(prepared.error || 'The dataset ZIP could not be prepared.');
                }
                if (!Number(prepared.size)) {
                    throw new Error('The server prepared an empty dataset ZIP.');
                }
                if (button) button.textContent = 'Downloading ZIP...';
                showExportProgress('Downloading ZIP: 0%', 0);
                const expectedSize = Number(prepared.size);
                const blob = await transferPreparedZip(
                    prepared.transfer_url || prepared.download_url, expectedSize);
                if (!blob.size || (expectedSize > 0 && blob.size !== expectedSize)) {
                    throw new Error(
                        `The ZIP transfer was incomplete (${blob.size} of ${expectedSize} bytes).`);
                }
                const objectUrl = URL.createObjectURL(blob);
                const anchor = document.createElement('a');
                anchor.href = objectUrl;
                anchor.download = prepared.filename || `${acronym()}.zip`;
                anchor.hidden = true;
                document.body.appendChild(anchor);
                anchor.click();
                window.setTimeout(() => {
                    anchor.remove();
                    URL.revokeObjectURL(objectUrl);
                }, 5 * 60 * 1000);
                showExportProgress(
                    `ZIP ready: ${(blob.size / 1048576).toFixed(1)} MB downloaded`, 100, 'success');
                window.PyPotteryUtils.showToast('Dataset ZIP downloaded', 'success');
                return;
            }
            const response = await fetch(`/api/projects/${encodeURIComponent(pid)}/export/${kind}`, {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({acronym: acronym()})
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.error || 'Export failed');
            }
            const blob = await response.blob();
            if (!blob.size) throw new Error('The server returned an empty export. Please try again.');
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `${acronym()}_metadata.csv`;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            // Large ZIP downloads can still be reading this object URL after
            // the synthetic click returns. Revoke it later; immediate cleanup
            // can produce a zero-byte or corrupt archive while small CSVs work.
            window.setTimeout(() => URL.revokeObjectURL(url), 60000);
            window.PyPotteryUtils.showToast('Export downloaded', 'success');
        } catch (error) {
            const status = document.getElementById('research-export-status');
            if (status) textStatus(status, error.message, 'error');
            window.PyPotteryUtils.showToast(error.message, 'error');
        } finally {
            if (button) {
                button.disabled = false;
                button.removeAttribute('aria-busy');
                button.textContent = originalButtonText;
            }
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        document.getElementById('research-export-search')?.addEventListener('input', renderMasks);
        document.getElementById('research-export-acronym')?.addEventListener('change', loadResearchExport);
        document.getElementById('research-export-select-all')?.addEventListener('click', () => {
            state.masks.forEach(mask => { mask.included = true; }); renderMasks(); scheduleSave();
        });
        document.getElementById('research-export-clear-all')?.addEventListener('click', () => {
            state.masks.forEach(mask => { mask.included = false; }); renderMasks(); scheduleSave();
        });
        document.getElementById('research-export-csv')?.addEventListener('click', () => download('csv'));
        document.getElementById('research-export-dataset')?.addEventListener('click', () => download('dataset'));
        window.addEventListener('projectChanged', loadResearchExport);
    });

    window.loadResearchExport = loadResearchExport;
})();
