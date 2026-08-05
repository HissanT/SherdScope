(function () {
    const state = {
        queryId: null,
        runId: null,
        result: null,
        forcedResult: null,
        libraryReady: false,
        previewObjectUrl: null,
        annotationImage: null,
        activeCurve: 'fracture',
        drawing: false,
        curves: { exterior: [], interior: [], fracture: [], rim_point: [] },
        masterBoundary: null,
        viewport: null,
        zoom: 1,
        panX: 0,
        panY: 0,
        mode: 'draw',
        panPointer: null,
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

    function number(value, digits = 4) {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed.toFixed(digits) : '—';
    }

    function setStatus(id, message, kind = '') {
        const element = document.getElementById(id);
        if (!element) return;
        element.textContent = message;
        element.className = `status-message ${kind}`;
    }

    async function loadMatcherTab() {
        const pid = projectId();
        const badge = document.getElementById('matcher-library-badge');
        const runButton = document.getElementById('matcher-run-btn');
        if (!pid) {
            state.libraryReady = false;
            if (badge) {
                badge.textContent = 'Select a project';
                badge.className = 'matcher-library-badge blocked';
            }
            if (runButton) runButton.disabled = true;
            return;
        }
        try {
            const response = await fetch(`/api/projects/${pid}/matcher/library`);
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Could not read contour library');
            const status = data.status;
            state.libraryReady = !!status.ready_to_match;
            if (badge) {
                badge.textContent = state.libraryReady
                    ? `${status.built} canonical references ready`
                    : status.solver && !status.solver.available
                        ? 'POT matcher unavailable — restart after installing dependencies'
                        : status.pending
                        ? `${status.pending} profile masks still need review`
                        : status.unresolved_flags
                            ? `${status.unresolved_flags} contour flags need review`
                            : 'Build canonical contours in Review Profiles';
                badge.className = `matcher-library-badge ${state.libraryReady ? 'ready' : 'blocked'}`;
            }
            if (runButton) runButton.disabled = !(state.libraryReady && state.queryId);
        } catch (error) {
            state.libraryReady = false;
            if (badge) {
                badge.textContent = error.message;
                badge.className = 'matcher-library-badge blocked';
            }
        }
    }

    function queryMetadata() {
        return {
            query_id: document.getElementById('matcher-query-name')?.value?.trim() || '',
            rim_diameter_cm: document.getElementById('matcher-query-diameter')?.value || '',
            fabric: document.getElementById('matcher-query-fabric')?.value?.trim() || '',
            surface: document.getElementById('matcher-query-surface')?.value?.trim() || '',
            notes: document.getElementById('matcher-query-notes')?.value?.trim() || '',
        };
    }

    const curveColors = {
        exterior: '#146ebe',
        interior: '#e15a2d',
        fracture: '#9146b9',
        rim_point: '#dc9819',
    };

    function requiredPoints(name) {
        if (name === 'rim_point') return 1;
        return name === 'exterior' || name === 'interior' ? 8 : 3;
    }

    function updateCurveStatus() {
        const labels = { exterior: 'Exterior', interior: 'Interior', fracture: 'Fracture', rim_point: 'Rim point' };
        const element = document.getElementById('matcher-curve-status');
        if (element) {
            element.textContent = Object.keys(labels).map(name =>
                `${labels[name]}: ${state.curves[name].length >= requiredPoints(name) ? 'drawn' : 'not drawn'}`
            ).join(' · ');
        }
    }

    function redrawAnnotation() {
        const canvas = document.getElementById('matcher-annotation-canvas');
        if (!canvas) return;
        const width = Math.max(320, Math.round(canvas.clientWidth || 640));
        const height = Math.max(240, Math.round(canvas.clientHeight || 330));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, width, height);
        context.fillStyle = '#f8fafc';
        context.fillRect(0, 0, width, height);
        if (!state.annotationImage) {
            context.fillStyle = '#68758a';
            context.textAlign = 'center';
            context.fillText('Choose a PNG to begin tracing', width / 2, height / 2);
            state.viewport = null;
            return;
        }
        const fitScale = Math.min(width / state.annotationImage.naturalWidth, height / state.annotationImage.naturalHeight);
        const scale = fitScale * state.zoom;
        const drawnWidth = state.annotationImage.naturalWidth * scale;
        const drawnHeight = state.annotationImage.naturalHeight * scale;
        const offsetX = (width - drawnWidth) / 2 + state.panX;
        const offsetY = (height - drawnHeight) / 2 + state.panY;
        state.viewport = { scale, offsetX, offsetY, width: drawnWidth, height: drawnHeight };
        context.drawImage(state.annotationImage, offsetX, offsetY, drawnWidth, drawnHeight);
        context.lineCap = 'round';
        context.lineJoin = 'round';
        Object.entries(state.curves).forEach(([name, points]) => {
            if (name === 'rim_point') return;
            if (points.length < 2) return;
            context.beginPath();
            points.forEach((point, index) => {
                const x = offsetX + point[0] * scale;
                const y = offsetY + point[1] * scale;
                if (index === 0) context.moveTo(x, y);
                else context.lineTo(x, y);
            });
            context.strokeStyle = curveColors[name];
            context.lineWidth = name === 'fracture' ? 4 : 3;
            context.stroke();
        });
        if (state.curves.rim_point.length) {
            const point = state.curves.rim_point[0];
            const x = offsetX + point[0] * scale;
            const y = offsetY + point[1] * scale;
            context.fillStyle = curveColors.rim_point;
            context.strokeStyle = '#fff';
            context.lineWidth = 2;
            context.beginPath();
            context.arc(x, y, 7, 0, Math.PI * 2);
            context.fill();
            context.stroke();
        }
    }

    function sourcePoint(event) {
        const canvas = document.getElementById('matcher-annotation-canvas');
        const viewport = state.viewport;
        if (!canvas || !viewport) return null;
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * canvas.width / rect.width;
        const y = (event.clientY - rect.top) * canvas.height / rect.height;
        if (x < viewport.offsetX || x > viewport.offsetX + viewport.width ||
            y < viewport.offsetY || y > viewport.offsetY + viewport.height) return null;
        return [(x - viewport.offsetX) / viewport.scale, (y - viewport.offsetY) / viewport.scale];
    }

    function beginCurve(event) {
        if (state.mode === 'pan') {
            event.currentTarget.setPointerCapture(event.pointerId);
            state.panPointer = { x: event.clientX, y: event.clientY };
            return;
        }
        const point = sourcePoint(event);
        if (!point) return;
        if (state.activeCurve === 'rim_point') {
            state.curves.rim_point = [point];
            state.curves.exterior = [];
            state.curves.interior = [];
            state.masterBoundary = null;
            state.queryId = null;
            state.runId = null;
            state.result = null;
            state.forcedResult = null;
            setEvaluationReady(false);
            document.getElementById('matcher-run-btn').disabled = true;
            document.getElementById('matcher-forced-score-btn').disabled = true;
            updateCurveStatus();
            redrawAnnotation();
            return;
        }
        event.currentTarget.setPointerCapture(event.pointerId);
        state.drawing = true;
        if (state.activeCurve === 'fracture') {
            state.curves.exterior = [];
            state.curves.interior = [];
        }
        if (state.activeCurve === 'fracture' ||
            state.activeCurve === 'exterior' ||
            state.activeCurve === 'interior') {
            state.masterBoundary = null;
        }
        state.curves[state.activeCurve] = [point];
        redrawAnnotation();
    }

    function extendCurve(event) {
        if (state.mode === 'pan' && state.panPointer) {
            const canvas = event.currentTarget;
            const rect = canvas.getBoundingClientRect();
            state.panX += (event.clientX - state.panPointer.x) * canvas.width / rect.width;
            state.panY += (event.clientY - state.panPointer.y) * canvas.height / rect.height;
            state.panPointer = { x: event.clientX, y: event.clientY };
            redrawAnnotation();
            return;
        }
        if (!state.drawing) return;
        const point = sourcePoint(event);
        if (!point) return;
        const points = state.curves[state.activeCurve];
        const previous = points[points.length - 1];
        if (!previous || Math.hypot(point[0] - previous[0], point[1] - previous[1]) >= 1.5) {
            points.push(point);
            redrawAnnotation();
        }
    }

    function finishCurve() {
        if (state.panPointer) {
            state.panPointer = null;
            return;
        }
        if (!state.drawing) return;
        state.drawing = false;
        state.queryId = null;
        state.runId = null;
        state.result = null;
        state.forcedResult = null;
        setEvaluationReady(false);
        document.getElementById('matcher-run-btn').disabled = true;
        document.getElementById('matcher-forced-score-btn').disabled = true;
        updateCurveStatus();
    }

    function selectCurve(name) {
        state.activeCurve = name;
        state.mode = 'draw';
        document.getElementById('matcher-pan-tool')?.classList.remove('active');
        document.querySelectorAll('.matcher-curve-tool').forEach(button => {
            button.classList.toggle('active', button.dataset.curve === name);
        });
    }

    function updateZoomLevel() {
        const output = document.getElementById('matcher-zoom-level');
        if (output) output.textContent = `${Math.round(state.zoom * 100)}%`;
    }

    function setZoom(nextZoom, focalX = null, focalY = null) {
        const canvas = document.getElementById('matcher-annotation-canvas');
        if (!canvas || !state.annotationImage) return;
        const oldViewport = state.viewport;
        const clamped = Math.min(8, Math.max(0.5, nextZoom));
        if (oldViewport && focalX != null && focalY != null) {
            const sourceX = (focalX - oldViewport.offsetX) / oldViewport.scale;
            const sourceY = (focalY - oldViewport.offsetY) / oldViewport.scale;
            const fitScale = Math.min(
                canvas.width / state.annotationImage.naturalWidth,
                canvas.height / state.annotationImage.naturalHeight
            );
            const nextScale = fitScale * clamped;
            const nextWidth = state.annotationImage.naturalWidth * nextScale;
            const nextHeight = state.annotationImage.naturalHeight * nextScale;
            state.panX = focalX - (canvas.width - nextWidth) / 2 - sourceX * nextScale;
            state.panY = focalY - (canvas.height - nextHeight) / 2 - sourceY * nextScale;
        }
        state.zoom = clamped;
        updateZoomLevel();
        redrawAnnotation();
    }

    function resetZoom() {
        state.zoom = 1;
        state.panX = 0;
        state.panY = 0;
        updateZoomLevel();
        redrawAnnotation();
    }

    function wheelZoom(event) {
        if (!state.annotationImage) return;
        event.preventDefault();
        const canvas = event.currentTarget;
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * canvas.width / rect.width;
        const y = (event.clientY - rect.top) * canvas.height / rect.height;
        setZoom(state.zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2), x, y);
    }

    async function prepareQuery() {
        const pid = projectId();
        const fileInput = document.getElementById('matcher-query-file');
        const file = fileInput?.files?.[0];
        if (!pid) {
            setStatus('matcher-query-status', 'Select a project first.', 'error');
            return;
        }
        if (!file) {
            setStatus('matcher-query-status', 'Choose a PNG query file.', 'error');
            return;
        }
        if (!file.name.toLowerCase().endsWith('.png')) {
            setStatus('matcher-query-status', 'Query input must be a PNG file.', 'error');
            return;
        }
        const missing = Object.entries(state.curves)
            .filter(([name, points]) => points.length < requiredPoints(name))
            .map(([name]) => name);
        if (missing.length) {
            setStatus('matcher-query-status', `Draw the ${missing.join(', ')} curve${missing.length === 1 ? '' : 's'} first.`, 'error');
            return;
        }
        const button = document.getElementById('matcher-preprocess-btn');
        try {
            button.disabled = true;
            button.textContent = 'Preparing…';
            setStatus('matcher-query-status', 'Detecting foreground and extracting three curves…');
            const body = new FormData();
            body.append('file', file);
            body.append('metadata', JSON.stringify(queryMetadata()));
            body.append('manual_curves', JSON.stringify({
                ...state.curves,
                ...(state.masterBoundary ? { master_boundary: state.masterBoundary } : {}),
            }));
            const response = await fetch(`/api/projects/${pid}/matcher/query`, {
                method: 'POST',
                body,
            });
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Query preprocessing failed');
            state.queryId = data.query.query_id;
            state.runId = null;
            state.result = null;
            state.forcedResult = null;
            setEvaluationReady(false);
            document.getElementById('matcher-forced-score-btn').disabled = true;
            document.getElementById('matcher-forced-results').innerHTML = '';
            document.getElementById('matcher-contour-preview').src =
                `${data.query.preview_url}?v=${Date.now()}`;
            setStatus('matcher-query-status', 'Query prepared. Metadata is stored but ignored by ranking.', 'success');
            document.getElementById('matcher-run-btn').disabled = !state.libraryReady;
        } catch (error) {
            state.queryId = null;
            state.runId = null;
            state.result = null;
            state.forcedResult = null;
            setEvaluationReady(false);
            document.getElementById('matcher-run-btn').disabled = true;
            document.getElementById('matcher-forced-score-btn').disabled = true;
            setStatus('matcher-query-status', error.message, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Prepare query';
        }
    }

    async function autoTraceWalls() {
        const pid = projectId();
        const fileInput = document.getElementById('matcher-query-file');
        const file = fileInput?.files?.[0];
        if (!pid) {
            setStatus('matcher-query-status', 'Select a project first.', 'error');
            return;
        }
        if (!file) {
            setStatus('matcher-query-status', 'Choose a PNG query file first.', 'error');
            return;
        }
        if (state.curves.fracture.length < requiredPoints('fracture')) {
            setStatus('matcher-query-status', 'Draw the purple fracture curve first.', 'error');
            return;
        }
        if (state.curves.rim_point.length < requiredPoints('rim_point')) {
            setStatus('matcher-query-status', 'Place the gold rim split point first.', 'error');
            return;
        }
        const button = document.getElementById('matcher-auto-curves');
        try {
            button.disabled = true;
            button.textContent = 'Tracing...';
            setStatus('matcher-query-status', 'Auto-tracing exterior and interior from the silhouette...');
            const body = new FormData();
            body.append('file', file);
            body.append('manual_curves', JSON.stringify({
                fracture: state.curves.fracture,
                rim_point: state.curves.rim_point[0],
            }));
            const response = await fetch(`/api/projects/${pid}/matcher/query/auto-curves`, {
                method: 'POST',
                body,
            });
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Auto trace failed');
            state.curves.exterior = data.curves.exterior || [];
            state.curves.interior = data.curves.interior || [];
            state.curves.fracture = data.curves.fracture || state.curves.fracture;
            state.curves.rim_point = data.curves.rim_point ? [data.curves.rim_point] : state.curves.rim_point;
            state.masterBoundary = data.curves.master_boundary || null;
            state.queryId = null;
            state.runId = null;
            state.result = null;
            state.forcedResult = null;
            setEvaluationReady(false);
            document.getElementById('matcher-run-btn').disabled = true;
            document.getElementById('matcher-forced-score-btn').disabled = true;
            updateCurveStatus();
            redrawAnnotation();
            setStatus('matcher-query-status', 'Auto-traced blue/red curves. You can adjust them manually or prepare the query.', 'success');
        } catch (error) {
            setStatus('matcher-query-status', error.message, 'error');
        } finally {
            button.disabled = false;
            button.textContent = 'Auto trace blue/red';
        }
    }

    async function runMatcher() {
        const pid = projectId();
        if (!pid || !state.queryId) return;
        const button = document.getElementById('matcher-run-btn');
        try {
            button.disabled = true;
            button.textContent = 'Matching…';
            document.getElementById('matcher-results').innerHTML = '';
            document.getElementById('matcher-forced-results').innerHTML = '';
            document.getElementById('matcher-forced-score-btn').disabled = true;
            state.result = null;
            state.forcedResult = null;
            setEvaluationReady(false);
            setStatus('matcher-progress', 'Starting shape-only match…');
            const response = await fetch(`/api/projects/${pid}/matcher/runs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query_id: state.queryId }),
            });
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Could not start matcher');
            state.runId = data.run_id;
            while (true) {
                await new Promise(resolve => setTimeout(resolve, 700));
                const poll = await fetch(`/api/projects/${pid}/matcher/jobs/${state.runId}`);
                const payload = await poll.json();
                if (!payload.success) throw new Error(payload.error || 'Matcher job disappeared');
                const job = payload.job;
                const percent = job.total ? Math.round(100 * job.current / job.total) : 0;
                setStatus('matcher-progress', `${job.message}${job.total ? ` (${percent}%)` : ''}`);
                if (job.state === 'failed') throw new Error(job.error || 'Matcher failed');
                if (job.state === 'complete') break;
            }
            const resultResponse = await fetch(`/api/projects/${pid}/matcher/runs/${state.runId}`);
            const resultData = await resultResponse.json();
            if (!resultData.success || !resultData.result) {
                throw new Error(resultData.error || 'Match result could not be loaded');
            }
            state.result = resultData.result;
            renderResults(state.result);
            document.getElementById('matcher-forced-score-btn').disabled = false;
            setEvaluationReady(true);
            const retrieval = resultData.result.retrieval || {};
            const retrievalSummary = retrieval.input_count
                ? ` Cheap retrieval kept ${retrieval.kept_count} of ${retrieval.input_count} references${retrieval.cache_hit ? ' using the cached index' : ' after building the index'}.`
                : '';
            setStatus('matcher-progress', `Top five shape candidates complete.${retrievalSummary}`, 'success');
        } catch (error) {
            setStatus('matcher-progress', error.message, 'error');
            document.getElementById('matcher-results').innerHTML =
                `<div class="empty-list">${escapeHtml(error.message)}</div>`;
        } finally {
            button.disabled = !(state.libraryReady && state.queryId);
            button.textContent = 'Run matcher';
        }
    }

    function metric(label, value) {
        return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`;
    }

    function renderDevResultCard(item) {
        const alignment = item.alignment || {};
        const joint = item.joint_alignment || {};
        const jointHistory = joint.history || [];
        const jointLast = jointHistory.length ? jointHistory[jointHistory.length - 1] : {};
        const warnings = item.warnings || [];
        return `<article class="matcher-result-card">
            <header>
                <span class="matcher-rank">Dev</span>
                <div>
                    <h4>${escapeHtml(item.citation_label || item.reference_id)}</h4>
                    <small>${escapeHtml(item.source_filename)}</small>
                </div>
                <strong class="matcher-score" title="Lower is better">Match cost ${number(item.overall_score)}</strong>
            </header>
            <div class="matcher-result-images">
                <figure>
                    <figcaption>Cleaned, centred reference used by matcher</figcaption>
                    <img src="${escapeHtml(item.reference_mask_url)}" alt="Cleaned canonical reference mask">
                </figure>
                <figure>
                    <figcaption>Aligned diagnostic</figcaption>
                    <img src="${escapeHtml(item.diagnostic_url)}" alt="Query and reference diagnostic overlay">
                </figure>
            </div>
            <dl class="matcher-metrics">
                ${metric('Ordered FGW cost', number(item.fgw_cost))}
                ${metric('POT FGW before order projection', number(item.unprojected_fgw_cost))}
                ${metric('Structural GW cost', number(item.structural_gw_cost))}
                ${metric('RTC feature cost', number(item.rtc_feature_cost))}
                ${metric('Symmetric local salience', number(item.salience_penalty))}
                ${metric('Reference to query salience', number(item.salience?.reference_to_query))}
                ${metric('Query to reference salience', number(item.salience?.query_to_reference))}
                ${metric('Salient alignment RMS', number(item.salience?.salient_alignment_rms))}
                ${metric('Two-wall ribbon cost', number(item.ribbon_cost ?? item.three_curve_cost))}
                ${metric('Seam-invariant rim-region cost', number(item.rim_region_cost))}
                ${metric('FGW score contribution', number(item.score_components?.fgw))}
                ${metric('Ribbon score contribution', number(item.score_components?.ribbon))}
                ${metric('Salience score contribution', number(item.score_components?.salience))}
                ${metric('Alignment-tail contribution', number(item.score_components?.alignment_tail))}
                ${metric('Rim-region contribution', number(item.score_components?.rim_region))}
                ${metric('Transform-reliability contribution', number(item.score_components?.transform_reliability))}
                ${metric('Completeness contribution', number(item.score_components?.completeness))}
                ${metric('Transform reliability penalty', number(item.transform_reliability?.overall))}
                ${metric('Scale disagreement penalty', number(item.transform_reliability?.scale_disagreement_penalty))}
                ${metric('Hypothesis instability penalty', number(item.transform_reliability?.hypothesis_instability_penalty))}
                ${metric('Reference interval selected', `${number(100 * item.matched_reference_fraction, 1)}%`)}
                ${metric('Query mass matched (required)', `${number(100 * item.query_coverage, 1)}%`)}
                ${metric('Transport backward steps', item.transport?.backward_steps ?? '—')}
                ${metric('Shared wall correspondence', item.transport?.shared_across_walls ? 'yes' : 'no')}
                ${metric('RMS residual', number(alignment.rms))}
                ${metric('Wall A RMS', number(alignment.per_curve?.wall_a?.rms))}
                ${metric('Wall B RMS', number(alignment.per_curve?.wall_b?.rms))}
                ${metric('Thickness RMS', number(alignment.thickness_rms))}
                ${metric('Wall A interval', (item.wall_intervals?.wall_a || []).join('â€“'))}
                ${metric('Wall B interval', (item.wall_intervals?.wall_b || []).join('â€“'))}
                ${metric('Median residual', number(alignment.median))}
                ${metric('95th residual', number(alignment.p95))}
                ${metric('Chamfer', number(alignment.chamfer))}
                ${metric('Hausdorff 95', number(alignment.hausdorff95))}
                ${metric('Rotation', `${number(alignment.rotation_degrees, 2)}Â°`)}
                ${metric('Scale ratio', number(alignment.scale))}
                ${metric('Procrustes scale', number(alignment.procrustes_scale))}
                ${metric('Median wall/thickness scale', number(alignment.median_ratio_scale))}
                ${metric('Scale-method disagreement', number(alignment.scale_log_disagreement))}
                ${metric('Translation', (alignment.translation || []).map(value => number(value, 3)).join(', '))}
                ${metric('Iterations', alignment.iterations ?? 'â€”')}
                ${metric('Converged', alignment.converged ? 'yes' : 'no')}
                ${metric('Joint alignment iterations', joint.iterations ?? '—')}
                ${metric('Joint alignment converged', joint.converged == null ? 'not run' : (joint.converged ? 'yes' : 'no'))}
                ${metric('Best joint iteration', joint.best_iteration ?? '—')}
                ${metric('Joint objective', number(joint.objective))}
                ${metric('Last progress change (samples)', number(jointLast.progress_delta_samples, 2))}
                ${metric('Last rotation change', `${number(jointLast.rotation_delta_degrees, 3)}°`)}
                ${metric('Last scale change', number(jointLast.scale_log_delta, 5))}
                ${metric('Interval iterations', item.interval_iterations ?? 'â€”')}
                ${metric('Interval refinement converged', item.interval_converged == null ? 'not run' : (item.interval_converged ? 'yes' : 'no'))}
                ${metric('Orientation stability', number(item.orientation_stability ?? item.initialization_stability))}
                ${metric('Orientation score spread', number(item.orientation_score_spread ?? item.initialization_score_spread))}
                ${metric('Traversal reversed', item.reverse_traversal ? 'yes' : 'no')}
                ${metric('Wall labels swapped', item.wall_swap ? 'yes' : 'no')}
                ${metric('Selected rim-seam offset', `${number(100 * (item.rim_seam?.query_offset_fraction || 0), 1)}%`)}
                ${metric('Selected reference rim split', item.reference_rim_split?.label || 'B')}
                ${metric('Reference split offset', `${number(100 * (item.reference_rim_split?.offset_fraction || 0), 1)}%`)}
                ${metric('Cheap retrieval rank', item.retrieval?.rank ?? '—')}
                ${metric('Outline retrieval rank', item.retrieval?.outline_rank ?? '—')}
                ${metric('Outline retrieval score', number(item.retrieval?.outline_score))}
                ${metric('Ribbon retrieval rank', item.retrieval?.ribbon_rank ?? '—')}
                ${metric('Ribbon retrieval score', number(item.retrieval?.ribbon_score))}
                ${metric('Retrieved by', (item.retrieval?.selected_by || []).join(', ') || '—')}
                ${metric('Gold point used as hard anchor', item.rim_seam?.gold_point_is_hard_anchor ? 'yes' : 'no')}
                ${metric('Adaptive rim expansion used', item.rim_seam?.adaptive_expansion_used ? 'yes' : 'no')}
                ${metric('Rim search still boundary-saturated', item.rim_seam?.boundary_saturated ? 'yes' : 'no')}
                ${metric('Exact hypotheses tested', item.hypothesis_search?.evaluated ?? '—')}
                ${metric('Distinct hypotheses', item.hypothesis_search?.distinct ?? '—')}
                ${metric('Selected hypothesis', item.hypothesis_search?.selected_strategy ?? '—')}
            </dl>
            ${warnings.length ? `<details class="matcher-warnings"><summary>${warnings.length} diagnostic warning${warnings.length === 1 ? '' : 's'}</summary><p>${escapeHtml(warnings.join(', '))}</p></details>` : ''}
        </article>`;
    }

    function renderResults(payload) {
        const container = document.getElementById('matcher-results');
        const margin = document.getElementById('matcher-confidence-margin');
        const results = payload.results || [];
        if (margin) {
            margin.textContent = payload.confidence_margin == null
                ? 'No first/second-place margin'
                : `First–second margin: ${number(payload.confidence_margin)}`;
        }
        if (!results.length) {
            container.innerHTML = '<div class="empty-list">No valid reference matches were produced.</div>';
            return;
        }
        container.innerHTML = results.map(item => {
            const alignment = item.alignment || {};
            const joint = item.joint_alignment || {};
            const jointHistory = joint.history || [];
            const jointLast = jointHistory.length ? jointHistory[jointHistory.length - 1] : {};
            const warnings = item.warnings || [];
            return `<article class="matcher-result-card">
                <header>
                    <span class="matcher-rank">#${Number(item.rank)}</span>
                    <div>
                        <h4>${escapeHtml(item.citation_label || item.reference_id)}</h4>
                        <small>${escapeHtml(item.source_filename)}</small>
                    </div>
                    <strong class="matcher-score" title="Lower is better">Match cost ${number(item.overall_score)}</strong>
                </header>
                <div class="matcher-result-images">
                    <figure>
                        <figcaption>Cleaned, centred reference used by matcher</figcaption>
                        <img src="${escapeHtml(item.reference_mask_url)}" alt="Cleaned canonical reference mask">
                    </figure>
                    <figure>
                        <figcaption>Aligned diagnostic</figcaption>
                        <img src="${escapeHtml(item.diagnostic_url)}" alt="Query and reference diagnostic overlay">
                    </figure>
                </div>
                <dl class="matcher-metrics">
                    ${metric('Ordered FGW cost', number(item.fgw_cost))}
                    ${metric('POT FGW before order projection', number(item.unprojected_fgw_cost))}
                    ${metric('Structural GW cost', number(item.structural_gw_cost))}
                    ${metric('RTC feature cost', number(item.rtc_feature_cost))}
                    ${metric('Symmetric local salience', number(item.salience_penalty))}
                    ${metric('Reference to query salience', number(item.salience?.reference_to_query))}
                    ${metric('Query to reference salience', number(item.salience?.query_to_reference))}
                    ${metric('Salient alignment RMS', number(item.salience?.salient_alignment_rms))}
                    ${metric('Two-wall ribbon cost', number(item.ribbon_cost ?? item.three_curve_cost))}
                    ${metric('Seam-invariant rim-region cost', number(item.rim_region_cost))}
                    ${metric('FGW score contribution', number(item.score_components?.fgw))}
                    ${metric('Ribbon score contribution', number(item.score_components?.ribbon))}
                    ${metric('Salience score contribution', number(item.score_components?.salience))}
                    ${metric('Alignment-tail contribution', number(item.score_components?.alignment_tail))}
                    ${metric('Rim-region contribution', number(item.score_components?.rim_region))}
                    ${metric('Transform-reliability contribution', number(item.score_components?.transform_reliability))}
                    ${metric('Completeness contribution', number(item.score_components?.completeness))}
                    ${metric('Transform reliability penalty', number(item.transform_reliability?.overall))}
                    ${metric('Scale disagreement penalty', number(item.transform_reliability?.scale_disagreement_penalty))}
                    ${metric('Hypothesis instability penalty', number(item.transform_reliability?.hypothesis_instability_penalty))}
                    ${metric('Reference interval selected', `${number(100 * item.matched_reference_fraction, 1)}%`)}
                    ${metric('Query mass matched (required)', `${number(100 * item.query_coverage, 1)}%`)}
                    ${metric('Transport backward steps', item.transport?.backward_steps ?? '—')}
                    ${metric('Shared wall correspondence', item.transport?.shared_across_walls ? 'yes' : 'no')}
                    ${metric('RMS residual', number(alignment.rms))}
                    ${metric('Wall A RMS', number(alignment.per_curve?.wall_a?.rms))}
                    ${metric('Wall B RMS', number(alignment.per_curve?.wall_b?.rms))}
                    ${metric('Thickness RMS', number(alignment.thickness_rms))}
                    ${metric('Wall A interval', (item.wall_intervals?.wall_a || []).join('–'))}
                    ${metric('Wall B interval', (item.wall_intervals?.wall_b || []).join('–'))}
                    ${metric('Median residual', number(alignment.median))}
                    ${metric('95th residual', number(alignment.p95))}
                    ${metric('Chamfer', number(alignment.chamfer))}
                    ${metric('Hausdorff 95', number(alignment.hausdorff95))}
                    ${metric('Rotation', `${number(alignment.rotation_degrees, 2)}°`)}
                    ${metric('Scale ratio', number(alignment.scale))}
                    ${metric('Procrustes scale', number(alignment.procrustes_scale))}
                    ${metric('Median wall/thickness scale', number(alignment.median_ratio_scale))}
                    ${metric('Scale-method disagreement', number(alignment.scale_log_disagreement))}
                    ${metric('Translation', (alignment.translation || []).map(value => number(value, 3)).join(', '))}
                    ${metric('Iterations', alignment.iterations ?? '—')}
                    ${metric('Converged', alignment.converged ? 'yes' : 'no')}
                    ${metric('Joint alignment iterations', joint.iterations ?? '—')}
                    ${metric('Joint alignment converged', joint.converged == null ? 'not run' : (joint.converged ? 'yes' : 'no'))}
                    ${metric('Best joint iteration', joint.best_iteration ?? '—')}
                    ${metric('Joint objective', number(joint.objective))}
                    ${metric('Last progress change (samples)', number(jointLast.progress_delta_samples, 2))}
                    ${metric('Last rotation change', `${number(jointLast.rotation_delta_degrees, 3)}°`)}
                    ${metric('Last scale change', number(jointLast.scale_log_delta, 5))}
                    ${metric('Interval iterations', item.interval_iterations ?? '—')}
                    ${metric('Interval refinement converged', item.interval_converged == null ? 'not run' : (item.interval_converged ? 'yes' : 'no'))}
                    ${metric('Orientation stability', number(item.orientation_stability ?? item.initialization_stability))}
                    ${metric('Orientation score spread', number(item.orientation_score_spread ?? item.initialization_score_spread))}
                    ${metric('Traversal reversed', item.reverse_traversal ? 'yes' : 'no')}
                    ${metric('Wall labels swapped', item.wall_swap ? 'yes' : 'no')}
                    ${metric('Selected rim-seam offset', `${number(100 * (item.rim_seam?.query_offset_fraction || 0), 1)}%`)}
                    ${metric('Selected reference rim split', item.reference_rim_split?.label || 'B')}
                    ${metric('Reference split offset', `${number(100 * (item.reference_rim_split?.offset_fraction || 0), 1)}%`)}
                    ${metric('Cheap retrieval rank', item.retrieval?.rank ?? '—')}
                    ${metric('Outline retrieval rank', item.retrieval?.outline_rank ?? '—')}
                    ${metric('Outline retrieval score', number(item.retrieval?.outline_score))}
                    ${metric('Ribbon retrieval rank', item.retrieval?.ribbon_rank ?? '—')}
                    ${metric('Ribbon retrieval score', number(item.retrieval?.ribbon_score))}
                    ${metric('Retrieved by', (item.retrieval?.selected_by || []).join(', ') || '—')}
                    ${metric('Gold point used as hard anchor', item.rim_seam?.gold_point_is_hard_anchor ? 'yes' : 'no')}
                    ${metric('Adaptive rim expansion used', item.rim_seam?.adaptive_expansion_used ? 'yes' : 'no')}
                    ${metric('Rim search still boundary-saturated', item.rim_seam?.boundary_saturated ? 'yes' : 'no')}
                    ${metric('Exact hypotheses tested', item.hypothesis_search?.evaluated ?? '—')}
                    ${metric('Distinct hypotheses', item.hypothesis_search?.distinct ?? '—')}
                    ${metric('Selected hypothesis', item.hypothesis_search?.selected_strategy ?? '—')}
                </dl>
                ${warnings.length ? `<details class="matcher-warnings"><summary>${warnings.length} diagnostic warning${warnings.length === 1 ? '' : 's'}</summary><p>${escapeHtml(warnings.join(', '))}</p></details>` : ''}
            </article>`;
        }).join('');
    }

    async function scoreForcedReference(event) {
        event.preventDefault();
        const pid = projectId();
        const figure = document.getElementById('matcher-forced-figure')?.value?.trim() || '';
        const item = document.getElementById('matcher-forced-item')?.value?.trim() || '';
        if (!pid || !state.runId) {
            setStatus('matcher-forced-status', 'Run the matcher once before scoring a specific sherd.', 'error');
            return;
        }
        if (!figure || !item) {
            setStatus('matcher-forced-status', 'Enter both Figure and Item.', 'error');
            return;
        }
        const button = document.getElementById('matcher-forced-score-btn');
        try {
            button.disabled = true;
            button.textContent = 'Scoringâ€¦';
            document.getElementById('matcher-forced-results').innerHTML = '';
            setStatus('matcher-forced-status', `Scoring Figure ${figure} Item ${item} against this queryâ€¦`);
            const response = await fetch(`/api/projects/${pid}/matcher/runs/${state.runId}/reference-score`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ figure, item }),
            });
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Specific sherd score failed');
            state.forcedResult = data.result || null;
            const results = data.result?.results || [];
            document.getElementById('matcher-forced-results').innerHTML = results.length
                ? results.map(renderDevResultCard).join('')
                : '<div class="empty-list">No matching reference was returned for that Figure and Item.</div>';
            setStatus('matcher-forced-status', `Specific sherd score complete for Figure ${figure} Item ${item}.`, 'success');
        } catch (error) {
            setStatus('matcher-forced-status', error.message, 'error');
            document.getElementById('matcher-forced-results').innerHTML =
                `<div class="empty-list">${escapeHtml(error.message)}</div>`;
        } finally {
            button.disabled = !state.runId;
            button.textContent = 'Score this sherd';
        }
    }

    function setEvaluationReady(ready) {
        const button = document.getElementById('matcher-evaluation-export-btn');
        if (button) button.disabled = !(ready && state.runId && state.result);
        if (!ready) {
            const link = document.getElementById('matcher-evaluation-download');
            if (link) {
                link.hidden = true;
                link.removeAttribute('href');
            }
            setStatus('matcher-evaluation-status', '');
        }
    }

    async function exportEvaluation() {
        const pid = projectId();
        if (!pid || !state.runId || !state.result) {
            setStatus('matcher-evaluation-status', 'Run the matcher before recording this evaluation.', 'error');
            return;
        }
        const button = document.getElementById('matcher-evaluation-export-btn');
        const link = document.getElementById('matcher-evaluation-download');
        try {
            button.disabled = true;
            button.textContent = 'Adding…';
            setStatus('matcher-evaluation-status', 'Building the project evaluation workbook…');
            const response = await fetch(
                `/api/projects/${pid}/matcher/runs/${state.runId}/evaluation`,
                { method: 'POST' },
            );
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Could not build evaluation workbook');
            if (link) {
                link.href = `${data.download_url}?v=${Date.now()}`;
                link.hidden = false;
            }
            const searched = data.forced_included
                ? ' The searched sherd was included.'
                : ' No searched sherd was included.';
            const action = data.already_present ? 'updated' : 'added';
            setStatus(
                'matcher-evaluation-status',
                `Evaluation ${action}. Workbook now contains ${data.query_count} quer${data.query_count === 1 ? 'y' : 'ies'} and ${data.result_count} result rows.${searched}`,
                'success',
            );
        } catch (error) {
            setStatus('matcher-evaluation-status', error.message, 'error');
        } finally {
            button.disabled = !(state.runId && state.result);
            button.textContent = 'Add to evaluation workbook';
        }
    }

    function handleFilePreview(event) {
        const file = event.target.files?.[0];
        const image = document.getElementById('matcher-upload-preview');
        state.queryId = null;
        state.runId = null;
        state.result = null;
        state.forcedResult = null;
        setEvaluationReady(false);
        document.getElementById('matcher-run-btn').disabled = true;
        document.getElementById('matcher-forced-score-btn').disabled = true;
        document.getElementById('matcher-forced-results').innerHTML = '';
        document.getElementById('matcher-contour-preview').removeAttribute('src');
        if (state.previewObjectUrl) URL.revokeObjectURL(state.previewObjectUrl);
        if (!file) {
            image.removeAttribute('src');
            state.annotationImage = null;
            state.curves = { exterior: [], interior: [], fracture: [], rim_point: [] };
            state.masterBoundary = null;
            resetZoom();
            updateCurveStatus();
            redrawAnnotation();
            return;
        }
        state.previewObjectUrl = URL.createObjectURL(file);
        image.src = state.previewObjectUrl;
        const annotationImage = new Image();
        annotationImage.onload = () => {
            state.annotationImage = annotationImage;
            state.curves = { exterior: [], interior: [], fracture: [], rim_point: [] };
            state.masterBoundary = null;
            resetZoom();
            updateCurveStatus();
            redrawAnnotation();
        };
        annotationImage.src = state.previewObjectUrl;
    }

    function initializeMatcherTab() {
        document.getElementById('matcher-query-file')?.addEventListener('change', handleFilePreview);
        document.getElementById('matcher-preprocess-btn')?.addEventListener('click', prepareQuery);
        document.getElementById('matcher-auto-curves')?.addEventListener('click', autoTraceWalls);
        document.getElementById('matcher-run-btn')?.addEventListener('click', runMatcher);
        document.getElementById('matcher-forced-form')?.addEventListener('submit', scoreForcedReference);
        document.getElementById('matcher-evaluation-export-btn')?.addEventListener('click', exportEvaluation);
        document.querySelectorAll('.matcher-curve-tool').forEach(button => {
            button.addEventListener('click', () => selectCurve(button.dataset.curve));
        });
        document.getElementById('matcher-undo-curve')?.addEventListener('click', () => {
            state.curves[state.activeCurve] = [];
            if (state.activeCurve === 'fracture' ||
                state.activeCurve === 'exterior' ||
                state.activeCurve === 'interior' ||
                state.activeCurve === 'rim_point') {
                state.masterBoundary = null;
            }
            state.queryId = null;
            state.runId = null;
            state.result = null;
            state.forcedResult = null;
            setEvaluationReady(false);
            document.getElementById('matcher-forced-score-btn').disabled = true;
            updateCurveStatus();
            redrawAnnotation();
        });
        document.getElementById('matcher-clear-curves')?.addEventListener('click', () => {
            state.curves = { exterior: [], interior: [], fracture: [], rim_point: [] };
            state.masterBoundary = null;
            state.queryId = null;
            state.runId = null;
            state.result = null;
            state.forcedResult = null;
            setEvaluationReady(false);
            document.getElementById('matcher-forced-score-btn').disabled = true;
            updateCurveStatus();
            redrawAnnotation();
        });
        document.getElementById('matcher-zoom-in')?.addEventListener('click', () => {
            setZoom(state.zoom * 1.25, document.getElementById('matcher-annotation-canvas').width / 2,
                document.getElementById('matcher-annotation-canvas').height / 2);
        });
        document.getElementById('matcher-zoom-out')?.addEventListener('click', () => {
            setZoom(state.zoom / 1.25, document.getElementById('matcher-annotation-canvas').width / 2,
                document.getElementById('matcher-annotation-canvas').height / 2);
        });
        document.getElementById('matcher-zoom-reset')?.addEventListener('click', resetZoom);
        document.getElementById('matcher-pan-tool')?.addEventListener('click', event => {
            state.mode = 'pan';
            event.currentTarget.classList.add('active');
            document.querySelectorAll('.matcher-curve-tool').forEach(button => button.classList.remove('active'));
        });
        const canvas = document.getElementById('matcher-annotation-canvas');
        canvas?.addEventListener('pointerdown', beginCurve);
        canvas?.addEventListener('pointermove', extendCurve);
        canvas?.addEventListener('pointerup', finishCurve);
        canvas?.addEventListener('pointercancel', finishCurve);
        canvas?.addEventListener('wheel', wheelZoom, { passive: false });
        window.addEventListener('resize', redrawAnnotation);
        redrawAnnotation();
    }

    document.addEventListener('DOMContentLoaded', initializeMatcherTab);
    window.loadMatcherTab = loadMatcherTab;
})();
