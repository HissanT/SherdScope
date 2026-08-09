"""Local-only expert scoring application for real-sherd matcher results."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageOps

from catalog.contours import _atomic_json
from catalog.real_sherd_evaluation import load_evaluation, read_json


RUBRIC = {
    0: "No match",
    1: "Weak/family-only",
    2: "Plausible parent",
    3: "Near-exact",
}


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Real-sherd expert scoring</title>
<style>
*{box-sizing:border-box}body{margin:0;font:15px system-ui;background:#eef2f6;color:#172033}header{height:62px;background:#fff;border-bottom:1px solid #cbd5e1;padding:10px 16px;display:flex;align-items:center;justify-content:space-between}.rubric{font-size:13px;color:#475569}.layout{height:calc(100vh - 62px);display:grid;grid-template-columns:210px 420px 1fr;gap:10px;padding:10px}.panel{background:#fff;border:1px solid #cbd5e1;border-radius:9px;overflow:auto;padding:10px}.queue button{display:block;width:100%;padding:7px;text-align:left;border:0;border-bottom:1px solid #e2e8f0;background:#fff}.queue button.active{background:#dbeafe}.queue button.done{color:#166534}.query-images{display:grid;grid-template-rows:1fr 1fr;gap:8px;height:65vh}.query-images img{width:100%;height:100%;object-fit:contain;background:#111;border-radius:6px}.candidate{border:2px solid #d5dde8;border-radius:9px;margin-bottom:9px;padding:9px;display:grid;grid-template-columns:170px 1fr;gap:10px}.candidate.selected{border-color:#2563eb;background:#eff6ff}.candidate img{width:170px;height:170px;object-fit:contain;background:#111}.score button{width:38px;height:34px;margin:2px;border:1px solid #94a3b8;border-radius:6px;background:#fff;font-weight:750}.score button.chosen{background:#1d4ed8;color:#fff}textarea{width:100%;min-height:54px;margin-top:6px}label{display:block;margin:8px 0}.toolbar button{padding:8px 10px}.status{font-weight:700}.muted{color:#64748b}.cost{font-variant-numeric:tabular-nums}
</style></head><body><header><div><strong>68 real-sherd expert scoring</strong> <span id="progress"></span><div class="rubric">0 no match · 1 weak/family-only · 2 plausible parent · 3 near-exact</div></div><div class="toolbar"><button id="first">First unscored</button> <span class="status" id="status">Loading…</span></div></header>
<div class="layout"><aside class="panel queue" id="queue"></aside><section class="panel"><h2 id="title"></h2><div class="query-images"><img id="photo" alt="Real sherd photograph"><img id="outline" alt="Saved query outline"></div><label><input type="checkbox" id="noMatch"> No acceptable match in the shown candidates</label><label>Query note<textarea id="queryNote"></textarea></label><div class="muted">Keyboard: 0–3 score selected candidate · J/K move · N next query</div></section><main class="panel" id="candidates"></main></div>
<script>
let queue=[],index=0,selected=0,saveTimer=null,pendingIndex=null;
const $=s=>document.querySelector(s);
function complete(q){return q.candidates.length>0&&q.candidates.every(c=>[0,1,2,3].includes(c.score))}
function progress(){const done=queue.filter(complete).length;$('#progress').textContent=`${done}/${queue.length} complete`;return done}
function renderQueue(){progress();$('#queue').innerHTML='';queue.forEach((q,i)=>{const b=document.createElement('button');b.textContent=`Q${q.number} ${complete(q)?'✓':'—'} ${q.filename}`;b.className=(i===index?'active ':'')+(complete(q)?'done':'');b.onclick=()=>load(i);$('#queue').appendChild(b)})}
function scoreButtons(candidate,ci){return [0,1,2,3].map(s=>`<button data-score="${s}" class="${candidate.score===s?'chosen':''}">${s}</button>`).join('')}
function renderCandidates(){const q=queue[index];$('#candidates').innerHTML='';q.candidates.forEach((c,ci)=>{const d=document.createElement('div');d.className='candidate '+(ci===selected?'selected':'');d.innerHTML=`<img src="${c.diagnostic_url}" alt="Aligned diagnostic"><div><strong>Rank ${c.rank}: ${c.citation}</strong><div class="cost">Match cost: ${Number(c.cost).toFixed(6)}</div><div class="score">${scoreButtons(c,ci)}</div><label>Candidate note<textarea>${c.note||''}</textarea></label></div>`;d.onclick=e=>{if(!e.target.closest('button,textarea')){selected=ci;localStorage.setItem('realSherdCandidate',selected);renderCandidates()}};d.querySelectorAll('button').forEach(b=>b.onclick=()=>setScore(ci,Number(b.dataset.score)));d.querySelector('textarea').oninput=e=>{c.note=e.target.value;scheduleSave()};$('#candidates').appendChild(d)})}
function load(i){if(pendingIndex!==null)saveIndex(pendingIndex);index=Math.max(0,Math.min(queue.length-1,i));selected=Math.max(0,Math.min(queue[index].candidates.length-1,selected));localStorage.setItem('realSherdQuery',index);const q=queue[index];$('#title').textContent=`Query ${q.number}: ${q.filename}`;$('#photo').src=q.photo_url;$('#outline').src=q.outline_url;$('#noMatch').checked=q.no_acceptable_match;$('#queryNote').value=q.note||'';renderQueue();renderCandidates()}
function setScore(ci,score){selected=ci;queue[index].candidates[ci].score=score;localStorage.setItem('realSherdCandidate',selected);renderCandidates();renderQueue();scheduleSave(true)}
function payload(i=index){const q=queue[i];return{number:q.number,no_acceptable_match:q.no_acceptable_match,note:q.note||'',candidates:q.candidates.map(c=>({rank:c.rank,reference_id:c.reference_id,citation:c.citation,score:c.score,note:c.note||''}))}}
async function saveIndex(i){clearTimeout(saveTimer);pendingIndex=null;$('#status').textContent='Saving…';const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload(i))});const d=await r.json();$('#status').textContent=d.success?'Saved':'Save failed';if(!d.success)alert(d.error)}
function scheduleSave(now=false){clearTimeout(saveTimer);pendingIndex=index;saveTimer=setTimeout(()=>saveIndex(pendingIndex),now?50:350)}
function firstUnscored(){const i=queue.findIndex(q=>!complete(q));load(i<0?0:i)}
$('#first').onclick=firstUnscored;$('#noMatch').onchange=()=>{queue[index].no_acceptable_match=$('#noMatch').checked;scheduleSave(true)};$('#queryNote').oninput=()=>{queue[index].note=$('#queryNote').value;scheduleSave()};
window.onkeydown=e=>{if(['TEXTAREA','INPUT'].includes(document.activeElement.tagName))return;const k=e.key.toLowerCase();if(['0','1','2','3'].includes(k)){e.preventDefault();setScore(selected,Number(k))}else if(k==='j'){selected=Math.min(queue[index].candidates.length-1,selected+1);localStorage.setItem('realSherdCandidate',selected);renderCandidates()}else if(k==='k'){selected=Math.max(0,selected-1);localStorage.setItem('realSherdCandidate',selected);renderCandidates()}else if(k==='n'){load(index+1)}};
fetch('/api/queue').then(r=>r.json()).then(d=>{queue=d.queries;const stored=Number(localStorage.getItem('realSherdQuery'));selected=Number(localStorage.getItem('realSherdCandidate'))||0;if(Number.isInteger(stored)&&stored>=0&&stored<queue.length)load(stored);else firstUnscored();$('#status').textContent='Ready'});
</script></body></html>"""


def _score_path(output: Path) -> Path:
    output = Path(output).resolve()
    return output if output.suffix.lower() == ".json" else output / "expert_scores.json"


def _photo_root(project: Path) -> Path:
    return project.parents[2] / "DL ArchProject" / "dataset_clean" / "train" / "images"


def _initial_state(
    records: list[dict[str, Any]], annotator: str, top_n: int
) -> dict[str, Any]:
    queries = {}
    for item in records:
        run = item["record"]["run"]
        candidates = []
        for row in run.get("results", [])[:top_n]:
            candidates.append(
                {
                    "rank": int(row.get("rank") or len(candidates) + 1),
                    "reference_id": str(row.get("reference_id") or ""),
                    "citation": str(row.get("citation_label") or row.get("reference_id") or ""),
                    "score": None,
                    "note": "",
                }
            )
        queries[str(item["number"])] = {
            "query_id": str(item["entry"].get("query_id") or ""),
            "run_id": str(item["entry"].get("run_id") or ""),
            "no_acceptable_match": False,
            "note": "",
            "candidates": candidates,
        }
    return {
        "schema_version": 1,
        "annotator": annotator,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "queries": queries,
    }


def _merge_state(current: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Retain valid annotations while refreshing immutable run/candidate identity."""
    existing = current.get("queries") or {}
    for number, query in expected["queries"].items():
        prior = existing.get(number) or {}
        if prior.get("query_id") != query["query_id"] or prior.get("run_id") != query["run_id"]:
            continue
        query["no_acceptable_match"] = bool(prior.get("no_acceptable_match"))
        query["note"] = str(prior.get("note") or "")
        by_rank = {int(row.get("rank") or 0): row for row in prior.get("candidates") or []}
        for candidate in query["candidates"]:
            old = by_rank.get(candidate["rank"]) or {}
            if old.get("reference_id") == candidate["reference_id"]:
                score = old.get("score")
                candidate["score"] = int(score) if score in {0, 1, 2, 3} else None
                candidate["note"] = str(old.get("note") or "")
    expected["updated_at"] = str(current.get("updated_at") or expected["updated_at"])
    return expected


def create_scoring_app(
    project: Path,
    run_dir: Path,
    output: Path,
    *,
    annotator: str,
    top_n: int = 5,
) -> Flask:
    project, run_dir = Path(project).resolve(), Path(run_dir).resolve()
    output_path = _score_path(output)
    manifest, records = load_evaluation(run_dir)
    if not 1 <= top_n <= 20:
        raise ValueError("top_n must be between 1 and 20")
    state = _initial_state(records, annotator, top_n)
    if output_path.is_file():
        state = _merge_state(read_json(output_path), state)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(output_path, state)

    query_set_path = project / "matcher" / "query_sets" / f"{manifest['query_set']}.json"
    query_set = read_json(query_set_path).get("queries") or {}
    by_number = {item["number"]: item for item in records}
    app = Flask(__name__)

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return HTML

    @app.get("/api/queue")
    def queue():
        output_rows = []
        for number in range(1, len(records) + 1):
            entry = query_set.get(str(number)) or {}
            scored = state["queries"][str(number)]
            run = by_number[number]["record"]["run"]
            candidates = []
            score_by_rank = {row["rank"]: row for row in scored["candidates"]}
            for row in run.get("results", [])[:top_n]:
                rank = int(row.get("rank") or len(candidates) + 1)
                saved = score_by_rank[rank]
                candidates.append(
                    {
                        **saved,
                        "cost": row.get("overall_score"),
                        "diagnostic_url": f"/asset/candidate/{number}/{rank}",
                    }
                )
            output_rows.append(
                {
                    "number": number,
                    "filename": str(entry.get("filename") or by_number[number]["record"].get("query", {}).get("source_filename") or ""),
                    "photo_url": f"/asset/query/{number}/photo",
                    "outline_url": f"/asset/query/{number}/outline",
                    "no_acceptable_match": scored["no_acceptable_match"],
                    "note": scored["note"],
                    "candidates": candidates,
                }
            )
        return jsonify(success=True, rubric=RUBRIC, queries=output_rows)

    @app.get("/asset/query/<int:number>/<kind>")
    def query_asset(number: int, kind: str):
        if number not in by_number or kind not in {"photo", "outline"}:
            return jsonify(success=False, error="Asset not found"), 404
        entry = query_set.get(str(number)) or {}
        if kind == "outline":
            query_id = str(by_number[number]["entry"].get("query_id") or "")
            path = project / "matcher" / "queries" / query_id / "preview.png"
            if not path.is_file():
                path = project / "matcher" / "queries" / query_id / "query.png"
            return send_file(path, max_age=0) if path.is_file() else (jsonify(success=False, error="Asset not found"), 404)
        path = _photo_root(project) / str(entry.get("filename") or "")
        if not path.is_file():
            return jsonify(success=False, error="Photograph not found"), 404
        if not entry.get("horizontal_flip"):
            return send_file(path, max_age=0)
        with Image.open(path) as source:
            image = ImageOps.mirror(source.convert("RGB"))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png", max_age=0)

    @app.get("/asset/candidate/<int:number>/<int:rank>")
    def candidate_asset(number: int, rank: int):
        if number not in by_number:
            return jsonify(success=False, error="Asset not found"), 404
        run_id = str(by_number[number]["entry"].get("run_id") or "")
        result = next(
            (row for row in by_number[number]["record"]["run"].get("results", []) if int(row.get("rank") or 0) == rank),
            None,
        )
        filename = str((result or {}).get("diagnostic") or "")
        path = project / "matcher" / "runs" / run_id / filename
        try:
            path.resolve().relative_to((project / "matcher" / "runs" / run_id).resolve())
        except ValueError:
            return jsonify(success=False, error="Invalid asset"), 400
        return send_file(path, max_age=0) if path.is_file() else (jsonify(success=False, error="Asset not found"), 404)

    @app.post("/api/save")
    def save():
        data = request.get_json(silent=True) or {}
        try:
            number = int(data.get("number"))
            target = state["queries"][str(number)]
            incoming = data.get("candidates") or []
            if len(incoming) != len(target["candidates"]):
                raise ValueError("Candidate count changed")
            for expected, value in zip(target["candidates"], incoming):
                if int(value.get("rank") or 0) != expected["rank"] or str(value.get("reference_id") or "") != expected["reference_id"]:
                    raise ValueError("Candidate identity changed")
                score = value.get("score")
                if score is not None and score not in {0, 1, 2, 3}:
                    raise ValueError("Scores must be 0, 1, 2, 3, or blank")
                expected["score"] = score
                expected["note"] = str(value.get("note") or "")
            target["no_acceptable_match"] = bool(data.get("no_acceptable_match"))
            target["note"] = str(data.get("note") or "")
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(output_path, state)
            return jsonify(success=True, updated_at=state["updated_at"])
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify(success=False, error=str(exc)), 400

    return app
