"""Local, one-time metadata entry UI for the fixed Hesban query set."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file

from catalog.contours import _atomic_json, matcher_root
from catalog.matcher_query_set import HESBAN_TARGETS
from catalog.metadata_fusion import canonical_metadata


FIELDS = (
    ("vessel_type", "Vessel type", "text"),
    ("rim_diameter_cm", "Rim diameter (cm)", "number"),
    ("fabric_exterior", "Fabric colour - exterior (Munsell)", "text"),
    ("fabric_core", "Fabric colour - core (Munsell)", "text"),
    ("fabric_interior", "Fabric colour - interior (Munsell)", "text"),
    ("nonplastics_type", "Non-plastics type (L/P/S)", "text"),
    ("nonplastics_size", "Non-plastics size (1-7)", "text"),
    ("nonplastics_shape", "Non-plastics shape (A/SA/SR/R)", "text"),
    ("nonplastics_density", "Non-plastics density (VL/L/M/MH/H)", "text"),
    ("voids_type_size", "Voids type and size", "text"),
    ("voids_density", "Voids density (L/M/MH/H)", "text"),
    ("manufacture", "Manufacture", "text"),
    ("surface_exterior", "Exterior surface treatment", "text"),
    ("surface_exterior_color", "Exterior surface colour", "text"),
    ("surface_interior", "Interior surface treatment", "text"),
    ("surface_interior_color", "Interior surface colour", "text"),
    ("decor", "Decoration", "text"),
    ("fire", "Firing", "text"),
)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Hesban query metadata</title>
<style>
*{box-sizing:border-box}body{margin:0;font:15px system-ui;background:#eef2f6;color:#172033}header{height:58px;padding:12px 18px;background:#fff;border-bottom:1px solid #ccd5e0;display:flex;justify-content:space-between;align-items:center}main{height:calc(100vh - 58px);padding:12px;display:grid;grid-template-columns:minmax(480px,1.35fr) minmax(420px,1fr);gap:12px}.panel{background:#fff;border:1px solid #ccd5e0;border-radius:10px;padding:12px;overflow:auto}canvas{width:100%;height:calc(100vh - 150px);display:block;background:#f8fafc;border:1px solid #ccd5e0}.reference{background:#f7fafc;border:1px solid #b8c5d4;border-radius:8px;padding:10px;margin-bottom:14px}.reference h2{font-size:16px;margin:0 0 8px}.reference-record+ .reference-record{border-top:1px solid #ccd5e0;margin-top:9px;padding-top:9px}.reference-grid{display:grid;grid-template-columns:minmax(150px,.8fr) 1.2fr;gap:4px 10px}.reference-key{font-weight:650;color:#46556a}.reference-value{white-space:pre-wrap}.form{display:grid;grid-template-columns:1fr 1fr;gap:9px 12px}.field{display:flex;flex-direction:column;gap:4px}.field label{font-weight:650}.field input{width:100%;padding:8px;border:1px solid #aeb9c8;border-radius:6px}.wide{grid-column:1/-1}.buttons{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:12px;position:sticky;bottom:0;background:white;padding:10px 0}button{padding:9px 12px;border:1px solid #aab5c4;border-radius:7px;background:white;font-weight:650}.save{background:#1769c2;color:#fff}.hint{color:#5e6b7d}.ok{color:#15803d;font-weight:700}
</style></head><body><header><div><strong id="title">Hesban metadata</strong> <span id="count"></span></div><div class="hint">Blank means unknown and has no effect. Enter saves and moves forward.</div></header><main><section class="panel"><canvas id="canvas"></canvas></section><section class="panel"><div class="reference" id="reference"></div><div class="form" id="form"></div><div class="buttons"><button id="prev">Previous</button><button id="next">Next</button><button class="save" id="save">Save and next (Enter)</button><span id="status"></span></div></section></main>
<script>
let queue=[],index=0,img=new Image(),view=null;const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');
function esc(value){return String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}
function redraw(){canvas.width=Math.max(600,canvas.clientWidth);canvas.height=Math.max(500,canvas.clientHeight);ctx.clearRect(0,0,canvas.width,canvas.height);if(!img.naturalWidth)return;const s=Math.min(canvas.width/img.naturalWidth,canvas.height/img.naturalHeight),w=img.naturalWidth*s,h=img.naturalHeight*s,x=(canvas.width-w)/2,y=(canvas.height-h)/2;view={s,x,y};ctx.drawImage(img,x,y,w,h);const q=queue[index];if((q.fracture||[]).length>1){ctx.beginPath();q.fracture.forEach((p,i)=>i?ctx.lineTo(x+p[0]*s,y+p[1]*s):ctx.moveTo(x+p[0]*s,y+p[1]*s));ctx.strokeStyle='#8b3bb5';ctx.lineWidth=3;ctx.stroke()}if(q.rim_point){ctx.beginPath();ctx.arc(x+q.rim_point[0]*s,y+q.rim_point[1]*s,7,0,Math.PI*2);ctx.fillStyle='#e09a13';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}}
function reference(){const q=queue[index],records=q.reference_records||[],box=document.querySelector('#reference');const heading=`Stored reference metadata - Figure ${esc(q.target.figure)} Item ${esc(q.target.item)}`;if(!records.length){box.innerHTML=`<h2>${heading}</h2><div class="hint">No stored catalogue metadata was found for this Figure + Item.</div>`;return}box.innerHTML=`<h2>${heading}</h2>`+records.map((record,i)=>`<div class="reference-record"><div class="hint">${records.length>1?`Stored record ${i+1} - `:''}${esc(record.mask_file)}</div><div class="reference-grid">${record.values.map(v=>`<div class="reference-key">${esc(v.label)}</div><div class="reference-value">${esc(v.value)}</div>`).join('')}</div></div>`).join('')}
function form(){const q=queue[index],m=q.metadata||{};document.querySelector('#form').innerHTML=q.fields.map(f=>`<div class="field"><label for="${f.key}">${esc(f.label)}</label><input id="${f.key}" data-key="${f.key}" type="${f.type}" step="any" value="${esc(m[f.key])}"></div>`).join('')+`<div class="field wide"><label for="diameter_uncertainty_cm">Estimated diameter uncertainty (+/- cm)</label><input id="diameter_uncertainty_cm" type="number" step="0.1" min="0" value="${q.diameter_uncertainty_cm??1.5}"><span class="hint">1.5 cm is a cautious default for human measurement. It is uncertainty, not a hard tolerance.</span></div>`}
function load(i){index=(i+queue.length)%queue.length;const q=queue[index];document.querySelector('#title').textContent=`Query ${q.number}: ${q.filename}`;document.querySelector('#count').textContent=`(${queue.filter(x=>x.saved).length}/${queue.length} saved)`;document.querySelector('#status').textContent=q.saved?'Saved values loaded - edit or keep them':'';reference();form();img.onload=redraw;img.src=`/image/${q.number}?v=${Date.now()}`}
async function save(){const q=queue[index],metadata={};document.querySelectorAll('[data-key]').forEach(el=>{if(el.value.trim())metadata[el.dataset.key]=el.value.trim()});const uncertainty=parseFloat(document.querySelector('#diameter_uncertainty_cm').value);document.querySelector('#status').textContent='Saving...';const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({number:q.number,metadata,diameter_uncertainty_cm:Number.isFinite(uncertainty)?uncertainty:1.5})}),d=await r.json();if(!d.success){document.querySelector('#status').textContent=d.error;return}q.metadata=metadata;q.diameter_uncertainty_cm=d.diameter_uncertainty_cm;q.saved=true;document.querySelector('#status').textContent='Saved';load(index+1)}
document.querySelector('#save').onclick=save;document.querySelector('#prev').onclick=()=>load(index-1);document.querySelector('#next').onclick=()=>load(index+1);window.onresize=redraw;window.onkeydown=e=>{if(e.key==='Enter'&&e.target.tagName!=='BUTTON'){e.preventDefault();save()}};fetch('/queue').then(r=>r.json()).then(d=>{queue=d.queries;load(Math.max(0,queue.findIndex(q=>!q.saved)))})
</script></body></html>"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _citation_parts(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", str(value)))


def _stored_reference_metadata(project_path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    path = project_path / "cards" / "mask_info.csv"
    if not path.is_file():
        return {}
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    labels = {key: label for key, label, _input_type in FIELDS}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            figure_parts = _citation_parts(row.get("Figure"))
            item_parts = _citation_parts(row.get("No."))
            if not figure_parts or not item_parts:
                continue
            figure = ".".join(str(part) for part in figure_parts)
            item = str(item_parts[0])
            canonical = canonical_metadata(row)
            values = [
                {"key": key, "label": labels[key], "value": str(canonical[key]).strip()}
                for key, _label, _input_type in FIELDS
                if canonical.get(key) is not None and str(canonical[key]).strip()
            ]
            indexed.setdefault((figure, item), []).append({
                "mask_file": str(row.get("mask_file") or row.get("file") or ""),
                "values": values,
            })
    return indexed


def create_metadata_curator_app(
    project_path: Path,
    queries_path: Path,
    *,
    set_name: str = "hesban_30",
    expected_count: int = 30,
    targets: list[tuple[str, str]] | None = None,
) -> Flask:
    project_path = Path(project_path).resolve()
    queries_path = Path(queries_path).resolve()
    set_root = matcher_root(project_path) / "query_sets"
    query_set_path = set_root / f"{set_name}.json"
    metadata_path = set_root / f"{set_name}_metadata.json"
    query_set = _read_json(query_set_path)
    prepared = query_set.get("queries") or {}
    if sorted(int(key) for key in prepared) != list(range(1, expected_count + 1)):
        raise ValueError(f"Save fracture and gold points for all Query 1-{expected_count} first")
    targets = targets or list(HESBAN_TARGETS[:expected_count])
    if len(targets) != expected_count:
        raise ValueError(f"Expected {expected_count} target labels, found {len(targets)}")
    images = {
        number: queries_path / str(prepared[str(number)].get("filename") or "")
        for number in range(1, expected_count + 1)
    }
    missing = [str(path) for path in images.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Query image is missing: {missing[0]}")
    stored_reference = _stored_reference_metadata(project_path)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return HTML

    @app.get("/queue")
    def queue():
        saved = (_read_json(metadata_path).get("queries") or {})
        return jsonify(queries=[{
            "number": number,
            "filename": images[number].name,
            "query_id": prepared[str(number)].get("query_id"),
            "fracture": prepared[str(number)].get("fracture") or [],
            "rim_point": prepared[str(number)].get("rim_point"),
            "metadata": (saved.get(str(number)) or {}).get("metadata") or {},
            "diameter_uncertainty_cm": (saved.get(str(number)) or {}).get(
                "diameter_uncertainty_cm", 1.5
            ),
            "saved": str(number) in saved,
            "target": {
                "figure": targets[number - 1][0],
                "item": targets[number - 1][1],
            },
            "reference_records": stored_reference.get(targets[number - 1], []),
            "fields": [
                {"key": key, "label": label, "type": input_type}
                for key, label, input_type in FIELDS
            ],
        } for number in range(1, expected_count + 1)])

    @app.get("/image/<int:number>")
    def image(number: int):
        if number not in images:
            return jsonify(error="Unknown query"), 404
        return send_file(images[number], mimetype="image/png")

    @app.post("/save")
    def save():
        data = request.get_json(silent=True) or {}
        number = int(data.get("number") or 0)
        if number not in images:
            return jsonify(success=False, error="Unknown query"), 400
        allowed = {key for key, _label, _input_type in FIELDS}
        raw = data.get("metadata") or {}
        metadata = {
            str(key): str(value).strip()
            for key, value in raw.items()
            if key in allowed and str(value).strip()
        }
        try:
            uncertainty = max(0.0, float(data.get("diameter_uncertainty_cm", 1.5)))
        except (TypeError, ValueError):
            uncertainty = 1.5
        value = _read_json(metadata_path) or {
            "schema_version": 1,
            "name": f"{set_name}_metadata",
            "policy": "shape_plus_metadata_only; missing fields are neutral",
            "queries": {},
        }
        value["queries"][str(number)] = {
            "number": number,
            "filename": images[number].name,
            "query_id": prepared[str(number)].get("query_id"),
            "metadata": metadata,
            "diameter_uncertainty_cm": uncertainty,
        }
        _atomic_json(metadata_path, value)
        return jsonify(success=True, diameter_uncertainty_cm=uncertainty)

    return app
