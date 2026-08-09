"""Photo-assisted fracture, rim, and optional wall-smoothing UI for real sherds."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file
import numpy as np
from PIL import Image, ImageOps

from catalog.contours import (
    _atomic_json,
    _smooth_curve,
    auto_query_wall_curves_from_fracture,
    matcher_root,
)
from catalog.matcher import preprocess_query


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Real-sherd outlines</title>
<style>
*{box-sizing:border-box}body{font:15px system-ui;margin:0;background:#eef2f6;color:#172033}header{height:58px;display:flex;justify-content:space-between;align-items:center;padding:10px 18px;background:#fff;border-bottom:1px solid #ccd5e0}main{height:calc(100vh - 58px);display:grid;grid-template-columns:205px 1fr;gap:10px;padding:10px}aside,section{background:#fff;border:1px solid #ccd5e0;border-radius:9px;padding:9px;overflow:auto}button{padding:8px 10px;margin:2px;border:1px solid #aab5c4;border-radius:6px;background:#fff;font-weight:650}button.active{color:#fff;background:#334155}.f.active{background:#9146b9}.g.active{background:#c8890c}.smooth{background:#e0f2fe;border-color:#38bdf8}.save{background:#1769c2;color:#fff}.row{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.views{height:calc(100vh - 145px);display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:7px}.pane{position:relative;min-width:0;overflow:hidden;border:1px solid #ccd5e0;background:#111}.pane-label{position:absolute;z-index:2;top:7px;left:7px;padding:4px 7px;border-radius:5px;background:#fffddc;font-weight:700}#photo{width:100%;height:100%;display:block;object-fit:contain}canvas{display:block;width:100%;height:100%;background:#f8fafc;touch-action:none;cursor:crosshair}#list button{display:block;width:100%;text-align:left}.hint{color:#596579}#status{font-weight:650}
</style></head><body><header><div><strong>Prepare 68 real-sherd queries</strong> <span id="count"></span></div><div class="hint">P preview · E exterior · I interior · B both · Enter save/next · Ctrl+Z undo</div></header><main><aside><div id="list"></div></aside><section><div class="row"><strong id="title"></strong><button class="f active" id="fracture">Purple fracture (F)</button><button class="g" id="gold">Gold rim point (G)</button><button id="undo">Undo (Ctrl+Z)</button><button id="clear">Clear</button><button id="preview">Preview outlines (P)</button><button class="smooth" id="smooth-exterior">Smooth exterior (E)</button><button class="smooth" id="smooth-interior">Smooth interior (I)</button><button class="smooth" id="smooth-both">Smooth both (B)</button><button class="save" id="save">Save and next (Enter)</button><span id="status"></span></div><div class="views"><div class="pane"><span class="pane-label">Real photograph</span><img id="photo"></div><div class="pane"><span class="pane-label">Mask and current outlines</span><canvas id="canvas"></canvas></div></div></section></main>
<script>
let queue=[],index=0,tool='fracture',drawing=false,fracture=[],gold=null,mask=new Image(),view=null,curves=null,previewReady=false,smoothed=false,curveHistory=[];
const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d'),photo=document.querySelector('#photo'),saveButton=document.querySelector('#save');
function line(points,color,width=4){if(!points||points.length<2||!view)return;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(view.x+p[0]*view.s,view.y+p[1]*view.s):ctx.moveTo(view.x+p[0]*view.s,view.y+p[1]*view.s));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.stroke()}
function redraw(){canvas.width=Math.max(500,canvas.clientWidth);canvas.height=Math.max(500,canvas.clientHeight);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,canvas.width,canvas.height);if(!mask.naturalWidth)return;const s=Math.min(canvas.width/mask.naturalWidth,canvas.height/mask.naturalHeight),w=mask.naturalWidth*s,h=mask.naturalHeight*s,x=(canvas.width-w)/2,y=(canvas.height-h)/2;view={s,x,y,w,h};ctx.drawImage(mask,x,y,w,h);if(curves){line(curves.exterior,'#00a7d6',5);line(curves.interior,'#16a34a',5)}line(fracture,'#9146b9',4);if(gold){ctx.beginPath();ctx.arc(x+gold[0]*s,y+gold[1]*s,8,0,Math.PI*2);ctx.fillStyle='#dc9819';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}}
function point(e){if(!view)return null;const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width,y=(e.clientY-r.top)*canvas.height/r.height;if(x<view.x||x>view.x+view.w||y<view.y||y>view.y+view.h)return null;return[(x-view.x)/view.s,(y-view.y)/view.s]}
function select(name){tool=name;document.querySelector('#fracture').classList.toggle('active',name==='fracture');document.querySelector('#gold').classList.toggle('active',name==='gold')}
function invalidate(){curves=null;previewReady=false;smoothed=false;curveHistory=[]}
function renderList(){document.querySelector('#count').textContent=`(${queue.filter(q=>q.saved).length}/${queue.length} saved)`;document.querySelector('#list').innerHTML=queue.map((q,i)=>`<button data-i="${i}" class="${i===index?'active':''}">${q.saved?'✓ ':''}${q.number}. ${q.filename}</button>`).join('');document.querySelectorAll('#list button').forEach(b=>b.onclick=()=>load(+b.dataset.i))}
function load(i){index=i;const q=queue[i];fracture=q.fracture||[];gold=q.rim_point||null;invalidate();document.querySelector('#title').textContent=`${q.number}: ${q.filename}`;document.querySelector('#status').textContent=q.saved?'Saved annotation loaded — Enter to preview it again':'';mask.onload=redraw;mask.src=`/mask/${q.number}?v=${Date.now()}`;photo.src=`/photo/${q.number}?v=${Date.now()}`;renderList()}
canvas.onpointerdown=e=>{const p=point(e);if(!p)return;invalidate();if(tool==='gold'){gold=p;redraw();return}drawing=true;fracture=[p];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drawing)return;const p=point(e);if(p){const old=fracture.at(-1);if(!old||Math.hypot(p[0]-old[0],p[1]-old[1])>1.5){fracture.push(p);redraw()}}};canvas.onpointerup=()=>drawing=false;
async function preview(){if(fracture.length<3||!gold){document.querySelector('#status').textContent='Draw fracture and place gold point first';return}document.querySelector('#status').textContent='Tracing current outlines…';try{const r=await fetch('/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({number:queue[index].number,fracture,rim_point:gold})}),d=await r.json();if(!d.success)throw Error(d.error);curves=d.curves;previewReady=true;smoothed=false;curveHistory=[];document.querySelector('#status').textContent='Cyan exterior, green interior. E/I/B smooth; Enter saves and advances.';redraw()}catch(e){invalidate();document.querySelector('#status').textContent=e.message}}
async function smooth(which){if(!previewReady){document.querySelector('#status').textContent='Press P to preview the outlines before smoothing.';return}const prior={curves:JSON.parse(JSON.stringify(curves)),smoothed};curveHistory.push(prior);document.querySelector('#status').textContent=`Smoothing ${which}…`;try{const r=await fetch('/smooth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({which,curves})}),d=await r.json();if(!d.success)throw Error(d.error);curves=d.curves;smoothed=true;document.querySelector('#status').textContent=`Smoothed ${which}. Ctrl+Z undoes this pass; Enter saves and advances.`;redraw()}catch(e){curveHistory.pop();document.querySelector('#status').textContent=e.message}}
function undo(){if(curveHistory.length){const prior=curveHistory.pop();curves=prior.curves;smoothed=prior.smoothed;previewReady=true;document.querySelector('#status').textContent='Last smoothing pass undone.';redraw();return}invalidate();if(tool==='gold')gold=null;else fracture=[];document.querySelector('#status').textContent='Annotation undone.';redraw()}
async function save(){if(!previewReady){document.querySelector('#status').textContent='Press P to preview the outlines before saving.';return}saveButton.disabled=true;document.querySelector('#status').textContent='Saving confirmed outlines…';try{const body={number:queue[index].number,fracture,rim_point:gold,...(smoothed?{curves_override:curves}:{})};const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!d.success)throw Error(d.error);queue[index]={...queue[index],saved:true,fracture:[...fracture],rim_point:[...gold]};renderList();if(index+1<queue.length)load(index+1);else document.querySelector('#status').textContent='All 68 reviewed. You can close this window.'}catch(e){document.querySelector('#status').textContent=e.message}finally{saveButton.disabled=false}}
document.querySelector('#fracture').onclick=()=>select('fracture');document.querySelector('#gold').onclick=()=>select('gold');document.querySelector('#undo').onclick=undo;document.querySelector('#clear').onclick=()=>{invalidate();fracture=[];gold=null;redraw()};document.querySelector('#preview').onclick=preview;document.querySelector('#smooth-exterior').onclick=()=>smooth('exterior');document.querySelector('#smooth-interior').onclick=()=>smooth('interior');document.querySelector('#smooth-both').onclick=()=>smooth('both');saveButton.onclick=save;window.onresize=redraw;window.onkeydown=e=>{const k=e.key.toLowerCase();if(k==='f')select('fracture');if(k==='g')select('gold');if(k==='p')preview();if(k==='e')smooth('exterior');if(k==='i')smooth('interior');if(k==='b')smooth('both');if(e.key==='Enter')save();if(e.ctrlKey&&k==='z'){e.preventDefault();undo()}};
fetch('/queue').then(r=>r.json()).then(d=>{queue=d.queries;const first=queue.findIndex(q=>!q.saved);load(first>=0?first:0)});
</script></body></html>"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _smooth_wall(points: Any) -> list[list[float]]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 8:
        raise ValueError("Preview a valid wall outline before smoothing")
    smoothed, _diagnostics = _smooth_curve(values, 3.0, 2.0)
    # The two endpoints encode the rim and fracture connections. Never move them.
    smoothed[0], smoothed[-1] = values[0], values[-1]
    return smoothed.tolist()


def create_real_sherd_annotation_app(
    project_path: Path,
    masks_path: Path,
    photos_path: Path,
    *,
    expected_count: int = 68,
    set_name: str = "real_sherds_68",
) -> Flask:
    project_path = Path(project_path).resolve()
    masks_path = Path(masks_path).resolve()
    photos_path = Path(photos_path).resolve()
    mask_files = sorted(masks_path.glob("*.png"), key=lambda path: path.name.lower())
    if len(mask_files) != expected_count:
        raise ValueError(f"Expected {expected_count} PNG masks, found {len(mask_files)}")
    masks = {number: path for number, path in enumerate(mask_files, start=1)}
    photo_by_stem = {
        path.stem.lower(): path
        for path in photos_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    photos = {number: photo_by_stem.get(path.stem.lower()) for number, path in masks.items()}
    missing = [masks[number].name for number in masks if photos[number] is None]
    if missing:
        raise ValueError(f"Real photograph is missing for {missing[0]}")
    set_path = matcher_root(project_path) / "query_sets" / f"{set_name}.json"

    def manifest() -> dict[str, Any]:
        return _read_json(set_path) or {
            "schema_version": 1,
            "name": set_name,
            "source_masks": str(masks_path),
            "source_photos": str(photos_path),
            "queries": {},
        }

    def prepared_mask(number: int) -> Image.Image:
        with Image.open(masks[number]) as source:
            value = ImageOps.invert(source.convert("L")).convert("RGBA")
        entry = (manifest().get("queries") or {}).get(str(number)) or {}
        return ImageOps.mirror(value) if entry.get("horizontal_flip") else value

    def trace(data: dict[str, Any]) -> tuple[int, list, list, dict[str, Any]]:
        number = int(data.get("number") or 0)
        fracture = data.get("fracture") or []
        rim_point = data.get("rim_point")
        if number not in masks or len(fracture) < 3 or not rim_point:
            raise ValueError("Draw fracture and place gold point first")
        curves = auto_query_wall_curves_from_fracture(
            prepared_mask(number), {"fracture": fracture, "rim_point": rim_point}
        )
        return number, fracture, rim_point, curves

    app = Flask(__name__)

    @app.get("/")
    def index():
        return HTML

    @app.get("/queue")
    def queue():
        saved = manifest().get("queries") or {}
        return jsonify(queries=[{
            "number": number,
            "filename": masks[number].name,
            "saved": str(number) in saved,
            "fracture": (saved.get(str(number)) or {}).get("fracture") or [],
            "rim_point": (saved.get(str(number)) or {}).get("rim_point"),
        } for number in range(1, expected_count + 1)])

    @app.get("/mask/<int:number>")
    def mask(number: int):
        if number not in masks:
            return jsonify(error="Unknown mask"), 404
        payload = BytesIO()
        prepared_mask(number).save(payload, format="PNG")
        payload.seek(0)
        return send_file(payload, mimetype="image/png")

    @app.get("/photo/<int:number>")
    def photo(number: int):
        if number not in photos:
            return jsonify(error="Unknown photograph"), 404
        entry = (manifest().get("queries") or {}).get(str(number)) or {}
        if not entry.get("horizontal_flip"):
            return send_file(photos[number])
        with Image.open(photos[number]) as source:
            value = ImageOps.mirror(source.convert("RGB"))
        payload = BytesIO()
        value.save(payload, format="PNG")
        payload.seek(0)
        return send_file(payload, mimetype="image/png")

    @app.post("/preview")
    def preview():
        try:
            _number, _fracture, _rim, curves = trace(request.get_json(silent=True) or {})
            return jsonify(success=True, curves={
                "exterior": curves.get("exterior") or [],
                "interior": curves.get("interior") or [],
            })
        except Exception as exc:
            return jsonify(success=False, error=str(exc)), 400

    @app.post("/smooth")
    def smooth():
        data = request.get_json(silent=True) or {}
        which = str(data.get("which") or "")
        curves = dict(data.get("curves") or {})
        if which not in {"exterior", "interior", "both"}:
            return jsonify(success=False, error="Choose exterior, interior, or both"), 400
        try:
            if which in {"exterior", "both"}:
                curves["exterior"] = _smooth_wall(curves.get("exterior") or [])
            if which in {"interior", "both"}:
                curves["interior"] = _smooth_wall(curves.get("interior") or [])
            return jsonify(success=True, curves=curves)
        except Exception as exc:
            return jsonify(success=False, error=str(exc)), 400

    @app.post("/save")
    def save():
        data = request.get_json(silent=True) or {}
        try:
            number, fracture, rim_point, curves = trace(data)
            override = data.get("curves_override")
            if override:
                curves = {
                    "exterior": override.get("exterior") or [],
                    "interior": override.get("interior") or [],
                    "fracture": fracture,
                    "rim_point": rim_point,
                }
            result = preprocess_query(
                project_path,
                prepared_mask(number),
                original_filename=masks[number].name,
                metadata={
                    "query_id": f"Real sherd {number}",
                    "rim_diameter_cm": "",
                    "fabric": "",
                    "surface": "",
                    "notes": "",
                },
                manual_curves=curves,
            )
            value = manifest()
            value["queries"][str(number)] = {
                "number": number,
                "filename": masks[number].name,
                "query_id": result["query_id"],
                "fracture": fracture,
                "rim_point": rim_point,
                "smoothing_applied": bool(override),
            }
            _atomic_json(set_path, value)
            return jsonify(success=True, query_id=result["query_id"])
        except Exception as exc:
            return jsonify(success=False, error=str(exc)), 400

    return app
