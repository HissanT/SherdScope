"""Small local curator for preparing a fixed set of matcher queries once."""

from __future__ import annotations

from io import BytesIO
import re
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageOps

from catalog.contours import _atomic_json, auto_query_wall_curves_from_fracture, matcher_root
from catalog.matcher import preprocess_query


def _number(path: Path) -> int:
    match = re.search(r"query\s*0*(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 10**9


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Hesban query points</title>
<style>body{font:16px system-ui;margin:0;background:#eef2f6;color:#172033}header{display:flex;justify-content:space-between;align-items:center;padding:12px 20px;background:white;border-bottom:1px solid #ccd5e0}main{display:grid;grid-template-columns:220px 1fr;gap:14px;padding:14px;height:calc(100vh - 82px)}aside,section{background:white;border:1px solid #ccd5e0;border-radius:10px;padding:12px;overflow:auto}button{padding:9px 12px;margin:3px;border:1px solid #aab5c4;border-radius:7px;background:white;font-weight:650}button.active{color:white;background:#334155}.f.active{background:#9146b9}.g.active{background:#c8890c}.save{background:#1769c2;color:white}.done{color:#15803d;font-weight:700}.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}canvas{display:block;width:100%;height:calc(100vh - 205px);background:#f8fafc;border:1px solid #ccd5e0;touch-action:none;cursor:crosshair}#list button{display:block;width:100%;text-align:left}.hint{color:#596579}#status{font-weight:650}</style></head>
<body><header><div><strong>Prepare all __EXPECTED_COUNT__ __DISPLAY_NAME__</strong> <span id="count"></span></div><div class="hint">F fracture · G gold point · Ctrl+Z undo · __ENTER_HINT__</div></header><main><aside><div id="list"></div></aside><section><div class="row"><strong id="title"></strong><button class="f active" id="fracture">Purple fracture (F)</button><button class="g" id="gold">Gold rim point (G)</button><button id="undo">Undo</button><button id="clear">Clear</button><button class="save" id="save">__SAVE_LABEL__</button><span id="status"></span></div><canvas id="canvas"></canvas></section></main>
<script>
const previewBeforeSave=__PREVIEW_BEFORE_SAVE__;let queue=[],index=0,tool='fracture',drawing=false,fracture=[],gold=null,img=new Image(),view=null,previewCurves=null,previewReady=false;
const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');
function line(points,color,width=4){if(!points||points.length<2)return;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(view.x+p[0]*view.s,view.y+p[1]*view.s):ctx.moveTo(view.x+p[0]*view.s,view.y+p[1]*view.s));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.stroke()}
function redraw(){canvas.width=Math.max(600,canvas.clientWidth);canvas.height=Math.max(500,canvas.clientHeight);ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,canvas.width,canvas.height);if(!img.naturalWidth)return;const s=Math.min(canvas.width/img.naturalWidth,canvas.height/img.naturalHeight),w=img.naturalWidth*s,h=img.naturalHeight*s,x=(canvas.width-w)/2,y=(canvas.height-h)/2;view={s,x,y,w,h};ctx.drawImage(img,x,y,w,h);if(previewCurves){line(previewCurves.exterior,'#00a7d6',5);line(previewCurves.interior,'#16a34a',5)}line(fracture,'#9146b9',4);if(gold){ctx.beginPath();ctx.arc(x+gold[0]*s,y+gold[1]*s,8,0,Math.PI*2);ctx.fillStyle='#dc9819';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke()}}
function point(e){if(!view)return null;const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width,y=(e.clientY-r.top)*canvas.height/r.height;if(x<view.x||x>view.x+view.w||y<view.y||y>view.y+view.h)return null;return[(x-view.x)/view.s,(y-view.y)/view.s]}
function select(name){tool=name;document.querySelector('#fracture').classList.toggle('active',name==='fracture');document.querySelector('#gold').classList.toggle('active',name==='gold')}
function renderList(){document.querySelector('#count').textContent=`(${queue.filter(q=>q.saved).length}/${queue.length} saved)`;document.querySelector('#list').innerHTML=queue.map((q,i)=>`<button data-i="${i}" class="${i===index?'active':''}">${q.saved?'✓ ':''}Query ${q.number}</button>`).join('');document.querySelectorAll('#list button').forEach(b=>b.onclick=()=>load(+b.dataset.i))}
function invalidate(){previewCurves=null;previewReady=false;document.querySelector('#save').textContent=previewBeforeSave?'Preview outlines (Enter)':'Save and next (Enter)'}
function load(i){index=i;fracture=queue[i].fracture||[];gold=queue[i].rim_point||null;invalidate();document.querySelector('#title').textContent=`Query ${queue[i].number}: ${queue[i].filename}`;document.querySelector('#status').textContent=queue[i].saved?'Saved — preview again before confirming':queue[i].seeded?'Previous fracture and gold point loaded — preview before confirming':'';img.onload=redraw;img.src=`/image/${queue[i].number}?v=${Date.now()}`;renderList()}
canvas.onpointerdown=e=>{const p=point(e);if(!p)return;invalidate();if(tool==='gold'){gold=p;redraw();return}drawing=true;fracture=[p];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drawing)return;const p=point(e);if(p){const old=fracture.at(-1);if(!old||Math.hypot(p[0]-old[0],p[1]-old[1])>1.5){fracture.push(p);redraw()}}};canvas.onpointerup=()=>drawing=false;
async function preview(){if(fracture.length<3||!gold){document.querySelector('#status').textContent='Draw fracture and place gold point first';return}const b=document.querySelector('#save');b.disabled=true;document.querySelector('#status').textContent='Tracing the actual wall outlines…';try{const r=await fetch('/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({number:queue[index].number,fracture,rim_point:gold})}),d=await r.json();if(!d.success)throw Error(d.error);previewCurves=d.curves;previewReady=true;b.textContent='Confirm save and next (Enter)';document.querySelector('#status').textContent='Preview: cyan exterior, green interior. Enter again to save.';redraw()}catch(e){invalidate();document.querySelector('#status').textContent=e.message}finally{b.disabled=false}}
async function save(){if(fracture.length<3||!gold){document.querySelector('#status').textContent='Draw fracture and place gold point first';return}if(previewBeforeSave&&!previewReady){await preview();return}const b=document.querySelector('#save');b.disabled=true;document.querySelector('#status').textContent='Saving confirmed outlines…';try{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({number:queue[index].number,fracture,rim_point:gold})}),d=await r.json();if(!d.success)throw Error(d.error);queue[index]={...queue[index],saved:true,fracture:[...fracture],rim_point:[...gold]};renderList();document.querySelector('#status').textContent='Saved';const next=queue.findIndex((q,j)=>j>index&&!q.saved);if(next>=0)load(next);else if(queue.every(q=>q.saved))document.querySelector('#status').textContent='All queries saved. You can close this window.'}catch(e){document.querySelector('#status').textContent=e.message}finally{b.disabled=false}}
document.querySelector('#fracture').onclick=()=>select('fracture');document.querySelector('#gold').onclick=()=>select('gold');document.querySelector('#undo').onclick=()=>{invalidate();if(tool==='gold')gold=null;else fracture=[];redraw()};document.querySelector('#clear').onclick=()=>{invalidate();fracture=[];gold=null;redraw()};document.querySelector('#save').onclick=save;window.onresize=redraw;window.onkeydown=e=>{if(e.key.toLowerCase()==='f')select('fracture');if(e.key.toLowerCase()==='g')select('gold');if(e.key==='Enter')save();if(e.ctrlKey&&e.key.toLowerCase()==='z'){e.preventDefault();invalidate();if(tool==='gold')gold=null;else fracture=[];redraw()}};
fetch('/queue').then(r=>r.json()).then(d=>{queue=d.queries;load(queue.findIndex(q=>!q.saved)>=0?queue.findIndex(q=>!q.saved):0)});
</script></body></html>"""


def create_annotation_app(project_path: Path, queries_path: Path, *, expected_count: int = 30,
                          set_name: str = "hesban_30", sorted_files: bool = False,
                          invert_masks: bool = False,
                          preview_before_save: bool = False,
                          display_name: str = "Hesban queries") -> Flask:
    project_path, queries_path = Path(project_path).resolve(), Path(queries_path).resolve()
    paths = sorted(queries_path.glob("*.png"), key=lambda path: path.name.lower())
    images = ({index: path for index, path in enumerate(paths, start=1)}
              if sorted_files else {
                  _number(path): path for path in paths
                  if _number(path) <= expected_count
              })
    if sorted(images) != list(range(1, expected_count + 1)):
        raise ValueError(f"Expected Query1-Query{expected_count} PNG files in {queries_path}")
    set_path = matcher_root(project_path) / "query_sets" / f"{set_name}.json"
    seed_by_number = {}
    records_root = matcher_root(project_path) / "evaluation" / "records"
    for record_path in sorted(records_root.glob("*.json")):
        try:
            with open(record_path, encoding="utf-8") as handle:
                record = json.load(handle)
            query = record.get("query") or {}
            number = _number(Path(str(query.get("source_filename") or "")))
            query_id = str(query.get("query_id") or "")
            artifact_path = matcher_root(project_path) / "queries" / query_id / "artifact.json"
            with open(artifact_path, encoding="utf-8") as handle:
                artifact = json.load(handle)
            normalization = artifact.get("normalization") or {}
            centroid = normalization.get("centroid") or []
            scale = float(normalization.get("scale") or 0)
            fracture_normalized = (artifact.get("curves") or {}).get("fracture") or []
            fracture = [
                [float(point[0]) * scale + float(centroid[0]),
                 float(point[1]) * scale + float(centroid[1])]
                for point in fracture_normalized
            ] if len(centroid) == 2 and scale > 0 else []
            rim = (artifact.get("rim_annotation") or {}).get("source_point")
            if not rim:
                master = artifact.get("query_master_boundary") or {}
                source_points = master.get("source_points") or []
                fraction = master.get("annotated_seam_fraction")
                if source_points and fraction is not None:
                    rim = source_points[min(
                        len(source_points) - 1,
                        max(0, round(float(fraction) * (len(source_points) - 1))),
                    )]
            if 1 <= number <= expected_count and len(fracture) >= 3 and rim:
                seed_by_number[number] = {"fracture": fracture, "rim_point": rim}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    app = Flask(__name__)

    def image_value(number: int) -> Image.Image:
        with Image.open(images[number]) as source:
            if invert_masks:
                return ImageOps.invert(source.convert("L")).convert("RGBA")
            return source.convert("RGBA")

    def manifest():
        if set_path.is_file():
            with open(set_path, encoding="utf-8") as handle:
                return json.load(handle)
        return {"schema_version": 1, "name": set_name, "queries": {}}

    @app.get("/")
    def index():
        return (HTML.replace("__EXPECTED_COUNT__", str(expected_count))
                .replace("__DISPLAY_NAME__", display_name)
                .replace("__PREVIEW_BEFORE_SAVE__",
                         "true" if preview_before_save else "false")
                .replace("__ENTER_HINT__", "Enter preview, then Enter save"
                         if preview_before_save else "Enter save & next")
                .replace("__SAVE_LABEL__", "Preview outlines (Enter)"
                         if preview_before_save else "Save and next (Enter)"))

    @app.get("/queue")
    def queue():
        saved = manifest().get("queries", {})
        return jsonify({"queries": [
            {"number": n, "filename": images[n].name, "saved": str(n) in saved,
             "seeded": str(n) not in saved and n in seed_by_number,
             "fracture": (saved.get(str(n)) or seed_by_number.get(n) or {}).get("fracture", []),
             "rim_point": (saved.get(str(n)) or seed_by_number.get(n) or {}).get("rim_point")}
            for n in range(1, expected_count + 1)
        ]})

    @app.get("/image/<int:number>")
    def image(number: int):
        if number not in images:
            return jsonify(error="Unknown query"), 404
        if not invert_masks:
            return send_file(images[number], mimetype="image/png")
        payload = BytesIO()
        image_value(number).save(payload, format="PNG")
        payload.seek(0)
        return send_file(payload, mimetype="image/png")

    def traced_curves(data: dict) -> tuple[int, list, list, dict]:
        number = int(data.get("number") or 0)
        fracture, rim_point = data.get("fracture") or [], data.get("rim_point")
        if number not in images or len(fracture) < 3 or not rim_point:
            raise ValueError("Draw fracture and place gold point first")
        curves = auto_query_wall_curves_from_fracture(
            image_value(number), {"fracture": fracture, "rim_point": rim_point}
        )
        return number, fracture, rim_point, curves

    @app.post("/preview")
    def preview():
        try:
            _number_value, _fracture, _rim_point, curves = traced_curves(
                request.get_json(silent=True) or {}
            )
            return jsonify(success=True, curves={
                "exterior": curves.get("exterior") or [],
                "interior": curves.get("interior") or [],
            })
        except Exception as exc:
            return jsonify(success=False, error=str(exc)), 400

    @app.post("/save")
    def save():
        data = request.get_json(silent=True) or {}
        try:
            number, fracture, rim_point, curves = traced_curves(data)
            prepared_image = image_value(number)
            result = preprocess_query(
                project_path, prepared_image, original_filename=images[number].name,
                metadata={"query_id": f"Query {number}", "rim_diameter_cm": "", "fabric": "", "surface": "", "notes": ""},
                manual_curves=curves,
            )
            value = manifest()
            value["queries"][str(number)] = {
                "number": number, "filename": images[number].name,
                "query_id": result["query_id"], "fracture": fracture,
                "rim_point": rim_point,
            }
            _atomic_json(set_path, value)
            return jsonify(success=True, query_id=result["query_id"])
        except Exception as exc:
            return jsonify(success=False, error=str(exc)), 400

    return app
