"""Phase 2: local FastAPI web UI for the existing pothole build.py pipeline.

Run from the project folder: py -m uvicorn app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

import build

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.getenv("WEIGHTS", "weights/best.pt"))
if not WEIGHTS.is_absolute():
    WEIGHTS = ROOT / WEIGHTS
# Compatibility with the existing project, where best.pt is in the root.
if os.getenv("WEIGHTS") is None and not WEIGHTS.is_file() and (ROOT / "best.pt").is_file():
    WEIGHTS = ROOT / "best.pt"
JOBS_DIR = Path(os.getenv("JOBS_DIR", str(ROOT / "jobs"))).resolve()
FOV_DEG = float(os.getenv("FOV_DEG", "65"))
SHADOW_FRACTION = float(os.getenv("SHADOW_FRACTION", "0.6"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "20"))
JOB_TTL_HOURS = float(os.getenv("JOB_TTL_HOURS", "24"))
MAX_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
ID_RE = re.compile(r"^[0-9a-f]{32}$")
EXTS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pothole")
state_lock = threading.Lock()
active_jobs: set[str] = set()


def status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def write_status(job_dir: Path, **updates):
    with state_lock:
        target = status_path(job_dir)
        old = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        old.update(updates)
        old["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = job_dir / f".status-{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(old, indent=2), encoding="utf-8")
        os.replace(tmp, target)
        return old


def get_job_dir(job_id: str) -> Path:
    if not ID_RE.fullmatch(job_id):
        raise HTTPException(404, "Job not found")
    directory = JOBS_DIR / job_id
    if not directory.is_dir() or not status_path(directory).is_file():
        raise HTTPException(404, "Job not found")
    return directory


def public_status(job_dir: Path) -> dict:
    try:
        s = json.loads(status_path(job_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(500, "Could not read job status")
    return {k: s.get(k) for k in ("job_id", "status", "step", "volume_litres", "error", "created_at", "updated_at")}


def cleanup_old_jobs():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=JOB_TTL_HOURS)
    with state_lock:
        active_snapshot = set(active_jobs)
    for folder in JOBS_DIR.iterdir():
        if not folder.is_dir() or not ID_RE.fullmatch(folder.name) or folder.name in active_snapshot:
            continue
        try:
            info = json.loads(status_path(folder).read_text(encoding="utf-8"))
            created = datetime.fromisoformat(info["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created < cutoff:
                shutil.rmtree(folder)
        except (OSError, KeyError, ValueError, TypeError):
            pass


def run_job_worker(job_id: str, upload_path: Path, metadata: dict):
    folder = JOBS_DIR / job_id
    try:
        write_status(folder, status="running", step="Starting analysis", error=None)

        def progress(message):
            write_status(folder, step=str(message))

        result = build.run_job(
            job_dir=folder,
            image_path=upload_path,
            metadata=metadata,
            weights=WEIGHTS,
            fov_deg=FOV_DEG,
            shadow_fraction=SHADOW_FRACTION,
            progress=progress,
        )
        write_status(folder, status="done", step="Complete", volume_litres=result.get("volume_litres", []), error=None)
    except Exception as exc:
        # Include the plain-language pipeline error in the local-only UI.
        write_status(folder, status="failed", step="Stopped", error=str(exc) or type(exc).__name__)
    finally:
        with state_lock:
            active_jobs.discard(job_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not WEIGHTS.is_file():
        raise RuntimeError(f"YOLO weights not found: {WEIGHTS}. Set WEIGHTS or put best.pt beside app.py.")
    if MAX_BYTES <= 0 or JOB_TTL_HOURS <= 0 or not 0 < SHADOW_FRACTION <= 1 or not 0 < FOV_DEG < 180:
        raise RuntimeError("Invalid server environment settings")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    # Queued/running tasks cannot survive a process restart (in-memory executor).
    for folder in JOBS_DIR.iterdir():
        if not folder.is_dir() or not ID_RE.fullmatch(folder.name):
            continue
        try:
            existing = json.loads(status_path(folder).read_text(encoding="utf-8"))
            if existing.get("status") in ("queued", "running"):
                write_status(folder, status="failed", step="Interrupted", error="Server restarted during processing. Please submit the photo again.")
        except (OSError, ValueError):
            pass
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Pothole Volume Service", lifespan=lifespan)


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/api/jobs", status_code=202)
async def create_job(
    photo: UploadFile = File(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
    direction: str = Form(...),
    date: str = Form(...),
    time: str = Form(...),
    height: str = Form(...),
):
    try:
        metadata = {"latitude": float(latitude), "longitude": float(longitude), "direction": float(direction),
                    "date": date.strip(), "time": time.strip(), "height": float(height)}
        build.validate_metadata(metadata)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f"Invalid photo details: {exc}") from exc

    ext = Path(photo.filename or "").suffix.lower()
    if ext not in EXTS:
        raise HTTPException(415, "Upload a .jpg, .jpeg, or .png photo")
    cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    upload_path = job_dir / ("upload" + ext)
    try:
        size = 0
        with upload_path.open("wb") as file:
            while chunk := await photo.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"Photo exceeds {MAX_UPLOAD_MB:g} MB upload limit")
                file.write(chunk)
        if size == 0:
            raise HTTPException(415, "Uploaded photo is empty")
        try:
            with Image.open(upload_path) as image:
                if image.format != EXTS[ext]:
                    raise HTTPException(415, "Photo contents do not match the file extension")
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(415, "File is not a valid image") from exc

        now = datetime.now(timezone.utc).isoformat()
        write_status(job_dir, job_id=job_id, status="queued", step="Waiting for worker", created_at=now,
                     volume_litres=[], error=None)
        with state_lock:
            active_jobs.add(job_id)
        try:
            executor.submit(run_job_worker, job_id, upload_path, metadata)
        except RuntimeError:
            with state_lock:
                active_jobs.discard(job_id)
            raise HTTPException(503, "Processing service is shutting down")
        return {"job_id": job_id, "status": "queued"}
    except Exception:
        with state_lock:
            active_jobs.discard(job_id)
        # If worker was submitted it owns the folder; otherwise remove incomplete submission.
        if not status_path(job_dir).exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await photo.close()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return public_status(get_job_dir(job_id))


def output_file(job_id: str, name: str):
    folder = get_job_dir(job_id)
    if public_status(folder)["status"] != "done":
        raise HTTPException(409, "Report not available: job is not complete")
    target = folder / name
    if not target.is_file():
        raise HTTPException(404, "Output file missing")
    media = "application/pdf" if name.endswith("pdf") else "text/csv"
    return FileResponse(target, media_type=media, filename=f"pothole_{job_id[:8]}_{name}")


@app.get("/api/jobs/{job_id}/report.pdf")
def report(job_id: str):
    return output_file(job_id, "report.pdf")


@app.get("/api/jobs/{job_id}/result.csv")
def csv(job_id: str):
    return output_file(job_id, "result.csv")


HTML = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pothole Volume Estimator</title><style>
*{box-sizing:border-box}body{font:16px system-ui,Arial;background:#f3f6fa;color:#152536;margin:0}
main{max-width:700px;margin:45px auto;padding:0 18px}section{background:white;border:1px solid #e4e9ee;border-radius:14px;padding:28px;box-shadow:0 8px 25px #2333  }
h1{font-size:27px;margin:0 0 8px}.help{color:#596878;margin:0 0 26px}label{display:block;font-weight:600;font-size:14px;margin:14px 0 6px}
input{width:100%;padding:11px;border:1px solid #c7d0da;border-radius:7px;font:inherit}input[type=file]{padding:10px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:0 16px}button{margin-top:24px;background:#125bb5;color:white;border:0;padding:13px 22px;border-radius:8px;font-weight:700;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
#results{margin-top:24px;padding:16px;background:#f3f7fd;border-radius:8px;display:none}#links a{display:inline-block;margin:12px 16px 0 0;color:#1056ac}#err{color:#a12626;white-space:pre-wrap}
@media(max-width:550px){.grid{grid-template-columns:1fr}section{padding:20px}}
</style></head><body><main><section><h1>Pothole Volume Estimator</h1>
<p class="help">Upload a top-down pothole photo to generate a PDF report with estimated volume in litres.</p>
<form id="form"><label for="photo">Pothole photo (JPG or PNG, up to 20 MB by default)</label><input required type="file" id="photo" name="photo" accept=".jpg,.jpeg,.png,image/jpeg,image/png">
<div class="grid"><div><label for="latitude">Latitude</label><input required id="latitude" name="latitude" type="number" step="any" placeholder="23.02"></div>
<div><label for="longitude">Longitude</label><input required id="longitude" name="longitude" type="number" step="any" placeholder="72.57"></div>
<div><label for="direction">True North angle in photo (degrees)</label><input required id="direction" name="direction" type="number" step="any" value="90"><small>Measured anticlockwise from image right; up = 90°. Not camera compass heading.</small></div>
<div><label for="height">Camera capture height (metres)</label><input required id="height" name="height" type="number" step="any" min="0.001" placeholder="1.5"></div>
<div><label for="date">Date of photo</label><input required id="date" name="date" type="date"></div>
<div><label for="time">Time of photo</label><input required id="time" name="time" type="time"></div></div>
<button id="submit" type="submit">Calculate volume</button></form>
<div id="results" role="status" aria-live="polite"><strong id="status"></strong><p id="step"></p><div id="volumes"></div><p id="err"></p><div id="links"></div></div></section></main>
<script>
const form=document.getElementById('form'),box=document.getElementById('results'),btn=document.getElementById('submit');
let timer=null;
function show(){box.style.display='block'}
function fail(msg){show();document.getElementById('status').textContent='Error';document.getElementById('err').textContent=msg;btn.disabled=false}
async function parse(res){let data=await res.json().catch(()=>({}));if(!res.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||'Request failed'));return data}
async function poll(id){try{let data=await parse(await fetch('/api/jobs/'+id));show();document.getElementById('status').textContent='Status: '+data.status;
document.getElementById('step').textContent=data.step||'';
if(data.status==='done'){btn.disabled=false;document.getElementById('volumes').textContent='Estimated volume(s): '+(data.volume_litres||[]).map(v=>v+' L').join(', ');
document.getElementById('links').innerHTML='<a href="/api/jobs/'+id+'/report.pdf">Download PDF</a><a href="/api/jobs/'+id+'/result.csv">Download CSV</a>';}
else if(data.status==='failed'){fail(data.error||'Processing failed');}
else {timer=setTimeout(()=>poll(id),2000)}
}catch(e){fail(e.message)}}
form.addEventListener('submit',async e=>{e.preventDefault();clearTimeout(timer);btn.disabled=true;show();document.getElementById('status').textContent='Uploading photo...';
document.getElementById('step').textContent='';document.getElementById('err').textContent='';document.getElementById('links').innerHTML='';document.getElementById('volumes').textContent='';
const values=new FormData(form);const date=values.get('date');if(date){let p=date.split('-');values.set('date',p[2]+'-'+p[1]+'-'+p[0]);}
try{let data=await parse(await fetch('/api/jobs',{method:'POST',body:values}));poll(data.job_id);}catch(e){fail(e.message)}});
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
